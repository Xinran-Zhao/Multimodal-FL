"""
Multimodal MIMIC-CXR Dataset
Loads paired (X-ray image, Findings text) for multimodal federated learning.
"""

import os
import re
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import pydicom
import cv2
from transformers import AutoTokenizer


class MIMICCXRMultimodal(Dataset):
    """
    Multimodal dataset that returns paired (image_tensor, text_tokens)
    from MIMIC-CXR for federated learning research.

    Args:
        data_root:    Path to MIMIC-CXR files directory
                      e.g. "/data/amciilab/mahfuz/mimic_cxr_dl_script/
                            physionet.org/files/mimic-cxr/2.0.0/files"
        split:        "train", "val", or "test" (requires split CSV) or None for all
        tokenizer_name: HuggingFace tokenizer name for text encoding
        max_text_len: Maximum token length for text
        image_size:   Target image size (square)
        use_clahe:    Whether to apply CLAHE contrast enhancement
        augment:      Whether to apply training augmentations
        missing_findings: How to handle reports without FINDINGS section
                          "skip" = exclude, "impression" = use IMPRESSION instead,
                          "full" = use full report text
    """

    def __init__(
        self,
        data_root: str,
        split: str = None,
        tokenizer_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_text_len: int = 512,
        image_size: int = 224,
        use_clahe: bool = True,
        augment: bool = False,
        missing_findings: str = "skip",
    ):
        super().__init__()
        self.data_root = data_root
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.augment = augment
        self.missing_findings = missing_findings

        # --- Text tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # --- Image transforms ---
        if augment:
            self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomRotation(5),
                transforms.CenterCrop(230),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        # --- Build index of (dicom_path, report_path) pairs ---
        self.samples = self._build_index()
        print(f"[MIMICCXRMultimodal] Loaded {len(self.samples)} image-text pairs "
              f"(missing_findings='{missing_findings}')")

    def _build_index(self):
        """Scan the directory tree and pair each DICOM with its study report."""
        samples = []

        # Iterate over patient group dirs: p10, p11, ..., p19
        patient_group_dirs = sorted(glob.glob(os.path.join(self.data_root, "p1[0-9]")))

        for pg_dir in patient_group_dirs:
            # Each patient dir: p10000032, p10000764, ...
            for patient_dir in sorted(os.listdir(pg_dir)):
                patient_path = os.path.join(pg_dir, patient_dir)
                if not os.path.isdir(patient_path):
                    continue

                # Each study dir: s50414267, s53189527, ...
                for entry in sorted(os.listdir(patient_path)):
                    study_path = os.path.join(patient_path, entry)

                    # Study directories contain DICOM files
                    if os.path.isdir(study_path) and entry.startswith("s"):
                        report_path = os.path.join(patient_path, entry + ".txt")
                        if not os.path.isfile(report_path):
                            continue

                        # Check if findings exist
                        text = self._extract_text(report_path)
                        if text is None:
                            continue

                        # Collect all DICOMs in this study
                        dcm_files = [
                            os.path.join(study_path, f)
                            for f in os.listdir(study_path)
                            if f.endswith(".dcm")
                        ]

                        # One sample per DICOM image, all sharing the same report
                        for dcm_path in dcm_files:
                            samples.append((dcm_path, report_path, text))

        return samples

    def _extract_text(self, report_path):
        """
        Extract the text target from a radiology report.
        Returns cleaned text string, or None if should be skipped.
        """
        with open(report_path, "r") as f:
            report = f.read()

        # Try to extract FINDINGS section
        match = re.search(
            r"FINDINGS\s*:(.*?)(?:IMPRESSION\s*:|$)", report, re.DOTALL
        )

        if match:
            findings = match.group(1).strip()
            # Skip placeholder-only findings (e.g., just "___")
            cleaned = re.sub(r"_+", "", findings).strip()
            if len(cleaned) == 0:
                # Findings is just placeholders, treat as missing
                return self._handle_missing(report)
            return self._clean_text(findings)
        else:
            return self._handle_missing(report)

    def _handle_missing(self, report):
        """Handle reports without a usable FINDINGS section."""
        if self.missing_findings == "skip":
            return None
        elif self.missing_findings == "impression":
            match = re.search(
                r"IMPRESSION\s*:(.*?)$", report, re.DOTALL
            )
            if match:
                return self._clean_text(match.group(1).strip())
            return None
        elif self.missing_findings == "full":
            # Use everything after INDICATION/HISTORY
            match = re.search(
                r"(?:INDICATION|HISTORY|REASON FOR EXAM)\s*:.*?\n(.*)",
                report, re.DOTALL,
            )
            if match:
                return self._clean_text(match.group(1).strip())
            return self._clean_text(report.strip())
        return None

    @staticmethod
    def _clean_text(text):
        """Clean report text: normalize whitespace, handle de-id tokens."""
        text = re.sub(r"_+", "[DEIDENTIFIED]", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) > 0 else None

    # ----- Image loading -----

    def _load_dicom(self, dcm_path):
        """Load a DICOM file and return a PIL Image (grayscale)."""
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)

        # Handle inverted images (MONOCHROME1 = white bones on dark background)
        if hasattr(dcm, "PhotometricInterpretation"):
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                img = img.max() - img

        # Normalize to 0-255 range
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
        img = img.astype(np.uint8)

        # Optional: CLAHE contrast enhancement
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

        return Image.fromarray(img, mode="L")

    # ----- Main interface -----

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path, report_path, text = self.samples[idx]

        # --- Image ---
        image = self._load_dicom(dcm_path)
        image_tensor = self.image_transform(image)

        # --- Text ---
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)          # (max_text_len,)
        attention_mask = encoding["attention_mask"].squeeze(0) # (max_text_len,)

        return {
            "image": image_tensor,            # (3, 224, 224)
            "input_ids": input_ids,           # (512,)
            "attention_mask": attention_mask,  # (512,)
            "dcm_path": dcm_path,
            "report_path": report_path,
        }


class MIMICCXRPreprocessed(Dataset):
    """
    Lightweight dataset that loads from preprocessed output of preprocess.py.
    Much faster than MIMICCXRMultimodal since images are already PNG and
    text is already extracted in manifest.csv.

    Args:
        preprocessed_dir: Path to preprocess.py output directory
                          (contains images/ and manifest.csv)
        tokenizer_name:   HuggingFace tokenizer name
        max_text_len:     Maximum token length
        image_size:       Target image size (should match preprocess.py --image_size)
        augment:          Whether to apply training augmentations
    """

    def __init__(
        self,
        preprocessed_dir: str,
        tokenizer_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_text_len: int = 512,
        image_size: int = 224,
        augment: bool = False,
    ):
        super().__init__()
        self.preprocessed_dir = preprocessed_dir
        self.image_dir = os.path.join(preprocessed_dir, "images")
        self.max_text_len = max_text_len

        # Load manifest
        import csv
        self.samples = []
        manifest_path = os.path.join(preprocessed_dir, "manifest.csv")
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Image transforms (images are already grayscale PNGs at target size)
        if augment:
            self.image_transform = transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomCrop(image_size, padding=16),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])

        print(f"[MIMICCXRPreprocessed] Loaded {len(self.samples)} samples "
              f"from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        # --- Image ---
        img_path = os.path.join(self.image_dir, row["image_path"])
        image = Image.open(img_path).convert("L")
        image_tensor = self.image_transform(image)

        # --- Text ---
        encoding = self.tokenizer(
            row["findings_text"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "patient_id": row["patient_id"],
            "study_id": row["study_id"],
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--preprocessed":
        # Test with preprocessed data
        dataset = MIMICCXRPreprocessed(
            preprocessed_dir="/data/amciilab/xinran/mimic_cxr_preprocessed",
            augment=False,
        )
    else:
        # Test with raw DICOM data
        DATA_ROOT = (
            "/data/amciilab/mahfuz/mimic_cxr_dl_script/"
            "physionet.org/files/mimic-cxr/2.0.0/files"
        )
        dataset = MIMICCXRMultimodal(
            data_root=DATA_ROOT,
            image_size=224,
            use_clahe=True,
            augment=False,
            missing_findings="skip",
        )

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape:  {sample['image'].shape}")
        print(f"Input IDs:    {sample['input_ids'].shape}")
        print(f"Attn mask:    {sample['attention_mask'].shape}")
    else:
        print("No samples found.")
