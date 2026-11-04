"""
backend.training.dataset_loader

Safe loading, parsing, and storage management for LexiRep training datasets.
Enforces:
- Strict np.load(..., allow_pickle=False)
- Mandatory keys: X_tr, Y_tr, W_tr (no silent substitutions)
- Rejection of malformed, non-npz, and oversized (>250 MB) files
- UUIDv4 dataset storage sandboxing under configured training root
- Directory traversal guards
"""

import os
import re
import json
import uuid
import datetime
import zipfile
from typing import Dict, Any, Tuple, Optional
import numpy as np

from backend.training.validator import validate_dataset, DatasetValidationResult

UUID4_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)
MAX_DATASET_UPLOAD_BYTES = 250 * 1024 * 1024  # 250 MB


class DatasetLoaderError(ValueError):
    """Raised when dataset loading or structural validation fails."""
    pass


class DatasetTooLargeError(ValueError):
    """Raised when uploaded dataset exceeds the 250 MB upload limit."""
    pass


def load_dataset_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Safely load a LexiRep dataset archive (.npz) with allow_pickle=False.

    Args:
        path: Filesystem path to the .npz archive.

    Returns:
        Dictionary containing at least 'X_tr', 'Y_tr', 'W_tr' as numpy arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetLoaderError: If the archive is corrupt, not a zip, or missing required keys.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if not path.lower().endswith(".npz"):
        raise DatasetLoaderError("Dataset must be a NumPy archive file with .npz extension.")

    # Validate that it is a valid zip archive without loading pickles
    if not zipfile.is_zipfile(path):
        raise DatasetLoaderError("File is not a valid zip/npz archive.")

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        raise DatasetLoaderError(f"Failed to load .npz archive safely (allow_pickle=False): {str(e)}")

    required_keys = ["X_tr", "Y_tr", "W_tr"]
    missing = [k for k in required_keys if k not in data.files]
    if missing:
        raise DatasetLoaderError(
            f"Dataset archive missing mandatory keys: {missing}. Found keys: {data.files}."
        )

    X_tr = data["X_tr"]
    Y_tr = data["Y_tr"]
    W_tr = data["W_tr"]

    if not isinstance(X_tr, np.ndarray) or not isinstance(Y_tr, np.ndarray) or not isinstance(W_tr, np.ndarray):
        raise DatasetLoaderError("Keys 'X_tr', 'Y_tr', and 'W_tr' must all be NumPy ndarrays.")

    # Check matching lengths
    if len(X_tr) != len(Y_tr) or len(X_tr) != len(W_tr):
        raise DatasetLoaderError(
            f"Mismatched array lengths: X_tr={len(X_tr)}, Y_tr={len(Y_tr)}, W_tr={len(W_tr)}."
        )

    result: Dict[str, Any] = {
        "X_tr": X_tr,
        "Y_tr": Y_tr,
        "W_tr": W_tr,
    }

    # Extract optional safe metadata if present (must be 1D string/int array or safe types)
    for opt_k in ["word_text", "words_text", "X_te", "Y_te", "W_te"]:
        if opt_k in data.files:
            result[opt_k] = data[opt_k]

    return result


class DatasetRegistry:
    """
    Manages sandboxed storage of uploaded datasets under backend/training_jobs/datasets/.
    """

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    def _validate_id(self, dataset_id: str) -> str:
        if not UUID4_PATTERN.match(dataset_id):
            raise ValueError(f"Invalid dataset ID '{dataset_id}'. Must be a valid UUIDv4.")
        return dataset_id

    def get_dataset_dir(self, dataset_id: str) -> str:
        self._validate_id(dataset_id)
        d_dir = os.path.abspath(os.path.join(self.root_dir, dataset_id))
        if os.path.commonpath([d_dir, self.root_dir]) != self.root_dir:
            raise ValueError("Path traversal detected in dataset ID.")
        return d_dir

    def get_dataset_path(self, dataset_id: str) -> str:
        d_dir = self.get_dataset_dir(dataset_id)
        npz_path = os.path.join(d_dir, "dataset.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_id}")
        return npz_path

    def get_metadata_path(self, dataset_id: str) -> str:
        d_dir = self.get_dataset_dir(dataset_id)
        meta_path = os.path.join(d_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Metadata not found for dataset: {dataset_id}")
        return meta_path

    def get_metadata(self, dataset_id: str) -> Dict[str, Any]:
        meta_path = self.get_metadata_path(dataset_id)
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_dataset(
        self,
        content: bytes,
        original_filename: str,
    ) -> Tuple[str, str, DatasetValidationResult]:
        """
        Persist uploaded dataset safely and validate contents.

        Args:
            content: Raw bytes of uploaded file.
            original_filename: Client-supplied filename for diagnostics.

        Returns:
            Tuple of (dataset_id, dataset_path, ValidationResult).

        Raises:
            DatasetTooLargeError: If content > 250 MB.
            DatasetLoaderError: If format, extension, or keys are invalid.
        """
        if len(content) > MAX_DATASET_UPLOAD_BYTES:
            raise DatasetTooLargeError(
                f"Dataset size ({len(content)} bytes) exceeds maximum limit of {MAX_DATASET_UPLOAD_BYTES} bytes (250 MB)."
            )

        # Disallow non-npz files (such as .pt, .pkl, .exe, etc.)
        base_name = os.path.basename(original_filename or "dataset.npz")
        if not base_name.lower().endswith(".npz"):
            raise DatasetLoaderError("Uploaded file must have a .npz extension.")

        # Check zip magic bytes (PK\x03\x04)
        if len(content) < 4 or content[:2] != b"PK":
            raise DatasetLoaderError("Uploaded file is not a valid zip/npz archive.")

        dataset_id = str(uuid.uuid4())
        dataset_dir = self.get_dataset_dir(dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)

        npz_path = os.path.join(dataset_dir, "dataset.npz")
        tmp_npz_path = os.path.join(dataset_dir, "dataset.npz.tmp")

        try:
            with open(tmp_npz_path, "wb") as f:
                f.write(content)
            os.replace(tmp_npz_path, npz_path)
        except Exception as e:
            if os.path.isfile(tmp_npz_path):
                try:
                    os.remove(tmp_npz_path)
                except Exception:
                    pass
            raise IOError(f"Failed to write dataset to disk: {str(e)}")

        # Validate through safe loader
        data = load_dataset_npz(npz_path)
        val_res = validate_dataset(data["X_tr"], data["Y_tr"], data["W_tr"])

        # Save metadata
        metadata = {
            "dataset_id": dataset_id,
            "original_filename": base_name,
            "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "file_size_bytes": len(content),
            "validation": val_res.to_dict(),
        }
        meta_path = os.path.join(dataset_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return dataset_id, npz_path, val_res
