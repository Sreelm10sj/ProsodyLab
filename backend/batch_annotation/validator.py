"""
backend.batch_annotation.validator

Security validation and extraction routines for batch audio ZIP archives.
Enforces limits, path-traversal prevention, format filtering, and filename sanitization.
"""

import os
import re
import zipfile
from typing import List, Dict, Any, Tuple

# Configurable limits
MAX_BATCH_ZIP_BYTES = 250 * 1024 * 1024       # 250 MB
MAX_AUDIO_FILES = 2000                        # Max 2000 audio files per batch
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024   # 1 GB decompression limit
MAX_DECOMPRESSION_RATIO = 100.0               # Zip-bomb protection

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_\-\.]")


class BatchValidationError(Exception):
    """Raised when a batch ZIP archive fails validation."""
    pass


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to only contain alphanumeric characters, underscores, hyphens, and dots.
    Leading dots or dashes are stripped to prevent hidden files or CLI flag confusion.
    """
    name = os.path.basename(filename).strip()
    name = SAFE_NAME_PATTERN.sub("_", name)
    name = name.lstrip(".-")
    if not name:
        name = "audio_file"
    return name


def normalize_archive_path(archive_entry: str) -> str:
    """
    Convert an internal archive path (e.g. 'speaker1/subfolder/audio.wav')
    into a flat, sanitized identifier ('speaker1_subfolder_audio.wav').
    """
    # Normalize slashes and remove leading/trailing slashes
    clean_path = archive_entry.replace("\\", "/").strip("/")
    parts = [p for p in clean_path.split("/") if p and p != "."]
    sanitized_parts = [sanitize_filename(p) for p in parts]
    return "_".join(sanitized_parts)


def get_json_filename_for_audio(audio_filename: str) -> str:
    """
    Given an audio filename (e.g. 'sample_01.wav' or 'speech.test.mp3'),
    strip ONLY the audio extension and replace with '.json'.
    """
    base, ext = os.path.splitext(audio_filename)
    clean_base = sanitize_filename(base)
    return f"{clean_base}.json"


def validate_zip_archive(zip_path: str) -> List[Dict[str, Any]]:
    """
    Inspect a ZIP file without extracting, validating size, contents, extensions,
    and security constraints.

    Returns:
        List of dicts representing valid audio file entries:
        [
            {
                "archive_name": str,       # Original name in zip
                "normalized_name": str,    # Clean flat name for input/
                "output_name": str,        # Clean JSON name for output/
                "file_size": int,          # Uncompressed size in bytes
            },
            ...
        ]

    Raises:
        BatchValidationError: If archive is invalid, oversized, empty, or violates security policies.
    """
    if not os.path.isfile(zip_path):
        raise BatchValidationError(f"Archive file not found: {zip_path}")

    archive_size = os.path.getsize(zip_path)
    if archive_size == 0:
        raise BatchValidationError("The uploaded archive is empty (0 bytes).")

    if archive_size > MAX_BATCH_ZIP_BYTES:
        raise BatchValidationError(
            f"Archive exceeds maximum size limit of {MAX_BATCH_ZIP_BYTES // (1024 * 1024)} MB. (Received: {archive_size / (1024 * 1024):.1f} MB)"
        )

    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as e:
        raise BatchValidationError(f"Invalid or corrupted ZIP archive: {str(e)}")

    valid_audio_entries: List[Dict[str, Any]] = []
    total_uncompressed_bytes = 0
    used_normalized_names: Dict[str, int] = {}

    with zf:
        # Check for corrupt zip structures
        test_result = zf.testzip()
        if test_result is not None:
            raise BatchValidationError(f"Corrupted file in ZIP archive: {test_result}")

        infolist = zf.infolist()
        if not infolist:
            raise BatchValidationError("ZIP archive contains no files.")

        for info in infolist:
            # 1. Check for symlinks
            # In Unix zip archives, symlinks have file mode S_IFLNK (0o120000) in upper 16 bits of external_attr
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise BatchValidationError(f"Symbolic links are prohibited in archive: {info.filename}")

            # 2. Path traversal security checks
            entry_name = info.filename
            if ".." in entry_name.replace("\\", "/").split("/"):
                raise BatchValidationError(f"Path traversal detected in archive entry: {entry_name}")

            if entry_name.startswith("/") or entry_name.startswith("\\"):
                raise BatchValidationError(f"Absolute path detected in archive entry: {entry_name}")

            # Check for Windows drive letters (e.g. C:foo)
            if re.match(r"^[a-zA-Z]:", entry_name):
                raise BatchValidationError(f"Absolute drive path detected in archive entry: {entry_name}")

            # Skip directories
            if info.is_dir() or entry_name.endswith("/") or entry_name.endswith("\\"):
                continue

            # 3. Size and compression ratio checks (zip-bomb guard)
            total_uncompressed_bytes += info.file_size
            if total_uncompressed_bytes > MAX_UNCOMPRESSED_BYTES:
                raise BatchValidationError(
                    f"Total uncompressed content exceeds limit of {MAX_UNCOMPRESSED_BYTES // (1024 * 1024)} MB."
                )

            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_DECOMPRESSION_RATIO and info.file_size > 10 * 1024 * 1024:
                    raise BatchValidationError(
                        f"Unreasonable compression ratio ({ratio:.1f}x) in entry: {entry_name}"
                    )

            # 4. Check extension
            _, ext = os.path.splitext(entry_name)
            ext_lower = ext.lower()

            # If it's not a supported audio extension, we ignore non-audio files or reject executables
            if ext_lower in {".exe", ".bat", ".cmd", ".sh", ".py", ".dll", ".so", ".bin"}:
                raise BatchValidationError(f"Prohibited executable file found in archive: {entry_name}")

            if ext_lower not in ALLOWED_AUDIO_EXTENSIONS:
                # Unsupported non-executable file (e.g. .txt, .pdf, .ds_store)
                continue

            # 5. Build normalized unique name
            base_norm = normalize_archive_path(entry_name)
            name_root, ext_part = os.path.splitext(base_norm)

            if base_norm in used_normalized_names:
                used_normalized_names[base_norm] += 1
                suffix = used_normalized_names[base_norm]
                norm_name = f"{name_root}_{suffix}{ext_part}"
            else:
                used_normalized_names[base_norm] = 1
                norm_name = base_norm

            json_name = get_json_filename_for_audio(norm_name)

            valid_audio_entries.append({
                "archive_name": entry_name,
                "normalized_name": norm_name,
                "output_name": json_name,
                "file_size": info.file_size,
            })

    if not valid_audio_entries:
        raise BatchValidationError(
            f"No valid audio files found in archive. Supported formats: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    if len(valid_audio_entries) > MAX_AUDIO_FILES:
        raise BatchValidationError(
            f"Archive contains {len(valid_audio_entries)} audio files, exceeding the maximum limit of {MAX_AUDIO_FILES}."
        )

    return valid_audio_entries


def extract_batch_audio_files(
    zip_path: str,
    valid_entries: List[Dict[str, Any]],
    destination_dir: str,
) -> None:
    """
    Safely extracts only the validated audio files into the destination directory,
    saving each file under its normalized name.

    Args:
        zip_path: Path to the validated ZIP archive.
        valid_entries: The list of validated entry metadata returned by validate_zip_archive.
        destination_dir: Directory where audio files will be written.
    """
    os.makedirs(destination_dir, exist_ok=True)
    dest_abs = os.path.abspath(destination_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for item in valid_entries:
            archive_name = item["archive_name"]
            normalized_name = item["normalized_name"]
            target_path = os.path.abspath(os.path.join(destination_dir, normalized_name))

            # Ensure target stays strictly inside destination_dir
            if not target_path.startswith(dest_abs + os.sep) and target_path != dest_abs:
                raise BatchValidationError(f"Target path escapes destination sandbox: {normalized_name}")

            with zf.open(archive_name) as source, open(target_path, "wb") as target:
                # Stream copy in 64KB chunks
                while True:
                    chunk = source.read(65536)
                    if not chunk:
                        break
                    target.write(chunk)
