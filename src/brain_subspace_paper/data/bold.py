from __future__ import annotations

from pathlib import Path


class AnnexedContentMissingError(RuntimeError):
    """Raised when a dataset clone contains annex placeholders instead of content."""


def count_nifti_volumes(path: Path) -> int:
    import nibabel as nib

    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic != b"\x1f\x8b":
        raise AnnexedContentMissingError(
            "Annexed content has not been fetched for "
            f"{path}. Install datalad or git-annex, then fetch annotation, stimuli, and derivatives."
        )

    image = nib.load(str(path))
    shape = image.shape
    if len(shape) == 3:
        return 1
    if len(shape) >= 4:
        return int(shape[3])
    raise ValueError(f"Unexpected NIfTI shape for {path}: {shape}")
