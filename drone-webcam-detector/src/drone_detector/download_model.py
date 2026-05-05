"""Download the pretrained drone detector checkpoint from Hugging Face."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "doguilmak/Drone-Detection-YOLOv8x"
MODEL_FILENAME = "weight/best.pt"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"


def _display_path(path: Path) -> str:
    """Return a readable path relative to the project root when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def download_pretrained_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the pretrained drone detector checkpoint if it is missing."""
    destination = Path(model_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"Model already exists at {_display_path(destination)}. Skipping download.")
        return destination

    print(f"Downloading {MODEL_FILENAME} from Hugging Face repo {MODEL_REPO_ID}...")
    try:
        cached_model = Path(
            hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        )
    except Exception as exc:  # pragma: no cover - depends on network/service state.
        raise RuntimeError(
            f"Failed to download {MODEL_FILENAME} from {MODEL_REPO_ID}: {exc}"
        ) from exc

    try:
        shutil.copy2(cached_model, destination)
    except OSError as exc:
        raise RuntimeError(
            f"Downloaded checkpoint but failed to copy it to "
            f"{_display_path(destination)}: {exc}"
        ) from exc

    print(f"Saved pretrained checkpoint to {_display_path(destination)}.")
    return destination


def main() -> int:
    """Run the model download from the command line."""
    try:
        download_pretrained_model()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
