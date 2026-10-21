"""Prepare the pinned Chronos files used by exp040.

Downloads use the Hugging Face Hub cache. Verified model copies and their
manifests are installed atomically at the locations declared by each recipe.
"""

from pathlib import Path

import argparse
import hashlib
import json
import logging
import shutil
import tempfile

from configgle import traverse
from treekle import to_plain

from priml.baselines.oscillator.experiments import exp040
from priml.baselines.oscillator.pretrained import AssetManifest, ModelFiles


try:
    from huggingface_hub import hf_hub_download
except ImportError as error:
    raise ImportError("Install priml[hub] to prepare pretrained assets.") from error


def main(argv: list[str] | None = None) -> int:
    """Prepare each distinct declared model location.

    Args:
      argv: Command arguments; None reads the process arguments.

    Returns:
      status: Zero after all declared model locations have been prepared.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    parser.parse_args(argv)
    prepared: set[tuple[str, str, Path]] = set()
    for match in traverse(exp040(), ModelFiles.Config):
        config = match.config
        key = config.model_id, config.revision, config.directory.expanduser().resolve()
        if key not in prepared:
            prepare_model(config)
            prepared.add(key)
    return 0


def prepare_model(config: ModelFiles.Config) -> None:
    """Download a pinned model atomically, or verify its existing local files.

    Args:
      config: Pinned model identity, revision, and local destination.

    """
    directory = config.directory.expanduser().resolve()
    if directory.exists():
        config.make()
        return
    if len(config.revision) != 40 or any(
        character not in "0123456789abcdef" for character in config.revision
    ):
        raise ValueError("Model revisions must be full hexadecimal commit identities.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-",
        dir=directory.parent,
    ) as temporary:
        staged = Path(temporary)
        digests: dict[str, str] = {}
        for name in ("config.json", "model.safetensors"):
            source = hf_hub_download(config.model_id, name, revision=config.revision)
            path = staged / name
            shutil.copyfile(source, path)
            with path.open("rb") as stream:
                digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        record: AssetManifest = {
            "model_id": config.model_id,
            "revision": config.revision,
            "files": digests,
        }
        (staged / "source.json").write_text(
            json.dumps(to_plain(record), indent=2, allow_nan=False) + "\n",
        )
        staged.rename(directory)
    config.make()
    logging.getLogger(__name__).info("Prepared %s at %s", config.model_id, directory)
