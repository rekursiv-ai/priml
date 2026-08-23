"""The game's sprites, fetched on demand rather than checked in.

143 PNGs, 624 KB. They are not source: they are a fixed upstream artifact that
never changes between releases and that nothing but the viewer reads. Vendoring
them would put most of a package's file-size budget into pixels that no test
asserts on and no training run opens.

So they are downloaded once, into the per-user cache, and verified by digest.
The digest is the part that matters: a texture silently swapped upstream would
otherwise change every rendered frame with nothing to notice it.

Craftax is MIT-licensed, which permits redistribution of these files; the
choice not to redistribute is about package weight, not permission.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import hashlib
import os
import urllib.error
import urllib.request

from priml.lib.userdirs import cache_dir


def asset_dir(*, revision: str = "v1.6.1") -> Path:
    """Return the directory the sprites are cached in.

    Args:
      revision: Upstream release the sprites come from. Matches the
        ``craftax>=1.6.1`` dependency the parity tests compare against, so the
        pixels and the rules come from one version of the game.

    Returns:
      directory: Per-user cache path for this revision's sprites.

    """
    return cache_dir() / "rekursiv-ai" / "craftax" / "assets" / revision


def fetch(
    name: str,
    *,
    directory: Path | None = None,
    revision: str = "v1.6.1",
    url_template: str = (
        "https://raw.githubusercontent.com/MichaelTMatthews/Craftax/"
        "{revision}/craftax/craftax/assets/{name}"
    ),
) -> Path:
    """Return a local path to one sprite, downloading it if absent.

    Args:
      name: File name, for example ``"zombie.png"``.
      directory: Cache directory; defaults to :func:`asset_dir`.
      revision: Upstream release to fetch from. A tag rather than a branch:
        ``main`` would let a texture change under a cache that has no reason
        to re-fetch it.
      url_template: Where one sprite lives, given a revision and a name.

    Returns:
      path: The cached file.

    Raises:
      RuntimeError: The download failed.

    """
    directory = directory or asset_dir(revision=revision)
    path = directory / name
    if path.exists():
        return path

    directory.mkdir(parents=True, exist_ok=True)
    url = url_template.format(revision=revision, name=name)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            payload = cast(bytes, response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not download the Craftax sprite {name}") from error

    # Written through a temporary path: an interrupted download must not leave
    # a truncated PNG that every later run treats as cached.
    #
    # The name carries the writer's pid because several processes fetch into
    # one shared cache -- pytest-xdist workers do exactly this. A fixed
    # ``.partial`` name means one worker renames the file while another is
    # still writing to it, and the second rename raises FileNotFoundError.
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    partial.write_bytes(payload)
    # ``replace`` is atomic on POSIX, so a reader either sees the old file or
    # the whole new one -- never a half-written PNG.
    partial.replace(path)
    return path


def digest(directory: Path | None = None) -> str:
    """Return one hash over every cached sprite.

    Args:
      directory: Cache directory; defaults to :func:`asset_dir`.

    Returns:
      digest: Hex SHA-256 over the sorted name/content pairs.

    """
    directory = directory or asset_dir()
    accumulator = hashlib.sha256()
    for path in sorted(directory.glob("*.png")):
        accumulator.update(path.name.encode())
        accumulator.update(path.read_bytes())
    return accumulator.hexdigest()
