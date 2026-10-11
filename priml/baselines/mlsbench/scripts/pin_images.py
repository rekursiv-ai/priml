"""Pin upstream Harbor base images to content digests in ``image_lock.json``.

Finds every ``FROM bohanlyu2022/mlsbench-harbor-<pkg>[:tag]`` line under
``<clone>/harbor/tasks-docker/*/environment/Dockerfile``, pulls the tag,
reads ``RepoDigests`` from ``docker image inspect`` and merges
``<pkg> -> repo@sha256:...`` into the lock. Existing entries are kept unless
``--repin`` is given, so a partial lock stays partial and honest. Only the
docker CLI (subprocess) and the stdlib are used; nothing upstream is copied.

Examples::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.pin_images --list
    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.pin_images --package gplearn --package deap
    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.pin_images --all  # ~80 GB of pulls
    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.pin_images --registry --all  # digests only

The clone is read from ``/opt/scratch/mlsbench`` unless ``--root`` says otherwise.

``--registry`` asks the Docker Hub registry API (stdlib ``urllib``; anonymous
pull token from auth.docker.io, then ``GET /v2/<repo>/manifests/<tag>``) and
records the ``Docker-Content-Digest`` of the top-level manifest (the
manifest list / index when multi-platform, else the single manifest). It
also writes per-platform digests to ``image_platforms.json``. Network access
happens only in this CLI path; tests inject a fake fetcher.
"""

# ruff: noqa: S603

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from priml.baselines.mlsbench.scripts.prepare_data import UPSTREAM_DIR, default_path


if TYPE_CHECKING:
    from http.client import HTTPResponse


_TESTDATA: Final = Path(__file__).resolve().parent.parent / "testdata"
DEFAULT_LOCK: Final = _TESTDATA / "image_lock.json"
DEFAULT_PLATFORMS: Final = _TESTDATA / "image_platforms.json"
_AUTH_URL: Final = "https://auth.docker.io/token"
_REGISTRY_URL: Final = "https://registry-1.docker.io/v2"
_ACCEPT: Final = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ),
)
_INDEX_TYPES: Final = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    },
)
BASE_PREFIX: Final = "bohanlyu2022/mlsbench-harbor-"
_FROM = re.compile(
    r"^\s*FROM\s+(?:--\S+\s+)*(" + re.escape(BASE_PREFIX) + r"(?P<pkg>[^\s:@]+)"
    r"(?P<ref>[:@]\S+)?)(?:\s+AS\s+\S+)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]
# (url, headers) -> (status, response headers with lower-case keys, body)
Fetcher = Callable[[str, dict[str, str]], tuple[int, dict[str, str], bytes]]


class PinError(RuntimeError):
    """A pull or inspect failed, or the inputs were unusable."""


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def find_bases(root: Path) -> dict[str, str]:
    """Return ``package -> "repo:tag"`` for every Harbor base under ``root``.

    Raises:
      PinError: No Dockerfile found, or one package appears with two refs.

    """
    dockerfiles = sorted(
        (root / "harbor" / "tasks-docker").glob("*/environment/Dockerfile"),
    )
    if not dockerfiles:
        raise PinError(f"no Harbor Dockerfiles under {root}/harbor/tasks-docker")
    found: dict[str, str] = {}
    for df in dockerfiles:
        for m in _FROM.finditer(df.read_text(errors="replace")):
            pkg, ref = m["pkg"], m[1]
            if "@sha256:" not in ref and ":" not in ref[len(BASE_PREFIX) :]:
                ref += ":latest"
            if found.setdefault(pkg, ref) != ref:
                raise PinError(
                    f"package {pkg!r} has conflicting refs: {found[pkg]}, {ref}",
                )
    return found


def pin_one(ref: str, docker: str = "docker", runner: Runner = _run) -> str:
    """Pull ``ref`` and return its ``repo@sha256:...``.

    Raises:
      PinError: The pull failed or the image has no digest for that repo.

    """
    pull = runner([docker, "pull", ref])
    if pull.returncode != 0:
        raise PinError(
            f"docker pull {ref} failed: {(pull.stderr or pull.stdout).strip()[-500:]}",
        )
    insp = runner(
        [docker, "image", "inspect", "--format", "{{json .RepoDigests}}", ref],
    )
    if insp.returncode != 0:
        raise PinError(
            f"docker image inspect {ref} failed: {insp.stderr.strip()[-500:]}",
        )
    try:
        digests = cast("object", json.loads(insp.stdout))
    except json.JSONDecodeError as e:
        raise PinError(f"unparseable RepoDigests for {ref}: {insp.stdout!r}") from e
    repo = (
        ref.split("@", maxsplit=1)[0].rsplit(":", 1)[0]
        if ":" in ref.rsplit("/", 1)[-1]
        else ref
    )
    for d in _as_list(digests):
        if isinstance(d, str) and d.startswith(repo + "@sha256:"):
            return d
    raise PinError(f"{ref} has no RepoDigest for {repo} (got {digests})")


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Blob downloads redirect to a CDN that must not see the registry token."""

    @override
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
        if new is not None:
            new.remove_header("Authorization")
        return new


def urllib_fetch(
    url: str,
    headers: dict[str, str],
    *,
    attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, dict[str, str], bytes]:
    """GET ``url`` with retries on 429/5xx and network errors (CLI path only).

    Returns:
      ``(status, lower-cased headers, body)``; 4xx other than 429 are returned,
      not raised.

    Raises:
      PinError: Still failing after ``attempts`` tries.

    """
    opener = urllib.request.build_opener(_DropAuthOnRedirect)
    last = ""
    for i in range(attempts):
        req = urllib.request.Request(url, headers=headers)  # noqa: S310
        try:
            with cast("HTTPResponse", opener.open(req, timeout=30)) as r:
                return (
                    int(r.status),
                    {str(k).lower(): str(v) for k, v in r.headers.items()},
                    bytes(r.read()),
                )
        except urllib.error.HTTPError as e:
            body = e.read()
            if e.code != 429 and e.code < 500:
                return e.code, {k.lower(): v for k, v in e.headers.items()}, body
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
        if i + 1 < attempts:
            sleep(2.0**i)
    raise PinError(f"GET {url} failed after {attempts} attempts: {last}")


class Resolved:
    """Registry answer for one ``repo:tag``."""

    def __init__(self, digest: str, platforms: dict[str, str]) -> None:
        self.digest = digest
        """Digest of the top-level manifest (list/index or single manifest)."""
        self.platforms = platforms
        """``"os/arch[/variant]" -> sha256:...`` of the per-platform manifests."""


def _token(repo: str, fetch: Fetcher) -> str:
    q = urllib.parse.urlencode(
        {
            "service": "registry.docker.io",
            "scope": f"repository:{repo}:pull",
        },
    )
    status, _, body = fetch(f"{_AUTH_URL}?{q}", {})
    if status != 200:
        raise PinError(f"token request for {repo} failed: HTTP {status}")
    try:
        return str(cast("dict[str, object]", json.loads(body))["token"])
    except (ValueError, KeyError, TypeError) as e:
        raise PinError(f"token response for {repo} has no token") from e


def _get(
    repo: str,
    kind: str,
    ref: str,
    token: str,
    fetch: Fetcher,
) -> tuple[dict[str, str], bytes]:
    url = f"{_REGISTRY_URL}/{repo}/{kind}/{ref}"
    status, hdrs, body = fetch(
        url,
        {"Authorization": f"Bearer {token}", "Accept": _ACCEPT},
    )
    if status != 200:
        detail = body[:200].decode(errors="replace")
        raise PinError(f"GET {url} -> HTTP {status}: {detail}")
    return hdrs, body


def _checked_digest(hdrs: dict[str, str], body: bytes, what: str) -> str:
    actual = "sha256:" + hashlib.sha256(body).hexdigest()
    claimed = hdrs.get("docker-content-digest", "")
    if claimed and claimed != actual:
        raise PinError(f"{what}: Docker-Content-Digest {claimed} != body {actual}")
    return actual


def resolve_registry(ref: str, fetch: Fetcher = urllib_fetch) -> Resolved:
    """Resolve ``repo:tag`` through the Docker Hub HTTP API, pulling no layers.

    The top-level digest is taken from ``Docker-Content-Digest`` and checked
    against the sha256 of the manifest body. For a single-platform manifest
    the platform comes from its config blob.

    Raises:
      PinError: Auth, HTTP, JSON or digest-consistency failure.

    """
    if "@" in ref:  # repo@sha256:<hex>: the digest is the manifest reference
        repo, _, tag = ref.partition("@")
    else:
        repo, _, tag = ref.partition(":")
        tag = tag or "latest"
    token = _token(repo, fetch)
    hdrs, body = _get(repo, "manifests", tag, token, fetch)
    digest = _checked_digest(hdrs, body, ref)
    try:
        doc = _as_dict(cast("object", json.loads(body)))
    except ValueError as e:
        raise PinError(f"{ref}: manifest is not JSON") from e
    platforms: dict[str, str] = {}
    media = doc.get("mediaType") or hdrs.get("content-type", "")
    if media in _INDEX_TYPES or "manifests" in doc:
        for m in map(_as_dict, _as_list(doc.get("manifests", []))):
            p = _as_dict(m.get("platform"))
            os_, arch = p.get("os", "?"), p.get("architecture", "?")
            if os_ == "unknown" or arch == "unknown":
                continue  # attestation manifests
            name = f"{os_}/{arch}" + (f"/{p['variant']}" if p.get("variant") else "")
            platforms[name] = str(m["digest"])
    else:
        cfg = _as_dict(doc.get("config")).get("digest")
        if cfg:
            _, cbody = _get(repo, "blobs", str(cfg), token, fetch)
            c = _as_dict(cast("object", json.loads(cbody)))
            name = f"{c.get('os', '?')}/{c.get('architecture', '?')}" + (
                f"/{c['variant']}" if c.get("variant") else ""
            )
            platforms[name] = digest
    return Resolved(digest, platforms)


def _as_dict(x: object) -> dict[str, object]:
    """``x`` if it is a JSON object, else empty."""
    return cast("dict[str, object]", x) if isinstance(x, dict) else {}


def _as_list(x: object) -> list[object]:
    """``x`` if it is a JSON array, else empty."""
    return cast("list[object]", x) if isinstance(x, list) else []


def read_lock(path: Path) -> dict[str, str]:
    """Read the lock; a missing file is an empty lock."""
    if not path.exists():
        return {}
    raw = cast("object", json.loads(path.read_text() or "{}"))
    if not isinstance(raw, dict):
        raise PinError(f"lock must be a JSON object: {path}")
    return {str(k): str(v) for k, v in cast("dict[object, object]", raw).items()}


def write_lock(path: Path, lock: dict[str, str]) -> None:
    """Write the lock atomically, sorted, so diffs are stable."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n")
    tmp.replace(path)


def pin(
    root: Path,
    lock_path: Path,
    packages: Sequence[str] | None,
    *,
    repin: bool = False,
    docker: str = "docker",
    runner: Runner = _run,
) -> tuple[dict[str, str], dict[str, str]]:
    """Pin ``packages`` (``None`` = all); return ``(lock, errors)``.

    The lock is written after each success, so an interrupted run keeps
    its progress. A failing package is reported in ``errors``, not raised.
    """
    bases = find_bases(root)
    wanted = list(bases) if packages is None else list(packages)
    unknown = [p for p in wanted if p not in bases]
    if unknown:
        raise PinError(f"unknown packages (not in any Dockerfile): {unknown}")
    lock = read_lock(lock_path)
    errors: dict[str, str] = {}
    for pkg in wanted:
        if pkg in lock and not repin:
            continue
        try:
            lock[pkg] = pin_one(bases[pkg], docker, runner)
        except PinError as e:
            errors[pkg] = str(e)
            continue
        write_lock(lock_path, lock)
    return lock, errors


def pin_registry(
    root: Path,
    lock_path: Path,
    packages: Sequence[str] | None,
    *,
    repin: bool = False,
    platforms_path: Path | None = None,
    fetch: Fetcher = urllib_fetch,
) -> tuple[dict[str, str], dict[str, str]]:
    """Like :func:`pin` but via the registry API (no pulls); return ``(lock, errors)``.

    Writes ``repo@<top-level digest>`` per package to the lock and, when
    ``platforms_path`` is given, ``package -> {platform: digest}`` there.
    The lock is rewritten after each success. Failures go to ``errors``.
    """
    bases = find_bases(root)
    wanted = list(bases) if packages is None else list(packages)
    unknown = [p for p in wanted if p not in bases]
    if unknown:
        raise PinError(f"unknown packages (not in any Dockerfile): {unknown}")
    lock = read_lock(lock_path)
    plats: dict[str, dict[str, str]] = {}
    if platforms_path is not None and platforms_path.exists():
        plats = cast(
            "dict[str, dict[str, str]]",
            json.loads(platforms_path.read_text() or "{}"),
        )
    errors: dict[str, str] = {}
    for pkg in wanted:
        if pkg in lock and not repin:
            continue
        ref = bases[pkg]
        try:
            res = resolve_registry(ref, fetch)
        except PinError as e:
            errors[pkg] = str(e)
            continue
        if "@" in ref:
            repo = ref.split("@", maxsplit=1)[0]
        else:
            repo = ref.rsplit(":", 1)[0]
        lock[pkg] = f"{repo}@{res.digest}"
        write_lock(lock_path, lock)
        if platforms_path is not None:
            plats[pkg] = dict(sorted(res.platforms.items()))
            write_lock_like(platforms_path, plats)
    return lock, errors


def write_lock_like(path: Path, data: dict[str, dict[str, str]]) -> None:
    """Write a nested mapping atomically with sorted keys."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")
    tmp.replace(path)


class _Flags(Protocol):
    root: str
    lock: Path
    package: list[str]
    all: bool
    list: bool
    repin: bool
    docker: str
    registry: bool
    platforms: Path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--root", default=str(default_path(UPSTREAM_DIR)))
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--package", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="pin every package (huge pulls)")
    ap.add_argument("--list", action="store_true", help="list packages and pin status")
    ap.add_argument("--repin", action="store_true")
    ap.add_argument("--docker", default="docker")
    ap.add_argument(
        "--registry",
        action="store_true",
        help="resolve digests via the Docker Hub HTTP API (no pull, no docker)",
    )
    ap.add_argument("--platforms", type=Path, default=DEFAULT_PLATFORMS)
    args = cast("_Flags", ap.parse_args(argv))
    root = Path(args.root).expanduser()
    try:
        if args.list:
            lock = read_lock(args.lock)
            for pkg, ref in find_bases(root).items():
                print(f"{'pinned' if pkg in lock else 'unpinned':9} {pkg:40} {ref}")
            return 0
        if not args.all and not args.package:
            print(
                "error: give --package NAME (repeatable), --all or --list",
                file=sys.stderr,
            )
            return 2
        if args.registry:
            lock, errors = pin_registry(
                root,
                args.lock,
                None if args.all else args.package,
                repin=args.repin,
                platforms_path=args.platforms,
                fetch=urllib_fetch,
            )
            for pkg, msg in errors.items():
                print(f"FAILED {pkg}: {msg}", file=sys.stderr)
            print(f"{len(lock)} package(s) pinned in {args.lock}")
            return 1 if errors else 0
        if shutil.which(args.docker) is None:
            print(f"error: docker binary not found: {args.docker}", file=sys.stderr)
            return 2
        lock, errors = pin(
            root,
            args.lock,
            None if args.all else args.package,
            repin=args.repin,
            docker=args.docker,
        )
    except PinError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for pkg, msg in errors.items():
        print(f"FAILED {pkg}: {msg}", file=sys.stderr)
    print(f"{len(lock)} package(s) pinned in {args.lock}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
