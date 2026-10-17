#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Pin upstream Harbor base images to content digests in ``image_lock.json``.

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

'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    override,
)
from urllib import error, parse, request

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time

from treekle import json_loads

from priml.baselines.mlsbench.experiments import MlsbenchEval
from priml.paths import resolve_working_dir


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from http.client import HTTPMessage, HTTPResponse
    from typing import IO


_TESTDATA: Final = _CWD.parent / "testdata"
DEFAULT_LOCK: Final = _TESTDATA / "image_lock.json"
DEFAULT_PLATFORMS: Final = _TESTDATA / "image_platforms.json"
_AUTH_URL: Final = "https://auth.docker.io/token"
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
    return subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603 -- docker argv built here, no shell.


def find_bases(root: Path) -> dict[str, str]:
    """Return ``package -> "repo:tag"`` for every Harbor base under ``root``.

    Args:
      root: The upstream clone.

    Returns:
      bases: Package to ``repo:tag``.

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

    Args:
      ref: ``repo:tag`` to pull.
      docker: The docker binary.
      runner: Runs one argv (injected by tests).

    Returns:
      ref: ``repo@sha256:...``.

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
        digests = json_loads(insp.stdout, list[str])
    except ValueError as e:
        raise PinError(f"unparseable RepoDigests for {ref}: {insp.stdout!r}") from e
    repo = (
        ref.split("@", maxsplit=1)[0].rsplit(":", 1)[0]
        if ":" in ref.rsplit("/", 1)[-1]
        else ref
    )
    for d in digests:
        if d.startswith(repo + "@sha256:"):
            return d
    raise PinError(f"{ref} has no RepoDigest for {repo} (got {digests})")


class _DropAuthOnRedirect(request.HTTPRedirectHandler):
    """Blob downloads redirect to a CDN that must not see the registry token."""

    @override
    def redirect_request(
        self,
        req: request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> request.Request | None:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
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

    Args:
      url: The URL to fetch.
      headers: Request headers.
      attempts: Tries before giving up.
      sleep: Backoff between tries (injected by tests).

    Returns:
      status: HTTP status; a 4xx other than 429 is returned, not raised.
      headers: Response headers, keys lower-cased.
      body: Response body.

    Raises:
      PinError: Still failing after ``attempts`` tries.

    """
    opener = request.build_opener(_DropAuthOnRedirect)
    last = ""
    for i in range(attempts):
        req = request.Request(url, headers=headers)  # noqa: S310 -- callers pass fixed https registry URLs.
        try:
            with cast("HTTPResponse", opener.open(req, timeout=30)) as r:
                return (
                    int(r.status),
                    {str(k).lower(): str(v) for k, v in r.headers.items()},
                    bytes(r.read()),
                )
        except error.HTTPError as e:
            body = e.read()
            if e.code != 429 and e.code < 500:
                return e.code, {k.lower(): v for k, v in e.headers.items()}, body
            last = f"HTTP {e.code}"
        except (error.URLError, TimeoutError, OSError) as e:
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


def resolve_registry(ref: str, fetch: Fetcher = urllib_fetch) -> Resolved:
    """Resolve ``repo:tag`` through the Docker Hub HTTP API, pulling no layers.

    The top-level digest is taken from ``Docker-Content-Digest`` and checked
    against the sha256 of the manifest body. For a single-platform manifest
    the platform comes from its config blob.

    Args:
      ref: ``repo:tag`` or ``repo@sha256:...``.
      fetch: GETs one URL (injected by tests).

    Returns:
      resolved: The top-level digest and the per-platform digests.

    Raises:
      PinError: Auth, HTTP, JSON or digest-consistency failure.

    """
    if "@" in ref:  # repo@sha256:<hex>: the digest is the manifest reference.
        repo, _, tag = ref.partition("@")
    else:
        repo, _, tag = ref.partition(":")
        tag = tag or "latest"
    token = _token(repo, fetch)
    hdrs, body = _get(repo, "manifests", tag, token, fetch)
    digest = _checked_digest(hdrs, body, ref)
    try:
        doc = json_loads(body, _Manifest)
    except ValueError as e:
        raise PinError(f"{ref}: manifest is not JSON") from e
    platforms: dict[str, str] = {}
    media = doc.get("mediaType") or hdrs.get("content-type", "")
    if media in _INDEX_TYPES or "manifests" in doc:
        for m in doc.get("manifests", []):
            p = m.get("platform", _Platform())
            if p.get("os") == "unknown" or p.get("architecture") == "unknown":
                continue  # Attestation manifests.
            platforms[_platform_name(p)] = m["digest"]
    elif cfg := doc.get("config", _Blob()).get("digest", ""):
        _, cbody = _get(repo, "blobs", cfg, token, fetch)
        try:
            platforms[_platform_name(json_loads(cbody, _Platform))] = digest
        except ValueError as e:
            raise PinError(f"{ref}: image config is not JSON") from e
    return Resolved(digest, platforms)


def read_lock(path: Path) -> dict[str, str]:
    """Read the lock; a missing file is an empty lock.

    Args:
      path: The lock file.

    Returns:
      lock: Package to ``repo@sha256:...``.

    """
    if not path.exists():
        return {}
    try:
        return json_loads(path.read_text() or "{}", dict[str, str])
    except ValueError as e:
        raise PinError(f"lock must map package to ref: {path}") from e


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

    Args:
      root: The upstream clone.
      lock_path: The lock file.
      packages: Packages to pin; ``None`` is every one.
      repin: Re-pin packages already in the lock.
      docker: The docker binary.
      runner: Runs one argv (injected by tests).

    Returns:
      lock: The lock after this run.
      errors: Package to why it failed.

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

    Args:
      root: The upstream clone.
      lock_path: The lock file.
      packages: Packages to pin; ``None`` is every one.
      repin: Re-pin packages already in the lock.
      platforms_path: Where per-platform digests go; ``None`` writes none.
      fetch: GETs one URL (injected by tests).

    Returns:
      lock: The lock after this run.
      errors: Package to why it failed.

    """
    bases = find_bases(root)
    wanted = list(bases) if packages is None else list(packages)
    unknown = [p for p in wanted if p not in bases]
    if unknown:
        raise PinError(f"unknown packages (not in any Dockerfile): {unknown}")
    lock = read_lock(lock_path)
    plats: dict[str, dict[str, str]] = {}
    if platforms_path is not None and platforms_path.exists():
        plats = json_loads(
            platforms_path.read_text() or "{}",
            dict[str, dict[str, str]],
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
      argv: Command-line arguments; ``None`` is ``sys.argv``.

    Returns:
      code: 0 on success, 1 if a package failed, 2 on a usage or setup error.

    """
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(ap)
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


def _add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--root",
        default=str(_default_clone()),
    )
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


def _token(repo: str, fetch: Fetcher) -> str:
    q = parse.urlencode(
        {
            "service": "registry.docker.io",
            "scope": f"repository:{repo}:pull",
        },
    )
    status, _, body = fetch(f"{_AUTH_URL}?{q}", {})
    if status != 200:
        raise PinError(f"token request for {repo} failed: HTTP {status}")
    try:
        return json_loads(body, _Token)["token"]
    except ValueError as e:
        raise PinError(f"token response for {repo} has no token") from e


def _get(
    repo: str,
    kind: str,
    ref: str,
    token: str,
    fetch: Fetcher,
) -> tuple[dict[str, str], bytes]:
    url = f"https://registry-1.docker.io/v2/{repo}/{kind}/{ref}"
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


def _default_clone() -> Path:
    """Return the upstream clone a default ``MlsbenchEval`` resolves."""
    cfg = MlsbenchEval()
    return resolve_working_dir(cfg.base_dir, cfg.mlsbench_root)


def _platform_name(p: _Platform) -> str:
    """Return ``os/arch[/variant]``, ``?`` for an absent part."""
    name = f"{p.get('os', '?')}/{p.get('architecture', '?')}"
    variant = p.get("variant", "")
    return f"{name}/{variant}" if variant else name


class _Token(TypedDict):
    token: str


class _Platform(TypedDict, total=False):
    os: str
    architecture: str
    variant: str


class _Blob(TypedDict, total=False):
    digest: str


class _Entry(TypedDict):
    digest: str
    platform: NotRequired[_Platform]


class _Manifest(TypedDict, total=False):
    """The fields of a registry manifest (index or single) this script reads."""

    mediaType: str
    manifests: list[_Entry]
    config: _Blob


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
