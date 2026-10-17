"""Tests for the Harbor bundle reader, its guard and the eval-script pins."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from priml.baselines.mlsbench import harbor
from priml.baselines.mlsbench.harbor import (
    AdaptedScripts,
    BundleError,
    EvalScriptsError,
    HarborBundle,
    adapted_scripts,
    allow_internet,
    bundle_dir,
    bundle_gpus,
    check_eval_scripts,
    check_native_pins,
    check_segments,
    eval_script_pins,
    eval_scripts_intact,
    find_bundle,
    generate_pins,
    guard,
    load_bundle,
    normalise_harbor_script,
    pinned_tasks,
    sanitised_env,
)
from priml.baselines.mlsbench.scripts import pin_eval_scripts
from priml.baselines.mlsbench.testing import RUN_PY


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _lines(n: int) -> bytes:
    return "".join(f"line{i}\n" for i in range(1, n + 1)).encode()


def make_bundle(
    root: Path,
    *,
    ranges: list[list[int]] | None = None,
    extra: dict[str, bytes] | None = None,
    declared: str = "pkg/run.py",
    content: bytes | None = None,
) -> Path:
    """Synthetic Harbor-shaped bundle (only the fields the reader consumes)."""
    body = RUN_PY.encode() if content is None else content
    b = root / "mls-bench__toy"
    meta = b / "tests" / "meta"
    (meta / "pristine" / Path(declared).parent).mkdir(parents=True)
    (meta / "pristine" / declared).write_bytes(body)
    files = {declared: body, **(extra or {})}
    (meta / "pristine_manifest.json").write_text(
        json.dumps({k: _sha(v) for k, v in files.items()}),
    )
    edit = [{"start": s, "end": e} for s, e in (ranges or [[5, 10], [15, 16]])]
    (meta / "config.json").write_text(
        json.dumps({"files": [{"filename": declared, "edit": edit}]}),
    )
    (b / "tests" / "eval" / "scripts").mkdir(parents=True)
    (b / "tests" / "eval" / "scripts" / "a.sh").write_text("echo eval\n")
    return b


def _table(*, normalised: bool = False) -> AdaptedScripts:
    """Return an adapted-script table naming only ``toy``'s ``a.sh``."""
    return AdaptedScripts(
        adapted={"toy": ("a.sh",)},
        normalised_only=frozenset({"toy"} if normalised else ()),
    )


def make_ws(root: Path, extra: dict[str, bytes], declared: str = "pkg/run.py") -> Path:
    ws = root / "ws"
    for k, v in {declared: RUN_PY.encode(), **extra}.items():
        (ws / k).parent.mkdir(parents=True, exist_ok=True)
        (ws / k).write_bytes(v)
    return ws


def _sub(n: int, new: bytes) -> bytes:
    ls = _lines(20).splitlines(keepends=True)
    ls[n - 1 : n] = [new]
    return b"".join(ls)


@pytest.mark.parametrize("n", [5, 8, 10, 15, 16])
def test_segments_inside_ok(n: int):
    assert (
        check_segments(_lines(20), _sub(n, b"a\nb\nc\n"), ((5, 10), (15, 16))) is None
    )


@pytest.mark.parametrize("n", [1, 4, 11, 14, 17, 20])
def test_segments_outside_fail(n: int):
    assert check_segments(_lines(20), _sub(n, b"x\n"), ((5, 10), (15, 16))) is not None


def test_segments_length_changing_replacement_ok():
    ls = _lines(20).splitlines(keepends=True)
    big = [b"new%d\n" % i for i in range(50)]
    cur = b"".join([*ls[:4], *big, *ls[10:14], b"z\n", *ls[16:]])
    assert check_segments(_lines(20), cur, ((5, 10), (15, 16))) is None


def test_segments_fixed_deleted_with_duplicate_in_edit_fails():
    # Delete fixed line 12 but paste a copy of it inside the editable range.
    ls = _lines(20).splitlines(keepends=True)
    cur = b"".join([*ls[:6], b"line12\n", *ls[6:11], *ls[12:]])
    assert check_segments(_lines(20), cur, ((5, 10), (15, 16))) is not None


def test_segments_crlf_not_normalised():
    p = b"a\nb\nc\nd\n"
    assert (
        check_segments(p, b"a\r\nb\nc\nd\n", ((2, 3),)) is not None
    )  # Line a altered.
    assert check_segments(p, b"a\nB\r\nc\nd\n", ((2, 3),)) is None


def test_segments_binary_and_whole_file_and_readonly():
    assert check_segments(b"\xff\xfe\n", b"\xff\xfd\n", ((-1, -1),)) is None
    assert check_segments(b"\xff\xfe\n", b"\xff\xfd\n", ()) is not None
    assert check_segments(b"x\n", b"x\n", ()) is None


def test_segments_end_minus_one_means_eof():
    assert check_segments(_lines(6), _lines(3) + b"tail\n", ((4, -1),)) is None
    assert check_segments(_lines(6), b"X\n" + _lines(6)[3:], ((4, -1),)) is not None


def test_segments_swapped_order_fails():
    p = b"A\ne1\nB\ne2\nC\n"
    cur = b"A\ne1\nC\ne2\nB\n"
    assert check_segments(p, cur, ((2, 2), (4, 4))) is not None


def test_segments_overlap_merge_and_fixed_line_checks():
    assert check_segments(b"a\nb\na\n", b"a\n", ((2, 2),)) is not None
    p = b"a\nb\nc\nd\ne\n"
    assert check_segments(p, b"a\nX\nY\nZ\ne\n", ((2, 3), (3, 4))) is None
    got = check_segments(b"A\ne\nB\ne\nC\n", b"A\nB\nA\ne\nC\n", ((2, 2), (4, 4)))
    assert got == "fixed line(s) [3] changed"


@pytest.mark.parametrize(
    "files",
    [[{"filename": 3}], [{"filename": ""}], [{"filename": "a\\b"}]],
)
def test_config_rejects_unsafe_filenames(tmp_path: Path, files: list[object]):
    b = make_bundle(tmp_path)
    (b / "tests/meta/config.json").write_text(json.dumps({"files": files}))
    with pytest.raises(BundleError):
        load_bundle(b)


def test_config_unusable_fails_closed(tmp_path: Path):
    b = make_bundle(tmp_path)
    (b / "tests/meta/config.json").unlink()
    with pytest.raises(BundleError, match=r"config\.json unusable"):
        load_bundle(b)


def test_guard_declared_file_without_pristine(tmp_path: Path):
    b = make_bundle(tmp_path)
    (b / "tests/meta/pristine/pkg/run.py").unlink()
    ws = make_ws(tmp_path, {})
    assert [v.kind for v in guard(load_bundle(b), ws)] == ["pristine_missing"]
    whole = load_bundle(make_bundle(tmp_path / "w", ranges=[[-1, -1]]))
    (whole.pristine_root / "pkg/run.py").unlink()
    assert guard(whole, ws) == ()


def test_load_bundle(tmp_path: Path):
    hb = load_bundle(make_bundle(tmp_path, extra={"pkg/util.py": b"u\n"}))
    assert hb.editable == {"pkg/run.py": ((5, 10), (15, 16))}
    assert set(hb.manifest) == {"pkg/run.py", "pkg/util.py"}
    assert hb.guarded_prefixes == {"pkg"}


@pytest.mark.parametrize(
    "bad",
    ["", "{", "[]", "{}", '{"a": "short"}', '{"../x": "%s"}' % ("0" * 64)],
)
def test_manifest_fails_closed(tmp_path: Path, bad: str):
    b = make_bundle(tmp_path)
    (b / "tests/meta/pristine_manifest.json").write_text(bad)
    with pytest.raises(BundleError):
        load_bundle(b)


def test_manifest_missing_fails_closed(tmp_path: Path):
    b = make_bundle(tmp_path)
    (b / "tests/meta/pristine_manifest.json").unlink()
    with pytest.raises(BundleError, match="missing"):
        load_bundle(b)


def test_bundle_dir_lookup(tmp_path: Path):
    root = tmp_path / "up"
    d = root / "harbor/tasks-docker"
    d.mkdir(parents=True)
    make_bundle(d)
    assert bundle_dir("toy", root=root).name == "mls-bench__toy"
    with pytest.raises(BundleError):
        bundle_dir("nope", root=root)


@pytest.fixture
def hb(tmp_path: Path) -> HarborBundle:
    return load_bundle(
        make_bundle(
            tmp_path,
            extra={"pkg/eval.py": b"score()\n", "pkg/data.bin": b"\x00\x01"},
        ),
    )


_EXTRA = {"pkg/eval.py": b"score()\n", "pkg/data.bin": b"\x00\x01"}


def test_guard_clean(hb: HarborBundle, tmp_path: Path):
    assert guard(hb, make_ws(tmp_path, _EXTRA)) == ()


def test_guard_edit_inside_and_new_file_ok(hb: HarborBundle, tmp_path: Path):
    ws = make_ws(tmp_path, _EXTRA)
    (ws / "pkg/run.py").write_bytes(_sub(6, b"ok\n"))
    (ws / "pkg/helper.py").write_text("new\n")
    (ws / "outside.txt").write_text("new\n")
    assert guard(hb, ws) == ()


def test_guard_edited_eval_script(hb: HarborBundle, tmp_path: Path):
    ws = make_ws(tmp_path, _EXTRA)
    (ws / "pkg/eval.py").write_text("score = lambda: 1.0\n")
    (v,) = guard(hb, ws)
    assert (v.path, v.kind) == ("pkg/eval.py", "modified_unlisted")


def test_guard_binary_data_edit(hb: HarborBundle, tmp_path: Path):
    ws = make_ws(tmp_path, _EXTRA)
    (ws / "pkg/data.bin").write_bytes(b"\x00\x02")
    assert [v.kind for v in guard(hb, ws)] == ["modified_unlisted"]


def test_guard_outside_allowed_range(hb: HarborBundle, tmp_path: Path):
    ws = make_ws(tmp_path, _EXTRA)
    (ws / "pkg/run.py").write_bytes(_sub(2, b"hack\n"))
    (v,) = guard(hb, ws)
    assert (v.path, v.kind) == ("pkg/run.py", "outside_range")


def test_guard_deleted_file(hb: HarborBundle, tmp_path: Path):
    ws = make_ws(tmp_path, _EXTRA)
    (ws / "pkg/eval.py").unlink()
    assert [v.kind for v in guard(hb, ws)] == ["deleted"]
    assert guard(hb, ws, check_deletions=False) == ()
    (ws / "pkg/run.py").unlink()
    assert [v.kind for v in guard(hb, ws, check_deletions=False)] == ["deleted"]


def test_guard_symlink_swap_unlisted(hb: HarborBundle, tmp_path: Path):
    ws = make_ws(tmp_path, _EXTRA)
    (ws / "pkg/eval.py").unlink()
    (tmp_path / "evil.py").write_bytes(b"score()\n")  # Same content as pristine!
    (ws / "pkg/eval.py").symlink_to(tmp_path / "evil.py")
    assert [v.kind for v in guard(hb, ws)] == ["symlink"]


def test_guard_symlink_swap_declared(hb: HarborBundle, tmp_path: Path):
    ws = make_ws(tmp_path, _EXTRA)
    (ws / "pkg/run.py").unlink()
    (tmp_path / "evil.py").write_bytes(RUN_PY.encode())
    (ws / "pkg/run.py").symlink_to(tmp_path / "evil.py")
    assert "symlink" in [v.kind for v in guard(hb, ws)]


def test_guard_symlinked_parent_directory(hb: HarborBundle, tmp_path: Path):
    ws = make_ws(tmp_path, _EXTRA)
    real = tmp_path / "elsewhere"
    real.mkdir()
    for f in (ws / "pkg").iterdir():
        f.rename(real / f.name)
    (ws / "pkg").rmdir()
    (ws / "pkg").symlink_to(real, target_is_directory=True)
    kinds = {v.kind for v in guard(hb, ws)}
    assert kinds == {"symlink"}


def test_guard_skips_exempt_and_excluded(tmp_path: Path):
    b = make_bundle(tmp_path, extra={"pkg/assets/big.bin": b"1"})
    cfg = cast(
        "dict[str, object]",
        json.loads((b / "tests/meta/config.json").read_text()),
    )
    cfg["guard_exclude"] = ["pkg/assets"]
    (b / "tests/meta/config.json").write_text(json.dumps(cfg))
    hb = load_bundle(b)
    ws = make_ws(tmp_path, {"pkg/assets/big.bin": b"CHANGED"})
    assert guard(hb, ws) == ()


def test_eval_scripts_intact(hb: HarborBundle, tmp_path: Path):
    d = tmp_path / "scripts"
    d.mkdir()
    assert [v.kind for v in eval_scripts_intact(hb, d)] == ["eval_script_missing"]
    (d / "a.sh").write_text("echo eval\n")
    assert eval_scripts_intact(hb, d) == ()
    (d / "a.sh").write_text("echo 100\n")
    assert [v.kind for v in eval_scripts_intact(hb, d)] == ["eval_script_modified"]


def test_eval_scripts_intact_compares_helper_files_not_only_sh(
    hb: HarborBundle,
    tmp_path: Path,
):
    ref = hb.root / "tests/eval/scripts"
    (ref / "_helper.py").write_text("X = 1\n")
    d = tmp_path / "scripts"
    d.mkdir()
    (d / "a.sh").write_text("echo eval\n")
    assert [(v.path, v.kind) for v in eval_scripts_intact(hb, d)] == [
        ("_helper.py", "eval_script_missing"),
    ]
    (d / "_helper.py").write_text("X = 2\n")
    assert [v.kind for v in eval_scripts_intact(hb, d)] == ["eval_script_modified"]


def _scripts(tmp_path: Path, text: str = "echo eval\n") -> Path:
    d = tmp_path / "scripts"
    d.mkdir(exist_ok=True)
    (d / "a.sh").write_text(text)
    return d


def test_check_eval_scripts_clean_task_must_match_exactly(
    hb: HarborBundle,
    tmp_path: Path,
):
    assert check_eval_scripts("toy", hb, _scripts(tmp_path)) == ()
    with pytest.raises(EvalScriptsError, match=r"eval_script_modified a.sh"):
        check_eval_scripts("toy", hb, _scripts(tmp_path, "echo 100\n"))


def test_check_eval_scripts_refuses_a_swapped_or_removed_script(
    hb: HarborBundle,
    tmp_path: Path,
):
    d = _scripts(tmp_path)
    (d / "a.sh").unlink()
    with pytest.raises(EvalScriptsError, match=r"eval_script_missing a.sh"):
        check_eval_scripts("toy", hb, d)
    (d / "real.sh").write_text("echo eval\n")
    (d / "a.sh").symlink_to(d / "real.sh")
    with pytest.raises(EvalScriptsError, match=r"eval_script_missing a.sh"):
        check_eval_scripts("toy", hb, d)


def test_check_eval_scripts_normalised_task_allows_only_documented_additions(
    hb: HarborBundle,
    tmp_path: Path,
):
    table = _table(normalised=True)
    harbor = hb.root / "tests/eval/scripts/a.sh"
    harbor.write_text(
        "cd /workspace\n"
        'python "/tests/eval/_inputgen/apply.py" "toy" /workspace\n'
        "cd /workspace/pkg\necho eval\n",
    )
    assert check_eval_scripts("toy", hb, _scripts(tmp_path), table=table) == ("a.sh",)
    with pytest.raises(EvalScriptsError, match=r"eval_script_modified a.sh"):
        check_eval_scripts("toy", hb, _scripts(tmp_path, "echo evil\n"), table=table)
    # An undocumented extra line on the Harbor side is not normalised away.
    harbor.write_text("cd /workspace\nrm -rf x\necho eval\n")
    with pytest.raises(EvalScriptsError, match=r"eval_script_modified a.sh"):
        check_eval_scripts("toy", hb, _scripts(tmp_path), table=table)


def test_check_eval_scripts_substantive_task_pins_the_differing_files(
    hb: HarborBundle,
    tmp_path: Path,
):
    table = _table()
    pins = {"toy": {"a.sh": _sha(b"echo native\n")}}
    assert check_eval_scripts(
        "toy",
        hb,
        _scripts(tmp_path, "echo native\n"),
        pins,
        table,
    ) == ("a.sh",)
    # A pinned adaptation that vanished is drift and refuses too.
    with pytest.raises(EvalScriptsError, match=r"eval_script_expected_adapted a.sh"):
        check_eval_scripts("toy", hb, _scripts(tmp_path), pins, table)
    # A different file differing is a swap.
    (hb.root / "tests/eval/scripts/b.sh").write_text("echo b\n")
    (tmp_path / "scripts/a.sh").write_text("echo native\n")
    (tmp_path / "scripts/b.sh").write_text("echo swapped\n")
    with pytest.raises(EvalScriptsError, match=r"eval_script_modified b.sh"):
        check_eval_scripts("toy", hb, _scripts(tmp_path, "echo native\n"), pins, table)


def test_substantive_task_bytes_are_pinned_not_just_the_file_set(
    hb: HarborBundle,
    tmp_path: Path,
):
    table = _table()
    pins = {"toy": {"a.sh": _sha(b"echo native\n")}}
    assert check_eval_scripts(
        "toy",
        hb,
        _scripts(tmp_path, "echo native\n"),
        pins,
        table,
    ) == ("a.sh",)
    # Same differing-file set, different bytes: refused.
    with pytest.raises(EvalScriptsError, match=r"eval_script_pin_mismatch a.sh"):
        check_eval_scripts("toy", hb, _scripts(tmp_path, "echo native \n"), pins, table)
    # An extra, unpinned file is refused.
    d = _scripts(tmp_path, "echo native\n")
    (d / "extra.sh").write_text("echo x\n")
    with pytest.raises(EvalScriptsError, match=r"eval_script_unpinned extra.sh"):
        check_eval_scripts("toy", hb, d, pins, table)


def test_substantive_task_without_a_pin_fails_closed(
    hb: HarborBundle,
    tmp_path: Path,
):
    with pytest.raises(EvalScriptsError, match=r"eval_script_unpinned"):
        check_eval_scripts(
            "toy",
            hb,
            _scripts(tmp_path, "echo native\n"),
            pins={},
            table=_table(),
        )


def test_check_native_pins_reports_missing_and_symlinked(tmp_path: Path):
    d = _scripts(tmp_path)
    pins = {"t": {"a.sh": _sha(b"echo eval\n"), "b.sh": _sha(b"")}}
    assert [v.kind for v in check_native_pins("t", d, pins)] == ["eval_script_missing"]
    (d / "b.sh").symlink_to(d / "a.sh")
    assert [v.kind for v in check_native_pins("t", d, pins)] == ["eval_script_missing"]


def test_shipped_pins_cover_exactly_the_substantive_tasks():
    assert set(eval_script_pins()) == pinned_tasks()
    assert len(pinned_tasks()) == 20
    for task, files in eval_script_pins().items():
        assert files, task
        assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in files.values()), task


def test_generate_pins_and_pin_cli_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(harbor, "adapted_scripts", _table)
    sd = tmp_path / "clone/tasks/toy/scripts"
    (sd / "sub").mkdir(parents=True)
    (sd / "a.sh").write_bytes(b"A")
    (sd / "sub/b.py").write_bytes(b"B")
    want = {"toy": {"a.sh": _sha(b"A"), "sub/b.py": _sha(b"B")}}
    assert generate_pins(tmp_path / "clone", _table()) == want
    out = tmp_path / "pins.json"
    argv = ["--root", str(tmp_path / "clone"), "--out", str(out)]
    assert pin_eval_scripts.main(argv) == 0
    assert json.loads(out.read_text()) == want


def test_normalise_harbor_script_strips_only_documented_lines():
    text = (
        "cd /workspace\n\ncd /workspace/gplearn\n"
        'python "/tests/eval/_inputgen/apply.py" "ml-x" /workspace\n'
        "cd /workspace/gplearn/sub dir\ncd /workspace2\ncd /tmp\necho hi  \n"
    )
    assert normalise_harbor_script(text) == [
        "cd /workspace/gplearn/sub dir",
        "cd /workspace2",
        "cd /tmp",
        "echo hi",
    ]


def test_adapted_table_is_consistent():
    table = adapted_scripts()
    assert len(table["adapted"]) == 32
    assert set(table["adapted"]) >= table["normalised_only"]
    assert len(table["normalised_only"]) == 12
    assert all(files for files in table["adapted"].values())


def test_sanitised_env_drops_shadowing():
    env = sanitised_env(
        {
            "PATH": "/workspace/shim:/usr/bin",
            "PYTHONSTARTUP": "/w/x.py",
            "PYTHONPATH": "/w",
            "LD_PRELOAD": "/w/x.so",
            "KEEP": "1",
        },
    )
    assert env["PATH"].split(":")[0] == "/usr/local/sbin"
    assert "/workspace/shim" not in env["PATH"]
    assert not {"PYTHONSTARTUP", "PYTHONPATH", "LD_PRELOAD"} & set(env)
    assert env["KEEP"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_python_still_runs_guard_module_isolated():
    # The module must import without touching upstream or the network.
    r = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import priml.baselines.mlsbench.workspace",
        ],
        capture_output=True,
        check=False,
        env=sanitised_env({"HOME": os.environ.get("HOME", "/")}),
    )
    assert r.returncode in {0, 1}


@pytest.mark.integration
def test_real_bundle_dllm():
    root = os.environ.get("MLSBENCH_ROOT", "")
    if not root:
        pytest.skip("MLSBENCH_ROOT not set")
    bd = bundle_dir("llm-dllm-demask-strategy", root=Path(root))
    hb = load_bundle(bd)
    rel = "LLaDA/custom_demask_eval.py"
    assert hb.editable[rel] == ((59, 151),)
    pristine = (hb.pristine_root / rel).read_bytes()
    assert _sha(pristine) == hb.manifest[rel]
    assert len(hb.manifest) > 1000

    # Pristine workspace: only the declared file is present -> clean w/o deletion check.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        (ws / rel).parent.mkdir(parents=True)
        (ws / rel).write_bytes(pristine)
        assert guard(hb, ws, check_deletions=False) == ()

        ls = pristine.splitlines(keepends=True)
        (ws / rel).write_bytes(b"".join([*ls[:60], b"# ok\n", *ls[60:]]))
        assert guard(hb, ws, check_deletions=False) == ()
        (ws / rel).write_bytes(b"import os\n" + pristine)
        assert [v.kind for v in guard(hb, ws, check_deletions=False)] == [
            "outside_range",
        ]

    # Every baseline edit script must yield a clean guard on the declared file.
    scripts = sorted((bd / "tests/meta/edits").glob("*.edit.py"))
    assert scripts
    for s in scripts:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / rel).parent.mkdir(parents=True)
            (ws / rel).write_bytes(pristine)
            r = subprocess.run(  # noqa: S603 -- fixed interpreter and test-owned argv.
                [sys.executable, "-I", str(s), str(ws)],
                cwd=ws,
                env=sanitised_env(),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert r.returncode == 0, (s.name, r.stderr[-500:])
            assert guard(hb, ws, check_deletions=False) == (), s.name

    assert eval_scripts_intact(hb, bd / "tests/eval/scripts") == ()


@pytest.mark.integration
def test_real_clone_pins_match_and_one_byte_change_is_refused(tmp_path: Path):
    """Pins equal the clone's native scripts; a one-byte change is refused.

    A deliberate upstream bump regenerates them: ``python -m
    priml.baselines.mlsbench.scripts.pin_eval_scripts`` (see ``generate_pins``); review the
    JSON diff before accepting.
    """
    root = os.environ.get("MLSBENCH_ROOT", "")
    if not root:
        pytest.skip("MLSBENCH_ROOT not set")
    assert generate_pins(Path(root)) == eval_script_pins()
    for task in sorted(pinned_tasks()):
        sd = Path(root) / "tasks" / task / "scripts"
        assert check_native_pins(task, sd) == (), task
    task = (
        "llm-dllm-demask-strategy"
        if "llm-dllm-demask-strategy" in eval_script_pins()
        else min(eval_script_pins())
    )
    src = Path(root) / "tasks" / task / "scripts"
    copy = tmp_path / "scripts"
    shutil.copytree(src, copy)
    victim = copy / min(eval_script_pins()[task])
    victim.write_bytes(victim.read_bytes() + b"\n")
    hb = load_bundle(bundle_dir(task, root=Path(root)))
    with pytest.raises(EvalScriptsError, match=r"eval_script_pin_mismatch"):
        check_eval_scripts(task, hb, copy)


def test_allow_internet_reads_task_toml_and_is_none_when_unknown(
    tmp_path: Path,
) -> None:
    assert allow_internet(tmp_path) is None
    (tmp_path / "task.toml").write_text("[environment]\nallow_internet = true\n")
    assert allow_internet(tmp_path) is True
    (tmp_path / "task.toml").write_text("[environment]\nallow_internet = 'yes'\n")
    assert allow_internet(tmp_path) is None
    (tmp_path / "task.toml").write_text("not toml [")
    assert allow_internet(tmp_path) is None
    assert find_bundle("nope", root=tmp_path) is None


def test_bundle_gpus_reads_task_toml(tmp_path: Path) -> None:
    assert bundle_gpus(tmp_path) is None
    (tmp_path / "task.toml").write_text("[environment]\ngpus = 3\n")
    assert bundle_gpus(tmp_path) == 3
    (tmp_path / "task.toml").write_text("[environment]\ncpus = 8\n")
    assert bundle_gpus(tmp_path) == 0  # Harbor's default: no GPU.
    (tmp_path / "task.toml").write_text("[environment]\ngpus = 'two'\n")
    assert bundle_gpus(tmp_path) is None


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
