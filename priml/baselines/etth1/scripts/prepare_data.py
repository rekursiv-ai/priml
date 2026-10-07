#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Prepare the checksum-pinned ETTh1 dataset.
'''
# fmt: on

from priml.baselines.etth1.scripts.data import main


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
