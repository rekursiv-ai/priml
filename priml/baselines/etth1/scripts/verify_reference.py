#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Check source parity; --mint runs the source capture through pytest.
'''
# fmt: on

from priml.baselines.etth1.scripts.reference import main


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
