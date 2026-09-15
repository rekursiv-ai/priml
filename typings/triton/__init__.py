"""Placeholder source for the hand-written `triton` stubs.

Triton is a linux-only dependency of torch (`uv.lock`: `sys_platform ==
'linux'`), so on darwin checkouts the wheel is absent and basedpyright's
`reportMissingModuleSource` fires on every `import triton` -- a diagnostic
about the environment, not the code. Suppressing it inline is impossible:
the comment is then flagged `reportUnnecessaryTypeIgnoreComment` on linux,
where the wheel *is* installed.

These placeholders satisfy source resolution on both platforms. Types still
come from the adjacent `.pyi` (stubs win over source), so nothing here needs
to mirror the real API. `typings` is on `extraPaths` so this file is found;
it is never on the runtime `sys.path`.
"""
