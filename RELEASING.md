# Releasing priml to PyPI

Maintainer-only. Replace `X.Y.Z` with the new version throughout.

A published GitHub Release triggers `.github/workflows/publish-pypi.yml`,
which builds, validates, and uploads to PyPI via OIDC trusted publishing
(no API token). Follow the steps in order.

## Steps

1. **(First release only) Register the trusted publisher on PyPI.**
   Skip if already done. Web-console only, no CLI. Go to
   https://pypi.org/manage/project/priml/settings/publishing/ and add
   a publisher with these exact values:

   | Field | Value |
   |---|---|
   | Owner | `rekursiv-ai` |
   | Repository name | `priml` |
   | Workflow filename | `publish-pypi.yml` |
   | Environment name | `pypi` |

   Environment name **must** be `pypi` (matches `publish-pypi.yml:16`).
   Blank here = `invalid-publisher` at publish time.

2. **Bump the version** in `pyproject.toml` (source of truth). Must
   increase; PyPI rejects re-uploads.

3. **Validate locally** (same checks CI runs):
   ```bash
   uv build
   uv run python -c "import priml; print(priml.__file__)"
   ```

4. **Commit and merge to `main`.** Confirm the published version:
   ```bash
   gh api repos/rekursiv-ai/priml/contents/pyproject.toml --jq .content | base64 -d | grep '^version'
   ```

5. **Cut the release** (this triggers the PyPI publish):
   ```bash
   gh release create vX.Y.Z --repo rekursiv-ai/priml --title "vX.Y.Z" --generate-notes
   ```

6. **Watch the workflow:**
   ```bash
   gh run watch --repo rekursiv-ai/priml $(gh run list --repo rekursiv-ai/priml --workflow publish-pypi.yml --limit 1 --json databaseId --jq '.[0].databaseId')
   ```

7. **Confirm it's live:**
   ```bash
   curl -s https://pypi.org/pypi/priml/json | jq -r '.info.version'   # should print X.Y.Z
   ```
   Browser: https://pypi.org/project/priml/

## If the workflow fails

- **`invalid-publisher`** ("valid token, but no corresponding
  publisher") — the trusted publisher isn't registered or doesn't match.
  Do step 1, then re-run: `gh workflow run publish-pypi.yml --repo rekursiv-ai/priml`.

- **Transient PyPI failure** — re-run without a new tag:
  `gh workflow run publish-pypi.yml --repo rekursiv-ai/priml`.

- **Must ship now, publishing still broken** — upload directly with the
  project's API token (`rekursiv-ai`-owned, e.g. `$PYPI_TOKEN_WORK`; a
  personal token can't publish `priml`):
  ```bash
  uv build --out-dir dist
  uv run --with twine twine check dist/priml-X.Y.Z*
  uv run --with twine twine upload dist/priml-X.Y.Z* -u __token__ -p "$PYPI_TOKEN_WORK"
  ```
  Permanent and public; PyPI never allows re-uploading a version.
