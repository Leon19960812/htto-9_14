# Engineering Guidelines (Active)

- Encoding and line endings
  - All text files must be UTF-8 encoded with LF line endings.
  - `.editorconfig` and `.gitattributes` already enforce this; do not override locally.
  - Use `tools/verify_encoding.py` to check/normalize files if needed.

- Comments and language
  - New code comments and docstrings must be written in English.
  - Avoid non-ASCII characters in identifiers; keep public APIs ASCII-only.
  - Do not use emojis anywhere (source code, comments, docs, commit messages).

- Source conventions
  - Keep modules minimal and focused; avoid unused legacy branches.
  - Prefer clear dataclasses and typed signatures.
  - Follow the refactor plan: one-dimensional theta variables + PolarGeometry as the single geometry source.

- Safety and correctness
  - Favor small, compiling skeletons first; iterate features after.
  - Add graceful fallbacks (e.g., solver downgrades) without hiding errors.

- Tooling
  - Recommended editor settings: `files.encoding=utf8`, `files.eol=\n`.
  - Optional pre-commit hook to reject non-UTF-8 files.

- Version control
  - Commit and push after every change so we can roll back quickly.

