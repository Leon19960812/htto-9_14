#!/usr/bin/env python3
import sys
import pathlib

TEXT_EXTS = {'.py', '.md', '.json', '.yml', '.yaml', '.toml', '.txt', '.cfg', '.ini'}
SKIP_DIRS = {'.git', '__pycache__', '.venv', 'env', 'venv', 'results', '.vscode'}


def is_text_file(p: pathlib.Path) -> bool:
    return p.suffix.lower() in TEXT_EXTS


def clean_bytes(data: bytes) -> bytes:
    # Normalize CRLF -> LF
    data = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    # Remove NULs
    data = data.replace(b'\x00', b'')
    return data


def clean_text(text: str) -> str:
    # Remove Unicode replacement character
    text = text.replace('\uFFFD', '')
    # Keep tabs and newlines; remove other ASCII control chars
    text = ''.join(ch for ch in text if (ch >= ' ' or ch in ('\n', '\t')))
    # Ensure final newline
    if not text.endswith('\n'):
        text += '\n'
    return text


def normalize(path: pathlib.Path) -> int:
    changed = 0
    for p in path.rglob('*'):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not is_text_file(p):
            continue
        try:
            raw = p.read_bytes()
        except Exception:
            continue
        cleaned = clean_bytes(raw)
        try:
            txt = cleaned.decode('utf-8', errors='replace')
        except Exception:
            continue
        fixed = clean_text(txt)
        if fixed.encode('utf-8') != raw:
            # Use binary write to control newlines explicitly
            p.write_bytes(fixed.encode('utf-8'))
            print(f"[FIX] {p}")
            changed += 1
    return changed


if __name__ == '__main__':
    root = pathlib.Path(__file__).resolve().parents[1]
    n = normalize(root)
    print(f"Done. Files updated: {n}")
    sys.exit(0)
