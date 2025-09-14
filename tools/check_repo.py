#!/usr/bin/env python3
import sys
import pathlib

TEXT_EXTS = {'.py', '.md', '.json', '.yml', '.yaml', '.toml', '.txt', '.cfg', '.ini'}
SKIP_DIRS = {'.git', '__pycache__', '.venv', 'env', 'venv', 'results', '.vscode'}


def is_text_file(p: pathlib.Path) -> bool:
    return p.suffix.lower() in TEXT_EXTS


def scan(path: pathlib.Path) -> int:
    rc = 0
    for p in path.rglob('*'):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not is_text_file(p):
            continue
        try:
            data = p.read_bytes()
        except Exception:
            continue
        # Quickly check for issues
        issues = []
        if b'\x00' in data:
            issues.append('NUL')
        if b'\xef\xbf\xbd' in data:  # UTF-8 replacement char
            issues.append('U+FFFD')
        if b'\r\n' in data:
            issues.append('CRLF')
        # ASCII control chars except tab/newline/carriage return
        for b in data:
            if b < 32 and b not in (9, 10, 13):
                issues.append('CTRL')
                break
        if issues:
            print(f"[ISSUE] {p}: {','.join(sorted(set(issues)))}")
            rc = 1
    return rc


if __name__ == '__main__':
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.exit(scan(root))

