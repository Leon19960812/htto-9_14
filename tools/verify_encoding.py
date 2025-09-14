import sys
from pathlib import Path


ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "utf-32",
    "utf-32-le",
    "utf-32-be",
    "gb18030",
    "gbk",
    "gb2312",
    "cp936",
    "latin-1",
]


def decode_best(data: bytes):
    for enc in ENCODINGS:
        try:
            text = data.decode(enc)
            return text, enc
        except UnicodeDecodeError:
            continue
    return None, None


def check_and_fix(path: Path, write: bool = False) -> tuple[bool, str]:
    data = path.read_bytes()
    text, enc = decode_best(data)
    if text is None:
        return False, "undecodable"

    normalized = text.replace("\r\n", "\n")
    if enc == "utf-8" and normalized == text:
        return True, "utf-8-ok"

    if write:
        path.write_text(normalized, encoding="utf-8")
        return True, f"converted-{enc}-to-utf8"
    else:
        return False, f"would-convert-{enc}"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Verify and optionally convert repo files to UTF-8 (LF)")
    ap.add_argument("paths", nargs="*", default=["."], help="Files or directories to check")
    ap.add_argument("--write", action="store_true", help="Rewrite files to UTF-8 (and normalize LF)")
    ap.add_argument("--exts", default=".py,.md,.yml,.yaml,.json,.toml,.txt,.csv", help="Comma-separated extensions")
    args = ap.parse_args()

    ok = True
    exts = {e.strip().lower() for e in args.exts.split(',')}
    for root in args.paths:
        rootp = Path(root)
        it = [rootp] if rootp.is_file() else rootp.rglob("*")
        for p in it:
            if p.is_dir():
                continue
            if p.suffix.lower() not in exts:
                continue
            status_ok, info = check_and_fix(p, write=args.write)
            print(("[OK]   " if status_ok else "[WARN] ") + f"{p} -> {info}")
            ok = ok and status_ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
