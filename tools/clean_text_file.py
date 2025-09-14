import re
import sys
from pathlib import Path


def clean_text(content: str) -> str:
    # Normalize newlines
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # Remove zero-width spaces and BOM if any sneaked into the middle (keep BOM only at start if present)
    content = content.replace('\ufeff', '')
    content = content.replace('\u200b', '')

    # Collapse 2+ blank lines to 1
    content = re.sub(r"\n{2,}", "\n", content)

    # Remove stray replacement-like '?' around CJK or punctuation contexts
    # - If '?' is surrounded by non-ASCII letters/numbers or punctuation, drop it.
    # This keeps legitimate question marks in English sentences, but removes typical mojibake.
    # Before: "初始化优?.." -> "初始化优.."
    content = re.sub(r"(?<=\w)\?(?=\W)|(?<=\W)\?(?=\w)|(?<=[\u4e00-\u9fff])\?|\?(?=[\u4e00-\u9fff])", "", content)

    # Also clean sequences like ' ?' or '? ' that are often artifacts
    content = re.sub(r"\s\?\s?", " ", content)

    # Trim trailing spaces on each line
    content = re.sub(r"[ \t]+(?=\n)", "", content)

    # Strip leading/trailing blank lines
    content = content.strip("\n") + "\n"
    return content


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/clean_text_file.py <file1> [<file2> ...]")
        sys.exit(2)

    for p in sys.argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] File not found: {p}")
            continue
        original = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(original)
        if cleaned != original:
            path.write_text(cleaned, encoding="utf-8")
            print(f"[CLEANED] {p}")
        else:
            print(f"[UNCHANGED] {p}")


if __name__ == "__main__":
    main()
