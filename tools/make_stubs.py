"""Turn the solution files into exercise stubs.

The `solutions` branch is the source of truth. Every piece of code the learner
has to write sits between two marker comments:

    # >>> week04: hint for the learner
    ...solution code...
    # <<< week04

This script replaces each marked block with a TODO comment and a
``raise NotImplementedError``, keeping the indentation. The result is what lives
on `main`.

Usage:
    python tools/make_stubs.py            # rewrite files in place
    python tools/make_stubs.py --check    # only list the files that contain markers
"""

import argparse
import re
from pathlib import Path

START = re.compile(r"^(?P<indent>\s*)# >>> (?P<week>week\d\d)(?::\s*(?P<hint>.*))?$")
END = re.compile(r"^\s*# <<< week\d\d\s*$")
ROOTS = ("src", "exercises")


def stub_text(text: str, path: Path) -> tuple[str, int]:
    out, n_blocks = [], 0
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        m = START.match(lines[i].rstrip("\n"))
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, week, hint = m["indent"], m["week"], m["hint"] or "write this part"
        j = i + 1
        while j < len(lines) and not END.match(lines[j].rstrip("\n")):
            j += 1
        if j == len(lines):
            raise ValueError(f"{path}:{i + 1}: unterminated solution block")
        out.append(f"{indent}# TODO({week}): {hint}\n")
        out.append(f'{indent}raise NotImplementedError("{week} exercise ({path.name})")\n')
        n_blocks += 1
        i = j + 1
    return "".join(out), n_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    total = 0
    for top in ROOTS:
        for path in sorted((root / top).rglob("*.py")):
            text = path.read_text()
            new, n = stub_text(text, path.relative_to(root))
            if n == 0:
                continue
            total += n
            print(f"{n:3d} blocks  {path.relative_to(root)}")
            if not args.check:
                path.write_text(new)
    print(f"{total} exercise blocks")


if __name__ == "__main__":
    main()
