"""Week 8: git kata. Builds throwaway repositories for you to practise on (provided, nothing to fill in).

    python exercises/week08/git_kata.py conflict  /tmp/kata-conflict
    python exercises/week08/git_kata.py bisect    /tmp/kata-bisect
    python exercises/week08/git_kata.py upstream  /tmp/kata-upstream
    python exercises/week08/git_kata.py check-bisect /tmp/kata-bisect <sha you found>

Each command prints the task. The lesson note "Week 8" walks through them.
"""

import subprocess
import sys
from pathlib import Path

ENV = {"GIT_AUTHOR_NAME": "kata", "GIT_AUTHOR_EMAIL": "kata@example.com", "GIT_COMMITTER_NAME": "kata", "GIT_COMMITTER_EMAIL": "kata@example.com"}


def git(repo: Path, *args: str) -> str:
    import os

    env = {**os.environ, **ENV}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def init(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=False)
    git(repo, "init", "-q", "-b", "main")


def commit(repo: Path, message: str, files: dict[str, str]) -> str:
    for name, text in files.items():
        (repo / name).write_text(text)
    git(repo, "add", *files)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


ATTENTION_V1 = """ATTN_TYPES = ["torch", "flex"]


def check(attn_type):
    assert attn_type in ATTN_TYPES, f"Invalid attention type: {attn_type}"
"""


def setup_conflict(repo: Path) -> None:
    init(repo)
    commit(repo, "add attention registry", {"attention.py": ATTENTION_V1})
    git(repo, "switch", "-q", "-c", "add-flash")
    commit(repo, "add flash backend", {"attention.py": ATTENTION_V1.replace('["torch", "flex"]', '["torch", "flex", "flash"]')})
    git(repo, "switch", "-q", "main")
    commit(repo, "add flash-varlen backend", {"attention.py": ATTENTION_V1.replace('["torch", "flex"]', '["torch", "flex", "flash-varlen"]')})
    print(
        f"""Repository ready at {repo}. Two branches changed the same line.

1. cd {repo} && git log --oneline --graph --all
2. git switch add-flash && git rebase main          -> CONFLICT
3. open attention.py, keep all four backends, remove the <<<<<<< ======= >>>>>>> markers
4. git add attention.py && git rebase --continue
5. git log --oneline --graph --all                   -> one straight line
Bonus: reset and do it again with `git merge` instead of rebase, and compare the graphs."""
    )


def setup_bisect(repo: Path) -> None:
    init(repo)
    body = "def window_mask(q, kv, window):\n    return abs(q - kv) <= window // 2\n"
    commit(repo, "add window mask", {"masks.py": body, "notes.txt": "v0\n"})
    for i in range(1, 25):
        files = {"notes.txt": f"v{i}\n"}
        if i == 13:
            files["masks.py"] = "def window_mask(q, kv, window):\n    return abs(q - kv) < window // 2\n"
        commit(repo, f"tweak {i}", files)
    test = "from masks import window_mask\nassert window_mask(0, 2, 4), 'edge of the window must be inside'\nprint('ok')\n"
    (repo / "test_masks.py").write_text(test)
    print(
        f"""Repository ready at {repo}. Somewhere in the last 24 commits the window lost its edge.
test_masks.py (untracked) passes on the first commit and fails on HEAD.

1. cd {repo} && python3 test_masks.py              -> AssertionError
2. git bisect start HEAD $(git rev-list --max-parents=0 HEAD)
3. git bisect run python3 test_masks.py            -> git finds the first bad commit for you
4. git bisect reset
5. python3 {Path(__file__).resolve()} check-bisect {repo} <sha>"""
    )


def check_bisect(repo: Path, sha: str) -> None:
    message = git(repo, "log", "-1", "--format=%s", sha)
    if message == "tweak 13":
        print(f"Correct: {sha[:8]} ({message}) changed <= to <.")
    else:
        print(f"Not quite: {sha[:8]} is '{message}'. Run the bisect again.")
        sys.exit(1)


MODEL_V1 = """DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 8
"""


def setup_upstream(root: Path) -> None:
    upstream = root / "upstream.git"
    root.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(upstream)], check=True)
    work = root / "maintainer"
    subprocess.run(["git", "clone", "-q", str(upstream), str(work)], check=True, capture_output=True)
    commit(work, "initial model", {"model.py": MODEL_V1})
    git(work, "push", "-q", "origin", "main")
    fork = root / "fork.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(upstream), str(fork)], check=True)
    mine = root / "mine"
    subprocess.run(["git", "clone", "-q", str(fork), str(mine)], check=True, capture_output=True)
    # Upstream edits the top of the file and you append at the bottom, so the rebase applies cleanly.
    commit(work, "upstream: widen the model", {"model.py": MODEL_V1.replace("DIM = 128", "DIM = 256")})
    git(work, "push", "-q", "origin", "main")
    print(
        f"""Ready at {root}: upstream.git (Sam's repo), fork.git (your GitHub fork), mine (your clone of the fork).
Upstream has moved on since you forked. This is exactly hepattn: origin = themrluke/hepattn, upstream = samvanstroud/hepattn.

1. cd {mine} && git remote -v                        -> only origin (your fork)
2. git remote add upstream {upstream}
3. git fetch upstream && git log --oneline --all --graph
4. git switch -c my-feature && echo 'NORM = "RMSNorm"' >> model.py && git commit -am "add norm"
5. git rebase upstream/main                          -> your commit now sits on top of upstream's
6. git push -u origin my-feature                     -> to your fork, ready for a pull request
7. git worktree add ../mine-talks -b talks           -> a second checkout of the same repo
   (your hepattn-talks worktree for the presentations branch is exactly this)"""
    )


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    command, path = sys.argv[1], Path(sys.argv[2]).resolve()  # absolute, so the printed steps work after `cd`
    if command == "conflict":
        setup_conflict(path)
    elif command == "bisect":
        setup_bisect(path)
    elif command == "check-bisect":
        check_bisect(path, sys.argv[3])
    elif command == "upstream":
        setup_upstream(path)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
