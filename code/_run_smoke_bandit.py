"""Wrapper: run _smoke_bandit_s5.py with output captured to a UTF-8 file."""
import sys, pathlib
out = pathlib.Path(__file__).parent / "runs" / "smoke_bandit_s5_out.txt"
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = f
    try:
        exec(compile(pathlib.Path("_smoke_bandit_s5.py").read_text("utf-8"),
                     "_smoke_bandit_s5.py", "exec"),
             {"__name__": "__main__", "__file__": "_smoke_bandit_s5.py"})
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
print("Output written to:", out)
