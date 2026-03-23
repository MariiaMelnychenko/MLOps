import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    py = sys.executable
    opt = ROOT / "src" / "optimize.py"
    for hpo in ("tpe", "random"):
        print(f"\n=== HPO sampler: {hpo} ===\n")
        subprocess.run(
            [str(py), str(opt), f"hpo={hpo}"],
            cwd=str(ROOT),
            check=True,
        )
    print("\nГотово")


if __name__ == "__main__":
    main()
