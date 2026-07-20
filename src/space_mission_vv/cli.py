from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Space Mission Pipeline V&V Lab")
    parser.add_argument("--config", default="configs/demo.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    metrics = run_demo(
        config["output_dir"],
        int(config["seed"]),
        int(config["n_astrometry_trials"]),
        int(config["n_bolide_groups"]),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
