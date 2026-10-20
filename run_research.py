#!/usr/bin/env python3
"""Run the Batik research pipeline by research question.

Stages:
  setup      Shared data/model/evaluation definitions and degradation calibration
  train      Train SigLIP adapters for the robustness objective (RQ1)
  evaluate   Evaluate robustness MRR by degradation family (RQ1)
  controls   Gabor, DINOv2, CLIP, blur controls, and WiSE-FT blending (RQ2)
  export     Export paper tables and figures (CSV + LaTeX)
  generalize Hold out unseen degradations: Gaussian noise and contrast (RQ3)
  all        Run all stages in the declared order

Examples:
  python run_research.py --stage setup
  python run_research.py --stage train
  python run_research.py --stage evaluate
  python run_research.py --stage controls
  python run_research.py --stage export
  python run_research.py --stage generalize
  python run_research.py --stage all
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ROOT / "pipeline" / "steps"

STAGES = ["setup", "train", "evaluate", "controls", "export", "generalize", "all"]
PIPELINE_ORDER = ["setup", "train", "evaluate", "controls", "export", "generalize"]

# In-namespace prerequisites per stage. The notebook shared globals across cells,
# so some stages reuse names defined by earlier cells. `setup` holds the shared
# definitions; `generalize` (notebook cell 26) additionally reuses the analysis
# scaffolding built by `export` (cell 24): Q_PATHS/Q_LAB/Q_FOLD, PAPER, ENC_LABEL,
# cfg_name, write_csv/write_tex. Running `export` first reproduces that namespace
# and matches the original cell order (export runs before the holdout evaluation).
PREREQS = {
    "setup":      ["setup"],
    "train":      ["setup", "train"],
    "evaluate":   ["setup", "evaluate"],
    "controls":   ["setup", "controls"],
    "export":     ["setup", "export"],
    "generalize": ["setup", "export", "generalize"],
}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="setup",
                        help="Research stage to run (default: setup).")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Set ALLOW_CPU=1; training may be prohibitively slow without a GPU.")
    args = parser.parse_args()

    if args.allow_cpu:
        os.environ["ALLOW_CPU"] = "1"

    # The source notebook uses shared globals across cells. Execute the required
    # prerequisites in one namespace, preserving the original execution model.
    namespace = {
        "__name__": "__batik_research_pipeline__",
        "__package__": None,
        "__builtins__": __builtins__,
    }

    target_stages = PIPELINE_ORDER if args.stage == "all" else PREREQS[args.stage]

    for stage in target_stages:
        path = STEPS / f"{stage}.py"
        print(f"\n{'=' * 78}\nStage: {stage}\nFile: {path.relative_to(ROOT)}\n{'=' * 78}", flush=True)
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        exec(code, namespace, namespace)

    print("\nRequested pipeline stage(s) completed.", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
