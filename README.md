# Batik Research Pipeline — Research-question organization

This project reorganizes `batik-peft-thesis (1).ipynb` around the research questions and experimental purpose, rather than notebook version numbers.

## Research questions and stages

| Stage | Research focus | Script |
|---|---|---|
| `setup` | Shared data, model, split, metric, and degradation-calibration definitions | `pipeline/steps/setup.py` |
| `train` | **RQ1:** Train SigLIP adapters using the robustness objective | `pipeline/steps/train.py` |
| `evaluate` | **RQ1:** Build the robustness results, including MRR across degradation families | `pipeline/steps/evaluate.py` |
| `controls` | **RQ2:** Compare Gabor, DINOv2, CLIP, blur augmentation, and WiSE-FT weight blending | `pipeline/steps/controls.py` |
| `export` | Export paper tables and figures as CSV and LaTeX | `pipeline/steps/export.py` |
| `generalize` | **RQ3:** Evaluate on unseen Gaussian-noise and contrast degradations | `pipeline/steps/generalize.py` |
| `all` | Run every stage in dependency order | `run_research.py` |

## Run

Use a compatible Python/GPU environment. The source notebook expects datasets under `/kaggle/input` and writable output under `/kaggle/working`, so running on another machine may require adapting those paths.

```bash
pip install -r requirements.txt
python run_research.py --stage setup
```

Run a specific stage (the shared `setup` stage is automatically executed first):

```bash
python run_research.py --stage train
python run_research.py --stage evaluate
python run_research.py --stage controls
python run_research.py --stage export
python run_research.py --stage generalize
```

Run the full workflow, including training and controls:

```bash
python run_research.py --stage all
```

`setup` defines shared helpers and calibrates degradation levels; it does not start the main training loop. `train` and `controls` can be compute-intensive. `evaluate`, `export`, and `generalize` expect relevant saved experiment artifacts to be present; missing artifacts may lead to incomplete outputs or errors depending on the original notebook's safeguards.

Because the notebook shared globals across cells, some stages reuse names defined by earlier ones. The runner resolves this automatically: `setup` holds all shared definitions (data/model/metric helpers plus the Gabor baseline and the v5 multi-encoder / weight-blending helpers), so each stage runs `setup` first. `generalize` additionally runs `export` first, because the holdout evaluation reuses the analysis scaffolding (query arrays, `paper/` output dir, and CSV/LaTeX writers) that `export` builds — this matches the original notebook cell order (cell 24 before cell 26).

Optional CPU override:

```bash
python run_research.py --stage setup --allow-cpu
```

CPU training may be prohibitively slow. The original notebook's GPU guard, session/time limits, checkpointing, and output paths are otherwise retained.

## Project layout

```text
batik_research_project/
├── README.md
├── requirements.txt
├── run_research.py
└── pipeline/
    └── steps/
        ├── setup.py
        ├── train.py
        ├── evaluate.py
        ├── controls.py
        ├── export.py
        └── generalize.py
```

## Reproducibility and implementation notes

- This is a structural refactor; the research logic, hyperparameters, evaluation definitions, and export code are taken from the notebook rather than redesigned.
- The notebook shares global variables and function definitions across cells. The runner therefore executes the setup and requested stage in a shared namespace. The stage files are separated by research purpose but are not yet standalone importable Python packages.
- The controls stage retains the notebook's DINOv2/CLIP reference evaluation, blur-control experiments, adapter training and WiSE-FT blending routines. It may take substantial GPU time.
- Static validation confirms that all stage files parse and that every stage resolves the names it references in the shared namespace (checked against the runner's prerequisite order). It does **not** confirm dataset availability, dependency compatibility, successful model downloads, runtime completion, or numerical reproduction — those require a real GPU run with the datasets attached.
