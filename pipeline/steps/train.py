"""RQ1 — Train SigLIP adapters using the robustness objective."""
# Requires setup.py to have executed in the shared pipeline namespace.
# Training execution block extracted from original notebook cell 19.

# ── Go ────────────────────────────────────────────────────────────────────
for f in RUNS_DIR.glob('*.tmp.npz'): f.unlink()        # leftovers from a killed session
run_reference_cv()

_hist = [float(np.load(p)['minutes']) for p in RUNS_DIR.glob('*_seed*.npz')]
queue = [(m, fr, k, s) for k in range(N_FOLDS) for m in CV_METHODS
         for fr in CV_FOURIER for s in CV_SEEDS
         if not (RUNS_DIR / f"{m}_f{int(fr)}_fold{k}_seed{s}.npz").exists()]
print(f"\n{len(queue)} training runs left.")
for m, fr, k, s in queue:
    elapsed_h = (time.time() - SESSION_START) / 3600
    est_h = (1.3 * max(_hist) if _hist else 45) / 60      # pessimistic estimate per run
    if elapsed_h + est_h > SESSION_BUDGET_H:
        print(f"\n⏸  Stopping to stay inside the session limit ({elapsed_h:.1f} h used, "
              f"next run ≈ {est_h * 60:.0f} min). Save this version, add its output as an "
              f"input, and run again — finished runs are kept.")
        break
    mins = run_cv(m, fr, k, s)
    if isinstance(mins, float):
        _hist.append(mins)
