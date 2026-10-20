""" RQ2 — Controls: Gabor baseline, DINOv2, CLIP, blur augmentation, and WiSE-FT blending. """

# Extracted from the research notebook; original experiment logic retained.

# This stage may train additional control configurations and can be compute-intensive.

# NOTE: The shared *definitions* for the Gabor baseline and the v5 multi-encoder /
# weight-blending experiments (make_gabor_bank, gabor_descriptor, build_encoder_v5,
# enc_feats, val_score, blend_, v5_tag, ref_tag, wise_jobs, run_wise, run_cv_v5, _go,
# the V5_PLAN / WISE_ALPHAS / NORM constants, etc.) now live in setup.py so that the
# train/evaluate/export/generalize stages can resolve them without executing this
# compute-heavy stage first. This file keeps only the RQ2 *execution* blocks.
# The research logic is unchanged.


# ==================== Original notebook cell 23 (execution) ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [v5-2] RUNS — frozen references · weight blending · new training runs
#  Same safety rules as v4: skip finished, resume, time guard, atomic writes.
# ═══════════════════════════════════════════════════════════════════════════
for f in RUNS_DIR.glob('*.tmp.npz'): f.unlink()

# ── A. Frozen DINOv2 / CLIP on the same folds and degradation levels ─────
for enc in V5_REF_ENCODERS:
    for k in range(N_FOLDS):
        out = RUNS_DIR / f'{ref_tag(enc, k)}.npz'
        if out.exists(): continue
        te = fold_records(k)[2]
        m = build_encoder_v5('frozen', enc)
        res = robust_eval(lambda rc, f, l: enc_feats(m, enc, rc, f, l), te, LEVELS)
        save_npz(out, paths=np.array([r['path'] for r in te]), **{f'rr__{c}': v for c, v in res.items()})
        del m; torch.cuda.empty_cache()
    print(f"  ✔ frozen {enc}: {N_FOLDS} folds")

# ── B + C. Weight blending and training (definitions in setup.py) ─────────
if not _go():
    print(f"\n⏸  Stopping to stay inside the session limit. Save Version, add its output as an "
          f"input, and run again — finished work is kept.")
