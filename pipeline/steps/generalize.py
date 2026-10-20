""" RQ3 — Evaluate saved models on held-out unseen degradations (noise and contrast). """

# Extracted from the research notebook; original experiment logic retained.


# ==================== Original notebook cell 26 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [v6] UNSEEN DEGRADATIONS — evaluation only (no training)
#  Gaussian noise and contrast reduction: neither resembles blur, and no model
#  was trained on them. Answers "blur only helps because it matches the test".
#  Re-scores frozen encoders, Gabor, every saved adapter, and each blend at the
#  α chosen on validation. Full fine-tuning is excluded (weights not saved).
# ═══════════════════════════════════════════════════════════════════════════
import zlib
HOLDOUT_GRIDS = {
    'noise':    [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40],   # Gaussian σ on [0,1] pixels
    'contrast': [0.50, 0.30, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01],   # factor around image mean
}
HOLDOUT_FILE = EVAL_DIR / 'holdout_levels.json'

_degrade_v4 = degrade_pil.__wrapped__ if hasattr(degrade_pil, '__wrapped__') else degrade_pil
def degrade_pil(img, fam, lvl, _path=None):
    if fam not in HOLDOUT_GRIDS:
        return _degrade_v4(img, fam, lvl)
    x = transforms.functional.to_tensor(_RESIZE(img.convert('RGB')))
    if fam == 'noise':        # deterministic per image: same noise for every model
        seed = zlib.crc32((getattr(img, 'filename', '') or '').encode()) ^ int(lvl * 1e4)
        g = torch.Generator().manual_seed(seed)
        x = x + float(lvl) * torch.randn(x.shape, generator=g)
    else:
        m = x.mean(dim=(1, 2), keepdim=True)
        x = m + float(lvl) * (x - m)
    return x.clamp(0, 1)
degrade_pil.__wrapped__ = _degrade_v4

# ── Calibrate strengths once with frozen SigLIP (same rule as v4) ─────────
if HOLDOUT_FILE.exists():
    HCAL = json.load(open(HOLDOUT_FILE))
else:
    print("Calibrating unseen-degradation strengths with frozen SigLIP (one-time) …")
    m = build_encoder_v5('frozen', 'siglip'); curve = {f: [] for f in HOLDOUT_GRIDS}
    for k in range(N_FOLDS):
        r = robust_eval(lambda rc, f, l: enc_feats(m, 'siglip', rc, f, l), fold_records(k)[2], HOLDOUT_GRIDS)
        for f, lv in HOLDOUT_GRIDS.items(): curve[f].append([r[f'{f}_{j}'] for j in range(len(lv))])
    del m
    HCAL = {'grids': HOLDOUT_GRIDS, 'frozen_mrr': {}, 'levels': {}}
    for f, lv in HOLDOUT_GRIDS.items():
        mm = [_mrr(np.concatenate([curve[f][k][j] for k in range(N_FOLDS)])) for j in range(len(lv))]
        HCAL['frozen_mrr'][f] = mm
        inside = [j for j, v in enumerate(mm) if CALIB_RANGE[0] <= v <= CALIB_RANGE[1]]
        if len(inside) > MAX_LEVELS: inside = [inside[round(i)] for i in np.linspace(0, len(inside) - 1, MAX_LEVELS)]
        if len(inside) < 3:
            if sum(v < CALIB_RANGE[1] for v in mm) < 2:
                print(f"  ⚠ {f}: frozen SigLIP stays ≥ {CALIB_RANGE[1]} at every strength — family dropped"); continue
            inside = sorted(sorted(range(len(mm)), key=lambda j: abs(mm[j] - 0.65))[:3])
        HCAL['levels'][f] = [lv[j] for j in inside]
    json.dump(HCAL, open(HOLDOUT_FILE, 'w'), indent=2)
HLEVELS = HCAL['levels']
for f, lv in HCAL['grids'].items():
    used = set(HLEVELS.get(f, []))
    print(f"  {f:9s} " + '  '.join(f"{l}:{v:.3f}{'✓' if l in used else ''}" for l, v in zip(lv, HCAL['frozen_mrr'][f])))
HCONDS = [f'{f}_{j}' for f, lv in HLEVELS.items() for j in range(len(lv))]

# ── Jobs: (config name, fold, model builder) ─────────────────────────────
def _holdout_jobs():
    jobs = []
    for k in range(N_FOLDS):
        for e in ['siglip', 'dinov2', 'clip']:
            jobs.append((f'{e}:frozen', k, e, None, None, None, None))
        jobs.append(('gabor', k, None, None, None, None, None))
        for e in ['siglip', 'dinov2', 'clip']:
            for m_ in ['lora', 'dora', 'krona']:
                for a in ['none', 'fourier', 'blur']:
                    tag = v5_tag(e, m_, a, k); w = RUNS_DIR / f'{tag}_weights.pt'
                    if not w.exists(): continue
                    c = cfg_name(e, m_, a)
                    jobs.append((c, k, e, m_, w, 1.0, tag))
                    wf = RUNS_DIR / f'wise__{tag}.npz'
                    if wf.exists():
                        z = np.load(wf); ak = float(z['alphas'][int(np.nanargmax(z['val']))])
                        jobs.append((c + ' (blend)', k, e, m_, w, ak, tag))
    return jobs

def _hfile(c, k):
    return RUNS_DIR / f"holdout__{re.sub(r'[^A-Za-z0-9+]+', '_', c)}__fold{k}.npz"

def run_holdout(c, k, e, m_, w, alpha, tag):
    te = fold_records(k)[2]
    if c == 'gabor':
        tr, va, _ = fold_records(k); ref = gabor_feats_raw(tr + va); mu, sd = ref.mean(0), ref.std(0) + 1e-8
        def fn(rc, f, l):
            z = (gabor_feats_raw(rc, f, l) - mu) / sd
            return z / np.linalg.norm(z, axis=1, keepdims=True)
        res = robust_eval(fn, te, HLEVELS)
    else:
        if m_ is None:
            model = build_encoder_v5('frozen', e)
        elif alpha == 0.0:
            model = build_encoder_v5('frozen', e)           # blend chose the frozen model
        else:
            model = build_encoder_v5(m_, e); load_trainable_state(model, torch.load(w, map_location='cpu'))
            if alpha != 1.0: blend_(model, alpha)
        res = robust_eval(lambda rc, f, l: enc_feats(model, e, rc, f, l), te, HLEVELS)
        del model; torch.cuda.empty_cache()
    save_npz(_hfile(c, k), paths=np.array([r['path'] for r in te]), alpha=np.array(alpha if alpha is not None else np.nan),
             **{f'rr__{cc}': v for cc, v in res.items()})

todo = [j for j in _holdout_jobs() if not _hfile(j[0], j[1]).exists()]
print(f"\n{len(todo)} unseen-degradation evaluations to do")
t_hist = []
for j in todo:
    if not time_left_ok(1.5 * max(t_hist or [3])):
        print("⏸  Stopping to stay inside the session limit — run again with this output attached."); break
    t0 = time.time(); run_holdout(*j); t_hist.append((time.time() - t0) / 60)
    print(f"  ✔ {j[0]} fold {j[1]}  ({t_hist[-1]:.1f} min)")

# ── Table + tests ───────────────────────────────────────────────────────────
HR = {}
for c in dict.fromkeys(j[0] for j in _holdout_jobs()):
    files = [_hfile(c, k) for k in range(N_FOLDS)]
    if not all(f.exists() for f in files): continue
    rr = {cc: [] for cc in ['clean'] + HCONDS}
    for k, f in enumerate(files):
        d = np.load(f); assert list(d['paths']) == list(Q_PATHS[Q_FOLD == k])
        for cc in rr: rr[cc].append(d[f'rr__{cc}'])
    rr = {cc: np.concatenate(v) for cc, v in rr.items()}
    rr['unseen'] = np.nanmean(np.stack([rr[cc] for cc in HCONDS]), 0)
    for f in HLEVELS: rr[f'fam:{f}'] = np.nanmean(np.stack([rr[f'{f}_{j}'] for j in range(len(HLEVELS[f]))]), 0)
    HR[c] = rr

HTABLE = []
print(f"\nUnseen degradations (no model trained on them) · levels {HLEVELS}")
print(f"{'config':32s} " + ' '.join(f'{f:>9s}' for f in HLEVELS) + f" {'UNSEEN':>7s}  95% CI")
for c in [c for c in order if c in HR] + [c for c in HR if c not in order]:
    m_, lo, hi = ci_mean(HR[c]['unseen'], Q_LAB)
    HTABLE.append({'config': c, **{f: _mrr(HR[c]['fam:' + f]) for f in HLEVELS}, 'unseen': m_, 'ci_lo': lo, 'ci_hi': hi,
                   'unseen_ci': f"{m_:.3f} [{lo:.3f}, {hi:.3f}]"})
    print(f"{c:32s} " + ' '.join(f"{_mrr(HR[c]['fam:' + f]):9.4f}" for f in HLEVELS) + f" {m_:7.4f}  [{lo:.4f}, {hi:.4f}]")

HTESTS = []
def hfamily(name, pairs):
    pairs = [(a, b) for a, b in pairs if a in HR and b in HR]
    if not pairs: return
    res = [paired_test(HR[a]['unseen'], HR[b]['unseen'], Q_LAB) for a, b in pairs]
    for (a, b), r, pa in zip(pairs, res, holm([r['p_boot'] for r in res])):
        HTESTS.append({'family': name, 'a': a, 'b': b, **r, 'p_holm': float(pa)})
for e in ['siglip', 'dinov2', 'clip']:
    hfamily(f'unseen vs_frozen[{ENC_LABEL[e]}]', [(c, f'{e}:frozen') for c in HR if c.startswith(e + ':') and c != f'{e}:frozen'])
hp = []
for e in ['siglip', 'dinov2', 'clip']:
    for m_ in ['lora', 'dora']:
        b = f'{e}:{m_}'; hp += [(f'{b}+blur', b), (f'{b}+fourier', b), (f'{b}+blur', f'{b}+fourier')]
hfamily('unseen augmentation', hp)
for name in dict.fromkeys(t['family'] for t in HTESTS):
    print(f"\n[{name}]")
    for t in [t for t in HTESTS if t['family'] == name]:
        print(f"  {t['a']:30s} vs {t['b']:24s} Δ={t['delta']:+.4f} [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] "
              f"p_holm={t['p_holm']:.4f} {'*' if t['p_holm'] < 0.05 else 'ns'}")

write_csv(PAPER / 'table_unseen.csv', HTABLE)
write_tex(PAPER / 'table_unseen.tex', HTABLE, ['config'] + list(HLEVELS) + ['unseen_ci'],
          ['Configuration'] + list(HLEVELS) + ['Unseen MRR [95\\% CI]'],
          'Retrieval under degradations no model was trained on (Gaussian noise, contrast reduction).', 'tab:unseen')
write_csv(PAPER / 'tests_unseen.csv', HTESTS)
left = len([j for j in _holdout_jobs() if not _hfile(j[0], j[1]).exists()])
print(f"\nExports → {PAPER}: table_unseen.csv/.tex, tests_unseen.csv")
print("✅ unseen-degradation evaluation complete." if not left else f"⏳ {left} evaluations left — run again with this output attached.")
