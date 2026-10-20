""" RQ1 — Evaluate robustness and report MRR across degradation families. """

# Extracted from the research notebook; original experiment logic retained.


# ==================== Original notebook cell 20 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [v4-3] ANALYSIS — robustness table, curves, paired tests
#  PRIMARY endpoint (fixed before seeing results): per-query MRR averaged over
#  ALL degraded conditions ("robust MRR"). Safe to run on partial results.
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

_run_re = re.compile(r'^(?P<m>\w+?)_f(?P<f>[01])_fold(?P<k>\d)_seed(?P<s>\d+)$')
CONDS = ['clean'] + [f'{fam}_{j}' for fam, lv in LEVELS.items() for j in range(len(lv))]
DEG   = CONDS[1:]

Q_PATHS = np.concatenate([[r['path'] for r in fold_records(k)[2]] for k in range(N_FOLDS)])
_by_path = {r['path']: r for r in nitik_records}
Q_LAB, _ = nitik_arrays([_by_path[p] for p in Q_PATHS])
Q_FOLD = np.concatenate([[k] * len(fold_records(k)[2]) for k in range(N_FOLDS)])

def _load(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}

configs = {}
def _assemble(name, per_fold):
    if sorted(per_fold) != list(range(N_FOLDS)):
        configs[name] = {'complete': False, 'folds': sorted(per_fold)}; return
    rr = {c: [] for c in CONDS}; seeds = []
    for k in range(N_FOLDS):
        runs = per_fold[k]; seeds.append(len(runs))
        for r in runs: assert list(r['paths']) == list(Q_PATHS[Q_FOLD == k]), "query order mismatch"
        for c in CONDS:
            rr[c].append(np.mean([r[f'rr__{c}'] for r in runs], 0))     # mean over seeds, per query
    rr = {c: np.concatenate(v) for c, v in rr.items()}
    rr['robust'] = np.nanmean(np.stack([rr[c] for c in DEG]), 0)
    for fam, lv in LEVELS.items():
        rr[f'fam:{fam}'] = np.nanmean(np.stack([rr[f'{fam}_{j}'] for j in range(len(lv))]), 0)
    configs[name] = {'complete': True, 'rr': rr, 'n_seeds': min(seeds),
                     'fold_robust': [float(np.nanmean(rr['robust'][Q_FOLD == k])) for k in range(N_FOLDS)]}

for ref in ['gabor', 'frozen']:
    _assemble(ref, {k: [_load(RUNS_DIR / f'{ref}_fold{k}.npz')] for k in range(N_FOLDS)
                    if (RUNS_DIR / f'{ref}_fold{k}.npz').exists()})
grouped = defaultdict(lambda: defaultdict(list))
for p in sorted(RUNS_DIR.glob('*.npz')):
    m = _run_re.match(p.stem)
    if m:
        grouped[f"{m['m']}{'+fourier' if m['f'] == '1' else ''}"][int(m['k'])].append(_load(p))
for name in [f"{m}{s}" for m in CV_METHODS for s in ('', '+fourier')]:
    if name in grouped: _assemble(name, grouped[name])

done = [n for n, c in configs.items() if c['complete']]

# ── Table ─────────────────────────────────────────────────────────────────
fams = list(LEVELS)
print(f"Nitik960 · 5-fold class-disjoint CV · {len(Q_PATHS)} queries · degraded query vs clean gallery")
print(f"Levels: {LEVELS}\n")
hdr = f"{'config':15s} {'clean':>6s} " + ' '.join(f"{f[:12]:>12s}" for f in fams) + \
      f" {'ROBUST MRR':>11s} {'95% CI (class boot.)':>21s} {'fold sd':>7s}"
print(hdr); print('─' * len(hdr))
table = {}
for n in done:
    rr = configs[n]['rr']
    m, lo, hi = ci_mean(rr['robust'], Q_LAB)
    print(f"{n:15s} {_mrr(rr['clean']):6.3f} " + ' '.join(f"{_mrr(rr['fam:'+f]):12.4f}" for f in fams) +
          f" {m:11.4f}   [{lo:.4f}, {hi:.4f}] {np.std(configs[n]['fold_robust'], ddof=1):7.4f}")
    table[n] = {'clean': _mrr(rr['clean']), 'robust': m, 'ci': [lo, hi],
                **{f'fam:{f}': _mrr(rr['fam:' + f]) for f in fams},
                'curve': {c: _mrr(rr[c]) for c in DEG}, 'n_seeds': configs[n]['n_seeds']}
for n, c in configs.items():
    if not c['complete']: print(f"{n:15s} incomplete — folds done: {c['folds']}")

# ── Paired tests (class-cluster bootstrap, Holm within each family) ───────
def _family(title, comps, key):
    comps = [(a, b) for a, b in comps if a in done and b in done]
    if not comps: return {}
    res = [paired_test(configs[a]['rr'][key], configs[b]['rr'][key], Q_LAB) for a, b in comps]
    adj = holm([r['p_boot'] for r in res])
    print(f"\n[{title}]  endpoint={key}  Δ = a − b, Holm over {len(comps)}")
    out = {}
    for (a, b), r, pa in zip(comps, res, adj):
        r['p_holm'] = float(pa)
        print(f"  {a:14s} vs {b:14s} Δ={r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
              f"p={r['p_boot']:.4f} p_holm={pa:.4f} {'*' if pa < 0.05 else 'ns'}"
              f"  (class Wilcoxon p={r['p_wilcoxon_class']:.4f})")
        out[f'{title}:{a}|{b}'] = r
    return out

trained = [n for n in done if n not in ('frozen', 'gabor')]
tests = {}
tests |= _family('vs_frozen', [(n, 'frozen') for n in trained], 'robust')
tests |= _family('fourier',   [(f'{m}+fourier', m) for m in CV_METHODS], 'robust')
tests |= _family('methods',   [(a, b) for i, a in enumerate(CV_METHODS) for b in CV_METHODS[i+1:]], 'robust')
tests |= _family('frozen_vs_gabor', [('frozen', 'gabor')], 'robust')
for f in fams:                                              # secondary: per degradation family
    tests |= _family(f'vs_frozen[{f}]', [(n, 'frozen') for n in trained], f'fam:{f}')

# ── Curves ────────────────────────────────────────────────────────────────
STYLE = {'frozen': ('#3d3d3a', '-'), 'gabor': ('#9a9992', ':'),
         'lora': ('#2a78d6', '-'), 'dora': ('#eb6834', '-'),
         'krona': ('#1baf7a', '-'), 'full': ('#eda100', '-')}
if done:
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fams):
        xs = ['clean'] + [str(l) for l in LEVELS[f]]
        for n in done:
            base = n.replace('+fourier', '')
            col, ls = STYLE[base]
            if n.endswith('+fourier'): ls = '--'
            ys = [_mrr(configs[n]['rr']['clean'])] + [_mrr(configs[n]['rr'][f'{f}_{j}']) for j in range(len(LEVELS[f]))]
            ax.plot(xs, ys, color=col, ls=ls, lw=2, marker='o', ms=4, label=n)
        ax.set_title(f, fontsize=10); ax.set_xlabel('strength (mild → severe)', fontsize=8)
        ax.grid(axis='y', color='#e1e0d9', lw=0.8); ax.tick_params(labelsize=8)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
    axes[0].set_ylabel('MRR (degraded query → clean gallery)', fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False, loc='lower left', ncol=2)
    plt.suptitle('Nitik960 retrieval under query degradation · dashed = + Fourier augmentation', fontsize=10)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / 'robustness_curves.png', dpi=150)
    plt.show()

with open(EVAL_DIR / 'nitik_robust_results.json', 'w') as fp:
    json.dump({'levels': LEVELS, 'calibration': CAL, 'table': table, 'tests': tests,
               'n_queries': int(len(Q_PATHS)),
               'protocol': 'Nitik960, 5-fold class-disjoint CV; query degraded, gallery clean; '
                           'own source group excluded; primary endpoint = per-query MRR averaged '
                           'over all degraded conditions; class-cluster bootstrap (10k); Holm per family'},
              fp, indent=2, default=float)
print(f"\nSaved → {EVAL_DIR / 'nitik_robust_results.json'} and robustness_curves.png")

_n_done = sum(1 for m in CV_METHODS for fr in CV_FOURIER for k in range(N_FOLDS) for s in CV_SEEDS
              if (RUNS_DIR / f"{m}_f{int(fr)}_fold{k}_seed{s}.npz").exists())
_n_all = len(CV_METHODS) * len(CV_FOURIER) * N_FOLDS * len(CV_SEEDS)
print("\n" + ("✅ ALL RUNS COMPLETE — these are the final numbers." if _n_done == _n_all else
      f"⏳ {_n_done}/{_n_all} training runs done. Save Version, add this version's output as an "
      f"input, and Save & Run All again."))
