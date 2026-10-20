""" Paper outputs — CSV/LaTeX tables and figures from available experiment artifacts. """

# Extracted from the research notebook; original experiment logic retained.


# ==================== Original notebook cell 24 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [v5-3] ANALYSIS + PAPER EXPORTS (items 1, 3, 4) — safe on partial results
#  Primary endpoint unchanged: robust MRR (per-query MRR over all degraded
#  conditions). Exports CSV + LaTeX tables and figures to EVAL_DIR/paper/.
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
PAPER = EVAL_DIR / 'paper'; PAPER.mkdir(exist_ok=True)
ENCS = ['siglip', 'dinov2', 'clip']
ENC_LABEL = {'siglip': 'SigLIP', 'dinov2': 'DINOv2', 'clip': 'CLIP'}
CONDS = ['clean'] + [f'{fam}_{j}' for fam, lv in LEVELS.items() for j in range(len(lv))]
DEG = CONDS[1:]; FAMS = list(LEVELS)

Q_PATHS = np.concatenate([[r['path'] for r in fold_records(k)[2]] for k in range(N_FOLDS)])
_by_path = {r['path']: r for r in nitik_records}
Q_LAB, _ = nitik_arrays([_by_path[p] for p in Q_PATHS])
Q_FOLD = np.concatenate([[k] * len(fold_records(k)[2]) for k in range(N_FOLDS)])
_ld = lambda p: {k: v for k, v in np.load(p, allow_pickle=True).items()}

# ── Collect every run as cfg → fold → [run dicts] ─────────────────────────
_v4 = re.compile(r'^(?P<m>[a-z]+)_f(?P<f>[01])_fold(?P<k>\d)_seed(?P<s>\d+)$')
_v5 = re.compile(r'^(?P<e>[a-z0-9]+)__(?P<m>[a-z]+)__(?P<a>[a-z]+)__fold(?P<k>\d)__seed(?P<s>\d+)$')
_rf = re.compile(r'^(?:(?P<e>[a-z0-9]+)__)?frozen(?:__|_)fold(?P<k>\d)$')
cfg_name = lambda e, m, a: f"{e}:{m}" + ('' if a == 'none' else f'+{a}')
runs, meta = defaultdict(lambda: defaultdict(list)), defaultdict(list)
for p in sorted(RUNS_DIR.glob('*.npz')):
    s = p.stem
    if s.startswith('wise__') or s.endswith('.tmp'): continue
    if (m := _v4.match(s)):
        c, k = cfg_name('siglip', m['m'], 'fourier' if m['f'] == '1' else 'none'), int(m['k'])
    elif (m := _v5.match(s)):
        c, k = cfg_name(m['e'], m['m'], m['a']), int(m['k'])
    elif (m := _rf.match(s)):
        c, k = f"{m['e'] or 'siglip'}:frozen", int(m['k'])
    elif s.startswith('gabor_fold'):
        c, k = 'gabor', int(s[-1])
    else:
        continue
    d = _ld(p); runs[c][k].append(d)
    if 'best_epoch' in d:
        meta[c].append({'fold': k, 'best_epoch': int(d['best_epoch']), 'val': float(d['best_val_mrr']),
                        'minutes': float(d['minutes']), 'nonfinite': int(d.get('nonfinite_steps', 0))})

def assemble(per_fold, key=lambda d, c: d[f'rr__{c}']):
    if sorted(per_fold) != list(range(N_FOLDS)): return None
    rr = {c: [] for c in CONDS}
    for k in range(N_FOLDS):
        for d in per_fold[k]: assert list(d['paths']) == list(Q_PATHS[Q_FOLD == k]), "query order mismatch"
        for c in CONDS: rr[c].append(np.mean([key(d, c) for d in per_fold[k]], 0))
    rr = {c: np.concatenate(v) for c, v in rr.items()}
    rr['robust'] = np.nanmean(np.stack([rr[c] for c in DEG]), 0)
    for f in FAMS: rr[f'fam:{f}'] = np.nanmean(np.stack([rr[c] for c in DEG if c.startswith(f + '_') and c[len(f)+1:].isdigit()]), 0)
    return rr

R = {c: rr for c, pf in runs.items() if (rr := assemble(pf)) is not None}
incomplete = {c: sorted(pf) for c, pf in runs.items() if c not in R}

# ── Weight blending: per α, plus α chosen per fold on validation ──────────
W, W_curve, W_alpha = {}, {}, {}
for c in [c for c in R if ':' in c and c.split(':')[1].split('+')[0] in ('lora', 'dora', 'krona')]:
    e, rest = c.split(':'); m, a = (rest.split('+') + ['none'])[:2]
    files = {k: RUNS_DIR / f"wise__{v5_tag(e, m, a, k)}.npz" for k in range(N_FOLDS)}
    if not all(f.exists() for f in files.values()) or f'{e}:frozen' not in R: continue
    alphas = [0.0] + WISE_ALPHAS + [1.0]
    def rr_at(k, a_):
        if a_ == 0.0: return runs[f'{e}:frozen'][k][0], 'rr__{}'
        if a_ == 1.0: return runs[c][k][0], 'rr__{}'
        return _ld(files[k]), f'rr__a{a_}__{{}}'
    per_alpha = {}
    for a_ in alphas:
        pf = {}
        for k in range(N_FOLDS):
            d, fmt = rr_at(k, a_)
            pf[k] = [{'paths': d['paths'], **{f'rr__{cc}': d[fmt.format(cc)] for cc in CONDS}}]
        per_alpha[a_] = assemble(pf)
    W_curve[c] = {a_: _mrr(per_alpha[a_]['robust']) for a_ in alphas}
    chosen, sel = [], {cc: [] for cc in CONDS}
    for k in range(N_FOLDS):
        val = _ld(files[k])['val']; ak = alphas[int(np.nanargmax(val))]; chosen.append(ak)
        for cc in CONDS: sel[cc].append(per_alpha[ak][cc][Q_FOLD == k])
    sel = {cc: np.concatenate(v) for cc, v in sel.items()}
    sel['robust'] = np.nanmean(np.stack([sel[cc] for cc in DEG]), 0)
    for f in FAMS: sel[f'fam:{f}'] = np.nanmean(np.stack([sel[cc] for cc in DEG if cc.rsplit('_', 1)[0] == f]), 0)
    W[c + ' (blend)'] = sel; W_alpha[c] = chosen
ALL = {**R, **W}

# ── Item 1a: main table ───────────────────────────────────────────────────
def row(c):
    rr = ALL[c]; m, lo, hi = ci_mean(rr['robust'], Q_LAB)
    return {'config': c, 'clean': _mrr(rr['clean']), **{f: _mrr(rr['fam:' + f]) for f in FAMS},
            'robust': m, 'ci_lo': lo, 'ci_hi': hi}
order = ['gabor'] if 'gabor' in ALL else []
for e in ENCS:
    if f'{e}:frozen' in ALL: order.append(f'{e}:frozen')
    order += sorted([c for c in ALL if c.startswith(e + ':') and c != f'{e}:frozen'],
                    key=lambda x: (x.split(':')[1].split('+')[0].split(' ')[0], x.endswith('(blend)'), x))
TABLE = [row(c) for c in order]
print(f"Nitik960 · 5-fold class-disjoint CV · {len(Q_PATHS)} queries · levels {LEVELS}\n")
print(f"{'config':32s} {'clean':>6s} " + ' '.join(f'{f[:12]:>12s}' for f in FAMS) + f" {'ROBUST':>7s}  95% CI")
for t in TABLE:
    print(f"{t['config']:32s} {t['clean']:6.3f} " + ' '.join(f"{t[f]:12.4f}" for f in FAMS) +
          f" {t['robust']:7.4f}  [{t['ci_lo']:.4f}, {t['ci_hi']:.4f}]")
for c, ks in incomplete.items(): print(f"{c:32s} incomplete — folds done: {ks}")

# ── Item 1b: paired tests (class-cluster bootstrap, Holm within family) ───
TESTS = []
def family(name, pairs, key='robust'):
    pairs = [(a, b) for a, b in pairs if a in ALL and b in ALL]
    if not pairs: return
    res = [paired_test(ALL[a][key], ALL[b][key], Q_LAB) for a, b in pairs]
    for (a, b), r, pa in zip(pairs, res, holm([r['p_boot'] for r in res])):
        TESTS.append({'family': name, 'endpoint': key, 'a': a, 'b': b, **r, 'p_holm': float(pa)})
trained = lambda e: [c for c in ALL if c.startswith(e + ':') and c != f'{e}:frozen']
for e in ENCS:
    family(f'vs_frozen[{ENC_LABEL[e]}]', [(c, f'{e}:frozen') for c in trained(e)])
aug_pairs = []
for e in ENCS:
    for m in ['lora', 'dora', 'krona', 'full']:
        b = f'{e}:{m}'
        aug_pairs += [(f'{b}+fourier', b), (f'{b}+blur', b), (f'{b}+fourier', f'{b}+blur')]
family('augmentation', aug_pairs)
family('methods[SigLIP]', [(f'siglip:{a}', f'siglip:{b}') for i, a in enumerate(['lora', 'dora', 'krona', 'full'])
                           for b in ['lora', 'dora', 'krona', 'full'][i + 1:]])
family('encoders_frozen', [('siglip:frozen', 'dinov2:frozen'), ('siglip:frozen', 'clip:frozen'),
                           ('dinov2:frozen', 'clip:frozen'), ('siglip:frozen', 'gabor')])
family('blending', [(c, c.replace(' (blend)', '')) for c in W] +
                   [(c, c.split(':')[0] + ':frozen') for c in W])
for f in FAMS:
    family(f'vs_frozen per degradation [{f}]',
           [(c, f'{c.split(":")[0]}:frozen') for e in ENCS for c in trained(e)], key=f'fam:{f}')
print()
for name in dict.fromkeys(t['family'] for t in TESTS):
    print(f"[{name}]")
    for t in [t for t in TESTS if t['family'] == name]:
        print(f"  {t['a']:30s} vs {t['b']:24s} Δ={t['delta']:+.4f} [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] "
              f"p_holm={t['p_holm']:.4f} {'*' if t['p_holm'] < 0.05 else 'ns'}")

# ── Blending summary ───────────────────────────────────────────────────────
if W_curve:
    print("\nRobust MRR by blend weight α (0 = frozen, 1 = fine-tuned) · α chosen on validation per fold:")
    for c, cur in W_curve.items():
        print(f"  {c:28s} " + '  '.join(f"α={a:g}: {v:.4f}" for a, v in cur.items()) + f"   chosen α: {W_alpha[c]}")

# ── Item 3: checkpoint-selection report ────────────────────────────────────
SEL = []
for c, rows_ in sorted(meta.items()):
    ep = np.array([r['best_epoch'] for r in rows_])
    SEL.append({'config': c, 'runs': len(rows_), 'best_epoch_median': float(np.median(ep)),
                'best_epoch_min': int(ep.min()), 'best_epoch_max': int(ep.max()),
                'share_epoch_le_2': float((ep <= 2).mean()),
                'val_at_selected': float(np.mean([r['val'] for r in rows_])),
                'minutes_mean': float(np.mean([r['minutes'] for r in rows_])),
                'nonfinite_steps': int(sum(r['nonfinite'] for r in rows_))})
print("\nCheckpoint selection (epoch with best degraded-query validation MRR):")
for s in SEL:
    print(f"  {s['config']:28s} runs={s['runs']}  epoch median {s['best_epoch_median']:.0f} "
          f"[{s['best_epoch_min']}–{s['best_epoch_max']}]  ≤2: {s['share_epoch_le_2']:.0%}  "
          f"val {s['val_at_selected']:.3f}  {s['minutes_mean']:.1f} min/run")

# ── Item 4: dataset / protocol / compute reporting ─────────────────────────
from collections import Counter
raw = Counter(r['dataset'] for r in all_records); ded = Counter(r['dataset'] for r in dedup_records)
lab, grp = nitik_arrays(nitik_records)
REPORT = {
    'datasets': {d: {'raw_images': raw[d], 'after_dedup': ded.get(d, 0),
                     'classes': len({r['class_name'] for r in dedup_records if r['dataset'] == d}),
                     'role': 'evaluation (5-fold, class-disjoint)' if d == NITIK else
                             ('training only' if d in EXTRA_TRAIN_DATASETS else 'dedup only')}
                 for d in sorted(raw)},
    'nitik': {'source_groups': int(len(set(grp))), 'scored_queries': int((~np.isnan(ALL['siglip:frozen']['clean'])).sum()) if 'siglip:frozen' in ALL else None,
              'excluded_classes': [NITIK_CLASSES[c] for c in range(len(NITIK_CLASSES)) if len(set(grp[lab == c])) < 2],
              'folds': [{'fold': k, **{n: len(x) for n, x in zip(['train_nitik', 'val', 'test'], fold_records(k))}}
                        for k in range(N_FOLDS)]},
    'degradation_levels': LEVELS, 'checkpoint_family': VAL_FAMILY,
    'training': {'epochs': CV_EPOCHS, 'steps_per_epoch': STEPS_PER_EPOCH, 'batch': f'{PK_P} classes x {PK_K} images',
                 'lr': LR_V5, 'optimizer': 'AdamW wd=1e-4, cosine', 'loss': 'SupCon tau=0.07',
                 'lora': 'r=16, alpha=32, dropout=0.05, q/k/v all blocks', 'krona': 'k=8, all vision Linear',
                 'amp': bool(USE_AMP), 'seeds_per_fold': len(CV_SEEDS)},
    'compute': {'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'training_gpu_hours': round(sum(r['minutes'] for rs in meta.values() for r in rs) / 60, 2),
                'blending_gpu_hours': round(sum(float(np.load(p)['minutes']) for p in RUNS_DIR.glob('wise__*.npz')) / 60, 2)},
    'software': {'python': platform.python_version(), 'torch': torch.__version__,
                 'transformers': __import__('transformers').__version__, 'peft': __import__('peft').__version__},
}

# ── Figures ────────────────────────────────────────────────────────────────
COL = {'frozen': '#3d3d3a', 'lora': '#2a78d6', 'dora': '#eb6834', 'krona': '#1baf7a', 'full': '#eda100'}
LS = {'none': '-', 'fourier': '--', 'blur': ':'}
encs_present = [e for e in ENCS if f'{e}:frozen' in R]
if encs_present:
    fig, axes = plt.subplots(len(encs_present), len(FAMS), figsize=(4.0 * len(FAMS), 3.0 * len(encs_present)),
                             sharey=True, squeeze=False)
    for i, e in enumerate(encs_present):
        for j, f in enumerate(FAMS):
            ax = axes[i, j]; xs = ['clean'] + [str(l) for l in LEVELS[f]]
            for c in [c for c in R if c.startswith(e + ':')]:
                m, a = (c.split(':')[1].split('+') + ['none'])[:2]
                ys = [_mrr(R[c]['clean'])] + [_mrr(R[c][f'{f}_{k}']) for k in range(len(LEVELS[f]))]
                ax.plot(xs, ys, color=COL[m], ls=LS[a], lw=2 if m == 'frozen' else 1.6, marker='o', ms=3,
                        label=f"{m}{'' if a == 'none' else ' + ' + a}")
            if e == 'siglip' and 'gabor' in R:
                ax.plot(xs, [_mrr(R['gabor']['clean'])] + [_mrr(R['gabor'][f'{f}_{k}']) for k in range(len(LEVELS[f]))],
                        color='#9a9992', ls=(0, (1, 2)), lw=1.4, label='Gabor')
            ax.set_title(f"{ENC_LABEL[e]} · {f}", fontsize=9); ax.tick_params(labelsize=7)
            ax.grid(axis='y', color='#e1e0d9', lw=0.8)
            for s in ('top', 'right'): ax.spines[s].set_visible(False)
        axes[i, 0].set_ylabel('MRR', fontsize=8)
    h, l = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, loc='lower center', ncol=min(len(l), 7), fontsize=8, frameon=False)
    fig.suptitle('Nitik960 retrieval, degraded query → clean gallery · solid = no aug, dashed = Fourier, dotted = blur', fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97)); fig.savefig(PAPER / 'fig_robustness_curves.png', dpi=200); plt.show()
if W_curve:
    fig, axes = plt.subplots(1, len(encs_present), figsize=(4.2 * len(encs_present), 3.2), sharey=True, squeeze=False)
    for i, e in enumerate(encs_present):
        ax = axes[0, i]
        for c, cur in [(c, v) for c, v in W_curve.items() if c.startswith(e + ':')]:
            m, a = (c.split(':')[1].split('+') + ['none'])[:2]
            ax.plot(list(cur), list(cur.values()), color=COL[m], ls=LS[a], marker='o', ms=3, lw=1.6,
                    label=f"{m}{'' if a == 'none' else ' + ' + a}")
        ax.set_title(ENC_LABEL[e], fontsize=9); ax.set_xlabel('α  (0 = frozen, 1 = fine-tuned)', fontsize=8)
        ax.grid(axis='y', color='#e1e0d9', lw=0.8); ax.legend(fontsize=7, frameon=False)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
    axes[0, 0].set_ylabel('robust MRR', fontsize=8)
    fig.tight_layout(); fig.savefig(PAPER / 'fig_weight_blending.png', dpi=200); plt.show()

# ── Exports ────────────────────────────────────────────────────────────────
def write_csv(path, rows_):
    if not rows_: return
    keys = list(dict.fromkeys(k for r in rows_ for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_)
def write_tex(path, rows_, cols, header, caption, label, fmt=lambda v: f'{v:.3f}' if isinstance(v, float) else str(v)):
    esc = lambda s: str(s).replace('_', r'\_').replace('+', r'+').replace('&', r'\&')
    lines = [r'\begin{table}[t]', r'\centering', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\begin{tabular}{l' + 'r' * (len(cols) - 1) + '}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    lines += [' & '.join(esc(fmt(r[c])) for c in cols) + r' \\' for r in rows_]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines))
for t in TABLE: t['robust_ci'] = f"{t['robust']:.3f} [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]"
write_csv(PAPER / 'table_main.csv', TABLE)
write_tex(PAPER / 'table_main.tex', TABLE, ['config', 'clean'] + FAMS + ['robust_ci'],
          ['Configuration', 'Clean'] + [f.replace('_', ' ') for f in FAMS] + ['Robust MRR [95\\% CI]'],
          'Nitik960 retrieval MRR, degraded query vs.\\ clean gallery (5-fold class-disjoint CV).', 'tab:main')
write_csv(PAPER / 'tests.csv', TESTS)
write_tex(PAPER / 'table_tests_vs_frozen.tex',
          [dict(t, ci=f"[{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}]", delta=f"{t['delta']:+.3f}", p=f"{t['p_holm']:.4f}")
           for t in TESTS if t['family'].startswith('vs_frozen[') or t['family'] in ('augmentation', 'blending')],
          ['family', 'a', 'b', 'delta', 'ci', 'p'], ['Family', 'A', 'B', '$\\Delta$ robust MRR', '95\\% CI', '$p_{\\mathrm{Holm}}$'],
          'Paired differences in robust MRR (class-cluster bootstrap, Holm-corrected within family).', 'tab:tests')
write_csv(PAPER / 'weight_blending.csv', [{'config': c, **{f'alpha_{a:g}': v for a, v in cur.items()},
                                          'chosen_alphas': W_alpha[c]} for c, cur in W_curve.items()])
write_csv(PAPER / 'epoch_selection.csv', SEL)
(PAPER / 'reporting.json').write_text(json.dumps(REPORT, indent=2, default=float))
json.dump({'table': TABLE, 'tests': TESTS, 'blending': {c: {'curve': W_curve[c], 'chosen': W_alpha[c]} for c in W_curve},
           'selection': SEL, 'report': REPORT}, open(EVAL_DIR / 'results_v5.json', 'w'), indent=2, default=float)
print(f"\nExports → {PAPER}: table_main.csv/.tex, tests.csv, table_tests_vs_frozen.tex, weight_blending.csv, "
      f"epoch_selection.csv, reporting.json, fig_robustness_curves.png, fig_weight_blending.png")

# ── Completion banner ──────────────────────────────────────────────────────
todo_train = sum(not (RUNS_DIR / f"{v5_tag(e, m, a, k)}.npz").exists() for e, m, a in V5_PLAN for k in range(N_FOLDS))
todo_wise = len(wise_jobs())
todo_ref = sum(not (RUNS_DIR / f"{ref_tag(e, k)}.npz").exists() for e in V5_REF_ENCODERS for k in range(N_FOLDS))
print("\n" + ("✅ v5 COMPLETE — these are the final numbers." if not (todo_train or todo_wise or todo_ref) else
      f"⏳ left: {todo_train} training runs, {todo_wise} blending evaluations, {todo_ref} references. "
      f"Save Version, add this version's output as an input, and Save & Run All again."))
