""" Shared experiment setup: environment, data, models, evaluation helpers, and degradation calibration. """

# Extracted from the research notebook; original experiment logic retained.


# ==================== Original notebook cell 2 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [K1] Environment, paths, session clock, restore previous session's output
# ═══════════════════════════════════════════════════════════════════════════
import os, re, json, glob, time, shutil, zipfile, warnings, itertools
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

SESSION_START = time.time()
SESSION_BUDGET_H = 11.0      # Kaggle kills sessions at 12 h; stop starting new runs before that

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import imagehash
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda' and not os.environ.get('ALLOW_CPU'):
    raise RuntimeError("No GPU. Settings → Accelerator → GPU T4 x2 (or P100), then run again.")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
USE_AMP = device.type == 'cuda'

KAGGLE_INPUT = Path(os.environ.get('KAGGLE_INPUT', '/kaggle/input'))
WORK         = Path(os.environ.get('KAGGLE_WORK',  '/kaggle/working'))
DATA_TMP     = Path(os.environ.get('DATA_TMP',     '/tmp/batik_data'))   # not saved to output
OUT_DIR      = WORK / 'batik_v3'                                          # saved to output
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Restore everything a previous version of this notebook produced
# (Add Input → Notebook Output → this notebook's previous version).
restored = 0
for prev in KAGGLE_INPUT.glob('**/batik_v3'):
    if not prev.is_dir() or prev.resolve() == OUT_DIR.resolve():
        continue
    for f in prev.rglob('*'):
        if f.is_file():
            dst = OUT_DIR / f.relative_to(prev)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst); restored += 1
    print(f"Restored from previous output: {prev}")
n_done = len(list(OUT_DIR.glob('nitik_robust_v4/runs/*_seed*.npz')))
print(f"{restored} files restored · {n_done} finished training runs already present")


# ==================== Original notebook cell 3 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [K2] Find the datasets under /kaggle/input (zip OR auto-extracted folder)
# ═══════════════════════════════════════════════════════════════════════════
# Name of each dataset's original zip (as on your Google Drive). Kaggle either
# keeps the .zip or auto-extracts it into a folder with the same name; both work.
SOURCES = {
    'nitik960':    'Batik Nitik 960',
    'sarimbit120': 'Batik Nitik Sarimbit 120',   # only used so dedup matches the Colab run
    'keraton':     'Batik Keraton',
    'yogyakarta':  'Batik Yogyakarta',
    'lasem':       'Batik Lasem Motifs Collections (Gunung Ringgit, Kr',
}
REQUIRED = {'nitik960', 'keraton', 'yogyakarta', 'lasem'}
# Manual override if auto-detection picks the wrong thing, e.g.
# MANUAL_PATHS = {'lasem': '/kaggle/input/my-batik/Lasem'}
MANUAL_PATHS = {}

_norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())

def _find(stem):
    want, hits = _norm(stem), []
    for p in KAGGLE_INPUT.rglob('*'):
        if len(p.relative_to(KAGGLE_INPUT).parts) > 4:
            continue
        name = p.stem if p.suffix.lower() == '.zip' else p.name
        if _norm(name) == want and (p.is_dir() or p.suffix.lower() == '.zip'):
            hits.append(p)
    hits.sort(key=lambda p: (p.suffix.lower() != '.zip', len(p.parts)))  # prefer zip, then shallowest
    return hits[0] if hits else None

DATA_TMP.mkdir(parents=True, exist_ok=True)
DATASETS = {}
for name, stem in SOURCES.items():
    src = Path(MANUAL_PATHS[name]) if name in MANUAL_PATHS else _find(stem)
    dst = DATA_TMP / name
    if src is None:
        msg = f"'{stem}' not found under {KAGGLE_INPUT}"
        if name in REQUIRED:
            tops = sorted(str(p.relative_to(KAGGLE_INPUT)) for p in KAGGLE_INPUT.glob('*/*'))[:40]
            raise FileNotFoundError(msg + f". Add the dataset as an input or set MANUAL_PATHS.\n"
                                          f"Seen under input: {tops}")
        print(f"[warn] {msg} — skipping {name}"); continue
    if not dst.exists():
        if src.suffix.lower() == '.zip':
            print(f"Extracting {src.name} → {dst}")
            with zipfile.ZipFile(src) as z: z.extractall(dst)
        else:
            dst.symlink_to(src.resolve(), target_is_directory=True)
    DATASETS[name] = str(dst)
    print(f"  {name:12s} ← {src}")


# ==================== Original notebook cell 4 ====================
# ═══ [K3] Experiment constants ═══
IMG_SIZE        = 224
BATCH_SIZE      = 32
PHASH_THRESHOLD = 10
SEED            = 42
torch.manual_seed(SEED); np.random.seed(SEED)


# ==================== Original notebook cell 5 ====================
# ═══ [K3b] Scan datasets (original notebook code) ═══
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Datasets where the class label is in the filename, NOT a subfolder.
FILENAME_LABEL_DATASETS = {'nitik960', 'sarimbit120'}

# nitik960  : "1 Sekar Kemuning 1_rotate_90.jpg"
#   pattern   {class} {digit}[_{augmentation}]
#   strip     trailing " \d+[_\w+]*"  →  "1 Sekar Kemuning"
_NITIK_RE = re.compile(r'^(.+) \d+(?:_\w+)*$')

# sarimbit120 — two filename variants (both have a single uppercase instance letter):
#   WITH random id : "1 Sekar Kemuning A-hwNBbJ.jpg"
#   WITHOUT id     : "28 Sekar Kenthang A.jpg"   (no hyphen suffix at all)
#   Strip trailing " {LETTER}[-{id}]"  →  "1 Sekar Kemuning" / "28 Sekar Kenthang"
#   The id part is made optional with (?:-…)? so one regex covers both variants.
_SARIMBIT_RE = re.compile(r'^(.+) [A-Z](?:-[A-Za-z0-9]+)?$')

def class_from_filename(stem: str, ds_name: str) -> str | None:
    """Extract class label from a filename stem, using the dataset-specific pattern."""
    if ds_name == 'nitik960':
        m = _NITIK_RE.match(stem)
    elif ds_name == 'sarimbit120':
        # Primary pattern: "{class} {LETTER}-{id}"
        m = _SARIMBIT_RE.match(stem)
        # Fallback: some sarimbit files may use the nitik-style augmentation suffix
        if m is None:
            m = _NITIK_RE.match(stem)
    else:
        return None
    return m.group(1).strip() if m else None


def scan_lasem(root: str) -> list[dict]:
    """
    Custom scanner for the Lasem dataset whose structure is:
        <root>/motifs (isen isen)/<MotifClass>/all_motifs/*.jpg
    The motif class folder name is used as the class label.
    28x28 image tiles and the 'original images' folder are excluded.
    Falls back to scanning the class folder directly if all_motifs/ is absent.
    """
    root = Path(root)
    motifs_dir = root / 'motifs (isen isen)'
    if not motifs_dir.exists():
        # Try to find any subfolder that isn't 'original images'
        candidates = [d for d in root.iterdir()
                      if d.is_dir() and 'original' not in d.name.lower()]
        motifs_dir = candidates[0] if candidates else root

    class_dirs = sorted(d for d in motifs_dir.iterdir() if d.is_dir())
    classes    = [d.name for d in class_dirs]
    class2id   = {c: i for i, c in enumerate(classes)}
    records    = []

    for cls_dir in class_dirs:
        cls = cls_dir.name
        # Prefer the all_motifs subfolder; fall back to the class folder itself
        all_motifs = cls_dir / 'all_motifs'
        scan_dir   = all_motifs if all_motifs.exists() else cls_dir

        for p in sorted(scan_dir.rglob('*')):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if '28x28' in p.parts or '28x28' in p.name:
                continue          # skip small patch tiles
            records.append({
                'path':       str(p),
                'dataset':    root.name,
                'split':      'all',
                'class_name': cls,
                'class_id':   class2id[cls],
                'n_classes':  len(classes),
            })
    return records

def scan_dataset(root: str, use_filename_labels: bool = False, ds_name: str = '') -> list[dict]:
    """
    Scan a dataset and return records.
    use_filename_labels=True  → class from filename stem (nitik/sarimbit style)
    use_filename_labels=False → class from parent folder name (all other datasets)
    Supports both flat (root/img.jpg or root/class/img.jpg) and
    split layouts (root/train|val|test/img.jpg or root/train|val|test/class/img.jpg).
    """
    root = Path(root)
    name = root.name
    records = []

    # Detect train/val/test split subdirs
    splits = [d for d in root.iterdir() if d.is_dir() and d.name in ('train', 'val', 'test')]
    search_roots = splits if splits else [root]

    # Collect all image paths
    all_paths = [
        p for sr in search_roots
        for p in sorted(sr.rglob('*'))
        if p.suffix.lower() in IMG_EXTS
    ]

    if use_filename_labels:
        # Class label lives in the filename stem (pattern depends on dataset)
        def get_class(p: Path) -> str | None:
            return class_from_filename(p.stem, ds_name)
    else:
        # Class label is the immediate parent folder name
        def get_class(p: Path) -> str | None:
            cls = p.parent.name
            # If parent is the search root itself (flat layout), skip
            return cls if cls != root.name and cls not in ('train', 'val', 'test') else None

    classes = sorted(set(
        c for p in all_paths
        if (c := get_class(p)) is not None
    ))
    class2id = {c: i for i, c in enumerate(classes)}

    for p in all_paths:
        cls = get_class(p)
        if cls is None or cls not in class2id:
            continue
        # Determine split tag
        split_name = p.parts[len(root.parts)] if len(p.parts) > len(root.parts) + 1 else 'all'
        records.append({
            'path':       str(p),
            'dataset':    name,
            'split':      split_name,
            'class_name': cls,
            'class_id':   class2id[cls],
            'n_classes':  len(classes),
        })
    return records

all_records = []
dataset_info = {}
for ds_name, ds_path in DATASETS.items():
    if not Path(ds_path).exists():
        print(f"[SKIP] {ds_name} — path not found: {ds_path}")
        continue
    if ds_name == 'lasem':
        recs = scan_lasem(ds_path)
    else:
        use_fname = ds_name in FILENAME_LABEL_DATASETS
        recs = scan_dataset(ds_path, use_filename_labels=use_fname, ds_name=ds_name)
    all_records.extend(recs)
    dataset_info[ds_name] = {
        'n_images':  len(recs),
        'n_classes': recs[0]['n_classes'] if recs else 0,
        'classes':   sorted(set(r['class_name'] for r in recs)),
    }
    print(f"[OK] {ds_name:15s}  {len(recs):4d} images  {recs[0]['n_classes'] if recs else 0} classes")

print(f"\nTotal records (before dedup): {len(all_records)}")


# ── Raw image counts must match the Colab run (catches wrong zip / folder layout) ──
EXPECTED_RAW = {'nitik960': 960, 'sarimbit120': 120, 'keraton': 1798, 'yogyakarta': 3000, 'lasem': 11720}
_bad = {d: (dataset_info[d]['n_images'], n) for d, n in EXPECTED_RAW.items()
        if d in dataset_info and dataset_info[d]['n_images'] != n}
if _bad and not os.environ.get('SKIP_COUNT_CHECK'):
    raise RuntimeError(f"Image counts differ from the Colab run (found, expected): {_bad}. "
                       f"Check the dataset layout or set MANUAL_PATHS in the data cell.")
print("Raw counts OK" if not _bad else f"[count check skipped] {_bad}")


# ==================== Original notebook cell 6 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [K4] Near-duplicate removal — same rule as the original notebook
#  (pHash, Hamming ≤ PHASH_THRESHOLD, union-find, preferred_canon), but the
#  pairwise step is vectorised and hashing is parallel. Cached in OUT_DIR.
# ═══════════════════════════════════════════════════════════════════════════
from multiprocessing import Pool

_DEDUP_CACHE = OUT_DIR / 'dedup_groups.json'

def preferred_canon(members):
    for ds_pref in ['nitik960', 'batik300', 'keraton', 'yogyakarta', 'lasem', 'sarimbit120']:
        for m in members:
            if ds_pref in m:
                return m
    return members[0]

def _phash_bits(path):
    try:
        h = imagehash.phash(Image.open(path).convert('RGB')).hash.flatten()
        return int(''.join('1' if b else '0' for b in h), 2)
    except Exception:
        return None

all_paths_list = [r['path'] for r in all_records]
_cache_ok = False
if _DEDUP_CACHE.exists():
    with open(_DEDUP_CACHE) as f:
        dup_groups = json.load(f)
    members = {m for v in dup_groups.values() for m in v}
    _cache_ok = members <= set(all_paths_list)
    print(f"dedup cache: {len(dup_groups)} groups · paths valid: {_cache_ok}")
if not _cache_ok:
    print(f"Computing pHash for {len(all_paths_list)} images …")
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        hv = list(tqdm(pool.imap(_phash_bits, all_paths_list, chunksize=64), total=len(all_paths_list)))
    paths = [p for p, h in zip(all_paths_list, hv) if h is not None]
    H = np.array([h for h in hv if h is not None], dtype=np.uint64)
    parent = list(range(len(paths)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    _lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    for s in tqdm(range(0, len(H), 256), desc='pairwise Hamming'):
        x = H[s:s + 256, None] ^ H[None, :]
        d = _lut[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1)
        ii, jj = np.nonzero(d <= PHASH_THRESHOLD)
        for i, j in zip(ii + s, jj):
            if i < j:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[find(i)].append(p)
    dup_groups = {v[0]: v for v in groups.values() if len(v) > 1}
    with open(_DEDUP_CACHE, 'w') as f:
        json.dump(dup_groups, f)
    print(f"{len(dup_groups)} duplicate groups → {_DEDUP_CACHE}")

canonical_paths, path_to_canon = set(), {}
for members in dup_groups.values():
    chosen = preferred_canon(members)
    canonical_paths.add(chosen)
    for m in members: path_to_canon[m] = chosen
canonical_paths |= {p for p in all_paths_list if p not in path_to_canon}
dedup_records = [r for r in all_records if r['path'] in canonical_paths]

from collections import Counter
_after = Counter(r['dataset'] for r in dedup_records)
EXPECTED_AFTER_DEDUP = {'keraton': 1361, 'lasem': 4671, 'nitik960': 947, 'yogyakarta': 2362}  # Colab run
print(f"\nAfter dedup: {len(dedup_records)} images (removed {len(all_records) - len(dedup_records)})")
for ds, n in sorted(_after.items()):
    exp = EXPECTED_AFTER_DEDUP.get(ds)
    note = '' if exp is None or exp == n else f"   ⚠ Colab run had {exp}"
    print(f"  {ds:12s}: {n}{note}")


# ==================== Original notebook cell 7 ====================
# ═══ Transforms & dataset (original notebook code) ═══
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]

def get_transform(encoder_name: str = 'siglip', augment: bool = False) -> transforms.Compose:
    mean = SIGLIP_MEAN if 'siglip' in encoder_name else IMAGENET_MEAN
    std  = SIGLIP_STD  if 'siglip' in encoder_name else IMAGENET_STD
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    if augment:
        base = [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
        ]
    base.append(transforms.Normalize(mean, std))
    return transforms.Compose(base)


class BatikDataset(Dataset):
    """
    Unified dataset over a list of records (from scan_dataset + dedup).
    Class IDs are re-indexed globally within this dataset instance.
    """
    def __init__(self, records: list[dict], transform=None, dataset_filter: str | None = None):
        if dataset_filter:
            records = [r for r in records if r['dataset'] == dataset_filter]
        self.records = records
        self.transform = transform

        # Build global class mapping
        all_classes = sorted(set(f"{r['dataset']}::{r['class_name']}" for r in records))
        self.class2idx = {c: i for i, c in enumerate(all_classes)}
        self.idx2class = {i: c for c, i in self.class2idx.items()}
        self.n_classes = len(all_classes)

        for r in self.records:
            r['global_class_id'] = self.class2idx[f"{r['dataset']}::{r['class_name']}"]

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, r['global_class_id'], r['dataset'], r['path']


def make_global_split(records: list[dict], train_frac=0.6, val_frac=0.2, seed=42):
    """
    Stratified class-disjoint split: classes are split *per dataset*, then merged.
    Class key = "{dataset}::{class_name}" — same motif name in two datasets
    counts as two distinct classes (avoids cross-dataset label leakage).
    No class appears in more than one split.

    For datasets with ≥ 3 classes, at least 1 class is guaranteed in val and test.
    For datasets with < 3 classes, all classes go to train.

    Returns (train_records, val_records, test_records).
    """
    rng = np.random.default_rng(seed)

    # Collect classes per dataset
    ds_to_classes: dict[str, list[str]] = {}
    for r in records:
        ds  = r['dataset']
        key = f"{ds}::{r['class_name']}"
        ds_to_classes.setdefault(ds, [])
        if key not in ds_to_classes[ds]:
            ds_to_classes[ds].append(key)

    train_cls, val_cls, test_cls = set(), set(), set()

    for ds in sorted(ds_to_classes):
        cls_arr = np.array(sorted(ds_to_classes[ds]))
        rng.shuffle(cls_arr)
        n = len(cls_arr)

        if n < 3:
            # Too few classes to spare any for val/test — all go to train
            train_cls.update(cls_arr)
        else:
            # Enforce at least 1 class in val and test
            n_val   = max(1, int(n * val_frac))
            n_test  = max(1, n - int(n * train_frac) - n_val)
            n_train = n - n_val - n_test

            train_cls.update(cls_arr[:n_train])
            val_cls.update(cls_arr[n_train:n_train + n_val])
            test_cls.update(cls_arr[n_train + n_val:])

    key = lambda r: f"{r['dataset']}::{r['class_name']}"
    train = [r for r in records if key(r) in train_cls]
    val   = [r for r in records if key(r) in val_cls]
    test  = [r for r in records if key(r) in test_cls]

    total_cls = len(train_cls) + len(val_cls) + len(test_cls)
    print(f"Global split  — {total_cls} total classes "
          f"({len(train_cls)} train / {len(val_cls)} val / {len(test_cls)} test)")
    print(f"               {len(train)} train img / {len(val)} val img / {len(test)} test img")

    # Per-dataset breakdown
    print("\nPer-dataset breakdown:")
    for ds in sorted(set(r['dataset'] for r in records)):
        tc  = len(set(key(r) for r in train if r['dataset'] == ds))
        vc  = len(set(key(r) for r in val   if r['dataset'] == ds))
        tec = len(set(key(r) for r in test  if r['dataset'] == ds))
        ti  = sum(1 for r in train if r['dataset'] == ds)
        vi  = sum(1 for r in val   if r['dataset'] == ds)
        tei = sum(1 for r in test  if r['dataset'] == ds)
        print(f"  {ds:15s}  {tc}/{vc}/{tec} cls   {ti}/{vi}/{tei} img")
    return train, val, test


# ==================== Original notebook cell 8 ====================
# ═══ Encoder (original notebook code; SigLIP only) ═══
from transformers import (
    SiglipModel, SiglipProcessor,
    CLIPModel, CLIPProcessor,
    AutoModel, AutoProcessor,
)

class ViTEncoder(nn.Module):
    """
    Wraps HuggingFace vision encoders.
    Returns CLS token or mean-pooled patch tokens.
    """
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if 'siglip' in name:
            self._model = SiglipModel.from_pretrained('google/siglip-base-patch16-224')
            self.vision = self._model.vision_model
            self.dim = self._model.config.vision_config.hidden_size   # 768
            self.pool = 'mean'   # SigLIP uses mean pool of patch tokens
        elif 'dinov2' in name:
            self.vision = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.dim = 768
            self.pool = 'cls'
        elif 'clip' in name:
            self._model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
            self.vision = self._model.vision_model
            self.dim = self._model.config.vision_config.hidden_size   # 768
            self.pool = 'cls'
        else:
            raise ValueError(f"Unknown encoder: {name}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns (B, dim) feature vectors."""
        if 'dinov2' in self.name:
            out = self.vision.forward_features(pixel_values)
            # Newer DINOv2 checkpoints (≥ Apr 2024) return a dict;
            # older ones return a plain (B, N+1, dim) tensor.
            if isinstance(out, dict):
                if self.pool == 'cls':
                    feat = out['x_norm_clstoken']           # (B, dim)
                else:
                    feat = out['x_norm_patchtokens'].mean(1) # (B, dim)
            else:
                if self.pool == 'cls':
                    feat = out[:, 0]
                else:
                    feat = out[:, 1:].mean(1)
            return F.normalize(feat, dim=-1)
        else:
            out = self.vision(pixel_values=pixel_values, output_hidden_states=False)
            if self.pool == 'cls':
                feat = out.last_hidden_state[:, 0]
            else:
                # mean over patch tokens (skip CLS if present)
                # SigLIP has no CLS token — all positions are patch tokens
                patch_tokens = out.last_hidden_state
                feat = patch_tokens.mean(1)
            return F.normalize(feat, dim=-1)

    def forward_with_patch_tokens(self, pixel_values: torch.Tensor):
        """Returns (B, N, dim) patch token tensor for feature-domain filtering."""
        if 'dinov2' in self.name:
            out = self.vision.forward_features(pixel_values)
            if isinstance(out, dict):
                return out['x_norm_patchtokens']   # (B, N_patches, dim) — no CLS
            else:
                return out[:, 1:]    # drop CLS from old-style tensor output
        else:
            out = self.vision(pixel_values=pixel_values)
            hs = out.last_hidden_state
            if 'siglip' in self.name:
                return hs        # all patch tokens (SigLIP has no CLS)
            return hs[:, 1:]     # drop CLS for CLIP


# Load encoders
print("Loading encoders...")
encoders = {}
for enc_name in ['siglip']:
    print(f"  {enc_name}...", end=' ')
    enc = ViTEncoder(enc_name).eval().to(device)
    encoders[enc_name] = enc
    print(f"dim={enc.dim}  pool={enc.pool}")


# ==================== Original notebook cell 9 ====================
@torch.no_grad()
def extract_features(encoder: ViTEncoder, records: list[dict],
                     batch_size: int = BATCH_SIZE,
                     encoder_name: str = 'siglip') -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        features : (N, dim) float32
        labels   : (N,)    int64
        paths    : list[str]
    """
    ds = BatikDataset(records, transform=get_transform(encoder_name, augment=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    feats, labels, paths = [], [], []
    for imgs, lbls, _, pths in tqdm(loader, desc=f'extract {encoder_name}', leave=False):
        imgs = imgs.to(device)
        f = encoder(imgs).cpu().numpy()
        feats.append(f)
        labels.extend(lbls.numpy())
        paths.extend(pths)

    return np.vstack(feats), np.array(labels), paths


# ==================== Original notebook cell 10 ====================
class FourierAugTransform:
    """
    Fourier amplitude interpolation augmentation (applied at the tensor level).
    Mixes amplitude spectrum of two same-class images.
    alpha ~ Uniform(0, 1) per sample.
    """
    def __init__(self, alpha_range=(0.0, 1.0)):
        self.alpha_range = alpha_range

    def __call__(self, img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        alpha = np.random.uniform(*self.alpha_range)
        img_fft = torch.fft.fft2(img)
        ref_fft = torch.fft.fft2(ref)
        mixed_amp  = (1 - alpha) * img_fft.abs() + alpha * ref_fft.abs()
        img_phase  = torch.angle(img_fft)
        img_aug_fft = mixed_amp * torch.exp(1j * img_phase)
        return torch.fft.ifft2(img_aug_fft).real


class BatikDatasetWithFourier(Dataset):
    """Dataset that applies Fourier augmentation by pairing with a same-class reference."""
    def __init__(self, records, transform, use_fourier=False, encoder_name='siglip'):
        self.base = BatikDataset(records, transform=transform)
        self.use_fourier = use_fourier
        self.fourier_aug = FourierAugTransform() if use_fourier else None

        # Build class → index list for sampling same-class references
        self.class_to_idxs = defaultdict(list)
        for i, r in enumerate(self.base.records):
            self.class_to_idxs[r['global_class_id']].append(i)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, label, ds, path = self.base[idx]
        if self.use_fourier:
            # Sample a different image from the same class
            same_class = self.class_to_idxs[label]
            ref_idx = idx
            if len(same_class) > 1:
                while ref_idx == idx:
                    ref_idx = np.random.choice(same_class)
            ref_img, _, _, _ = self.base[ref_idx]
            img = self.fourier_aug(img, ref_img)
        return img, label, ds, path


# ==================== Original notebook cell 11 ====================
from peft import (
    get_peft_model, LoraConfig, TaskType,
    # DoRA is available as LoRA with use_dora=True in recent peft versions
)

def apply_lora(encoder: ViTEncoder, r=16, alpha=32, use_dora=False) -> nn.Module:
    """Apply LoRA (or DoRA) to the vision transformer attention layers."""
    target_modules = []
    # Find query/value projection layer names
    for name, _ in encoder.vision.named_modules():
        if any(x in name for x in ['query', 'key', 'value', 'q_proj', 'v_proj', 'k_proj']):
            target_modules.append(name)
    # Deduplicate
    target_modules = list(dict.fromkeys(target_modules))[:20]  # cap at 20

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules or ['query', 'value'],
        lora_dropout=0.05,
        bias='none',
        use_dora=use_dora,
    )
    return get_peft_model(encoder.vision, config)


class KroneckerAdapter(nn.Module):
    """
    Kronecker-structured adapter for a linear layer.
    ΔW = A1 ⊗ A2, where A1, A2 are (k×k) matrices.
    k=8 for a 64-factor approximation of the weight update.
    """
    def __init__(self, in_features: int, out_features: int, k: int = 8):
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        # Kronecker factors
        self.A1 = nn.Parameter(torch.randn(k, k) * 0.01)
        self.A2 = nn.Parameter(torch.randn(
            out_features // k, in_features // k) * 0.01)
        self.scaling = 1.0

    def get_delta_weight(self) -> torch.Tensor:
        return torch.kron(self.A1, self.A2)

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor,
                base_bias: torch.Tensor | None = None) -> torch.Tensor:
        dW = self.get_delta_weight()
        return F.linear(x, base_weight + dW, base_bias)


def apply_krona(encoder: nn.Module, k: int = 8) -> nn.Module:
    """
    Freeze all parameters, then attach KroneckerAdapters as side modules.
    Returns a wrapper that intercepts forward calls.
    """
    for p in encoder.parameters():
        p.requires_grad_(False)

    adapters = nn.ModuleDict()
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            if in_f % k == 0 and out_f % k == 0 and in_f >= k and out_f >= k:
                adapter_name = name.replace('.', '_')
                adapters[adapter_name] = KroneckerAdapter(in_f, out_f, k)
                module._krona_name = adapter_name

    # Monkey-patch forward on affected Linear layers
    def make_hooked_forward(mod, adapter):
        def hooked_forward(x):
            return adapter(x, mod.weight, mod.bias)
        return hooked_forward

    for name, module in encoder.named_modules():
        if hasattr(module, '_krona_name'):
            aname = module._krona_name
            module.forward = make_hooked_forward(module, adapters[aname])

    encoder._krona_adapters = adapters

    # Move adapters to the same device as the encoder
    try:
        enc_device = next(encoder.parameters()).device
        adapters.to(enc_device)
    except StopIteration:
        pass  # encoder has no parameters (shouldn't happen)

    return encoder

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: (N, dim) L2-normalized
        # labels:   (N,)
        N = features.shape[0]
        sim = features @ features.T / self.temperature        # (N, N)
        sim.fill_diagonal_(-1e9)

        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T).float()
        mask_pos.fill_diagonal_(0)

        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        n_pos = mask_pos.sum(1).clamp(min=1)
        loss = -(mask_pos * log_prob).sum(1) / n_pos
        return loss.mean()


def train_one_epoch(encoder, optimizer, loader, criterion, use_fourier=False):
    encoder.train()
    total_loss = 0.0
    for imgs, labels, _, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        features = encoder(imgs)
        loss = criterion(features, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(encoder, val_records, encoder_name='siglip') -> dict:
    # NOTE: this legacy helper depends on `instance_pool_gallery` and `evaluate`,
    # which came from a notebook cell that was not carried over into this refactor.
    # It is not used by any pipeline stage (the robustness protocol uses
    # `nitik_val_mrr` / `val_score` instead). Kept for reference; raises a clear
    # error instead of a bare NameError if ever called without those definitions.
    if 'instance_pool_gallery' not in globals() or 'evaluate' not in globals():
        raise NotImplementedError(
            "validate() requires `instance_pool_gallery` and `evaluate`, which are "
            "not defined in this refactored pipeline. Use `nitik_val_mrr` or "
            "`val_score` for validation instead.")
    encoder.eval()
    feats, labels, _ = extract_features(encoder, val_records, encoder_name=encoder_name)
    gf, gl = instance_pool_gallery(feats, labels)
    return evaluate(feats, labels, gf, gl)


# ==================== Original notebook cell 14 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [v3-1] Nitik960 corrected evaluation — setup, source groups, class folds
#  Requires cells above: imports, config, scan + dedup, split, encoders,
#  extract_features, evaluate/instance_pool_gallery, Fourier aug, PEFT defs.
# ═══════════════════════════════════════════════════════════════════════════
import random, time
from collections import Counter
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold

NITIK    = 'nitik960'
N_FOLDS  = 5
CV_SEED  = 42
EVAL_DIR = OUT_DIR / 'nitik_eval_v3'
(EVAL_DIR / 'runs').mkdir(parents=True, exist_ok=True)

# Source image = filename stem minus the augmentation suffix, e.g.
#   "1 Sekar Kemuning 1_rotate_90"  →  "1 Sekar Kemuning 1"
# All rotations of one photo share a source group and are never allowed to
# act as each other's positives at retrieval time.
_NITIK_SRC_RE = re.compile(r'^(.+ \d+)(?:_\w+)*$')

def nitik_source(path: str) -> str:
    stem = Path(path).stem
    m = _NITIK_SRC_RE.match(stem)
    return m.group(1).strip() if m else stem

nitik_records = sorted((r for r in dedup_records if r['dataset'] == NITIK),
                       key=lambda r: r['path'])
NITIK_CLASSES = sorted({r['class_name'] for r in nitik_records})
_CLS2IDX = {c: i for i, c in enumerate(NITIK_CLASSES)}

def nitik_arrays(recs):
    """(labels, groups) for a list of Nitik records — independent of the
    'global_class_id' field, which BatikDataset overwrites."""
    labels = np.array([_CLS2IDX[r['class_name']] for r in recs], dtype=np.int64)
    groups = np.array([nitik_source(r['path']) for r in recs])
    return labels, groups

# ── Sanity checks on the grouping (read these before trusting any number) ──
_lab, _grp = nitik_arrays(nitik_records)
_unmatched = [r['path'] for r in nitik_records if not _NITIK_SRC_RE.match(Path(r['path']).stem)]
_gsize = Counter(Counter(_grp).values())
_per_cls_groups = Counter(len(set(_grp[_lab == c])) for c in range(len(NITIK_CLASSES)))
print(f"Nitik960 after dedup: {len(nitik_records)} images · {len(NITIK_CLASSES)} classes · "
      f"{len(set(_grp))} source groups")
print(f"  group size → #groups      : {dict(sorted(_gsize.items()))}   (expect mostly 4)")
print(f"  source groups per class   : {dict(sorted(_per_cls_groups.items()))}")
print(f"  filenames not matching    : {len(_unmatched)}  {_unmatched[:3]}")
_classes_1grp = [NITIK_CLASSES[c] for c in range(len(NITIK_CLASSES)) if len(set(_grp[_lab == c])) < 2]
if _classes_1grp:
    print(f"  ⚠ classes with <2 source groups (no valid positive → excluded from metrics): {_classes_1grp}")

# ── 5-fold class split: fold k → test = F[k], val = F[k+1], train = rest ──
# (60/20/20 of classes per fold, same proportions as the original split;
#  every class is tested exactly once across the 5 folds.)
_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
FOLD_CLASSES = [set(np.array(NITIK_CLASSES)[te]) for _, te in _kf.split(NITIK_CLASSES)]

def fold_records(k: int):
    test_c = FOLD_CLASSES[k]
    val_c  = FOLD_CLASSES[(k + 1) % N_FOLDS]
    pick = lambda cs: [r for r in nitik_records if r['class_name'] in cs]
    train_c = set(NITIK_CLASSES) - test_c - val_c
    return pick(train_c), pick(val_c), pick(test_c)

for k in range(N_FOLDS):
    tr, va, te = fold_records(k)
    print(f"  fold {k}: train {len(tr):3d} img / val {len(va):3d} / test {len(te):3d}  "
          f"({len({r['class_name'] for r in te})} test classes)")


# ==================== Original notebook cell 15 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [v3-2] Group-aware retrieval metrics + class-cluster statistics
# ═══════════════════════════════════════════════════════════════════════════

def rr_instance(feats, labels, groups):
    """Leave-group-out instance retrieval. Query i ranks every other image;
    images from its own source group (itself + its rotations) are removed.
    Returns per-query reciprocal rank (NaN if the class has no other group)."""
    sim  = feats @ feats.T
    same = groups[:, None] == groups[None, :]
    pos  = (labels[:, None] == labels[None, :]) & ~same
    neg  =  labels[:, None] != labels[None, :]
    best_pos = np.where(pos, sim, -np.inf).max(1)
    n_better = (np.where(neg, sim, -np.inf) > best_pos[:, None]).sum(1)
    return np.where(pos.any(1), 1.0 / (1 + n_better), np.nan)


def rr_prototype(feats, labels, groups):
    """Prototype (class-mean) retrieval, the corrected version of
    instance_pool_gallery + evaluate: the query's own class prototype is
    rebuilt WITHOUT the query's source group. Other prototypes are unchanged."""
    classes, cidx = np.unique(labels, return_inverse=True)
    _, gidx = np.unique(groups, return_inverse=True)
    S = np.zeros((len(classes), feats.shape[1])); np.add.at(S, cidx, feats)
    G = np.zeros((gidx.max() + 1, feats.shape[1])); np.add.at(G, gidx, feats)
    P = S / np.linalg.norm(S, axis=1, keepdims=True)
    sims = feats @ P.T
    own   = S[cidx] - G[gidx]
    own_n = np.linalg.norm(own, axis=1)
    valid = own_n > 1e-8
    own_sim = (feats * own).sum(1) / np.maximum(own_n, 1e-8)
    sims[np.arange(len(labels)), cidx] = -np.inf
    n_better = (sims > own_sim[:, None]).sum(1)
    return np.where(valid, 1.0 / (1 + n_better), np.nan)


def summarize(rr) -> dict:
    v = rr[~np.isnan(rr)]
    return {'mrr': float(v.mean()), 'r@1': float((v == 1).mean()),
            'r@5': float((v >= 0.2).mean()), 'n': int(len(v))}


def _cluster_boot(values, clusters, n_boot=10_000, seed=0):
    """Bootstrap the mean by resampling whole classes (queries of one class —
    and rotations of one photo — are correlated, so resampling single
    queries would give CIs that are too narrow)."""
    ok = ~np.isnan(values)
    v, c = values[ok], clusters[ok]
    _, ci = np.unique(c, return_inverse=True)
    sums, cnts = np.bincount(ci, weights=v), np.bincount(ci)
    idx  = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(v.mean()), boot


def ci_mean(rr, clusters):
    m, boot = _cluster_boot(rr, clusters)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return m, float(lo), float(hi)


def paired_test(rr_a, rr_b, clusters) -> dict:
    """a − b on the SAME queries. Class-cluster bootstrap CI + p, and a
    Wilcoxon signed-rank test on per-class mean RR as a second opinion."""
    d = rr_a - rr_b
    m, boot = _cluster_boot(d, clusters)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_boot = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    p_boot = max(p_boot, 1 / len(boot))            # resolution floor
    ok = ~np.isnan(d)
    cls = np.unique(clusters[ok])
    ca = np.array([np.nanmean(rr_a[clusters == c]) for c in cls])
    cb = np.array([np.nanmean(rr_b[clusters == c]) for c in cls])
    try:
        p_w = float(wilcoxon(ca, cb).pvalue) if np.any(ca != cb) else 1.0
    except ValueError:
        p_w = float('nan')
    return {'delta': m, 'ci_lo': float(lo), 'ci_hi': float(hi),
            'p_boot': float(p_boot), 'p_wilcoxon_class': p_w, 'n_classes': int(len(cls))}


def holm(pvals):
    """Holm–Bonferroni adjusted p-values (monotone), in the input order."""
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[i]))
        adj[i] = running
    return adj


# ==================== Original notebook cell 16 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [v3-3] Model construction / loading (reuses the notebook's own classes)
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


def apply_lora_v2(encoder: ViTEncoder, r=16, alpha=32, use_dora=False):
    """FIXED LoRA/DoRA placement. The original apply_lora() kept only the first
    20 matching module names, i.e. q/k/v of blocks 0–5 plus k/v of block 6 —
    blocks 7–11 got no adapter at all. This targets q/k/v in all 12 blocks."""
    targets = [n for n, m in encoder.vision.named_modules()
               if isinstance(m, nn.Linear) and n.split('.')[-1] in ('q_proj', 'k_proj', 'v_proj')]
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=targets,
                     lora_dropout=0.05, bias='none', use_dora=use_dora)
    return get_peft_model(encoder.vision, cfg)


def build_encoder(method: str, lora_fn=None) -> ViTEncoder:
    """Fresh SigLIP ViTEncoder with the adapter for `method` attached exactly as
    in training. Pass lora_fn=apply_lora to rebuild the ORIGINAL (v1) runs."""
    lora_fn = lora_fn or apply_lora_v2
    enc = deepcopy(encoders['siglip']).to(device)
    if method == 'frozen':
        for p in enc.parameters(): p.requires_grad_(False)
    elif method == 'lora':
        lora_fn(enc, r=16, alpha=32, use_dora=False)
    elif method == 'dora':
        lora_fn(enc, r=16, alpha=32, use_dora=True)
    elif method == 'krona':
        apply_krona(enc.vision, k=8)
    elif method == 'full':
        for p in enc.parameters(): p.requires_grad_(True)
    else:
        raise ValueError(method)
    # The text tower / logit scale are never used by ViTEncoder.forward.
    for n, p in enc.named_parameters():
        if not n.startswith('_model.vision_model'):
            p.requires_grad_(False)
    return enc


def trainable_state(enc) -> dict:
    return {n: p.detach().cpu().clone() for n, p in enc.named_parameters() if p.requires_grad}


def load_trainable_state(enc, state: dict):
    res = enc.load_state_dict(state, strict=False)
    assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys[:5]}"
    loaded = set(state) - set(res.missing_keys)
    assert loaded == set(state), "some saved tensors were not loaded"


@torch.no_grad()
def embed(enc, recs) -> np.ndarray:
    """Features in record order, via the notebook's own transform + forward.
    Records are copied because BatikDataset rewrites 'global_class_id'."""
    enc.eval()
    feats, _, paths = extract_features(enc, [dict(r) for r in recs], encoder_name='siglip')
    assert list(paths) == [r['path'] for r in recs]
    return feats


# ==================== Original notebook cell 18 ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [v4-1] Degraded-query retrieval: definitions + one-time level calibration
#  Query = degraded image · gallery = clean images · positives = same class,
#  different source photo (own rotations excluded, as before).
# ═══════════════════════════════════════════════════════════════════════════
import io

EVAL_DIR = OUT_DIR / 'nitik_robust_v4'
RUNS_DIR = EVAL_DIR / 'runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LEVELS_FILE = EVAL_DIR / 'degradation_levels.json'

# Candidate strengths, mild → severe. Calibration keeps the ones where frozen
# SigLIP is neither saturated nor destroyed.
GRIDS = {
    'lowpass':      [0.20, 0.15, 0.10, 0.075, 0.05, 0.035, 0.025, 0.015, 0.01],  # fraction of Nyquist kept
    'lowpass_gray': [0.20, 0.15, 0.10, 0.075, 0.05, 0.035, 0.025, 0.015, 0.01],  # same, colour removed
    'down':         [2, 3, 4, 6, 8, 12, 16, 24, 32],                              # down-then-up factor
    'jpeg':         [30, 15, 10, 7, 5, 3, 2, 1],                                  # JPEG quality
}
CALIB_RANGE   = (0.30, 0.97)    # frozen MRR window a level must fall in
MAX_LEVELS    = 5
VAL_FAMILY_PREFERENCE = ['lowpass', 'lowpass_gray', 'down', 'jpeg']

_RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE))       # identical to get_transform
_NORM   = transforms.Normalize(SIGLIP_MEAN, SIGLIP_STD)

def _lowpass(x: torch.Tensor, frac: float) -> torch.Tensor:
    """Same mask as the notebook's radial_lowpass: keep radius ≤ frac × Nyquist."""
    H, W = x.shape[-2:]
    fy, fx = torch.meshgrid(torch.fft.fftfreq(H), torch.fft.fftfreq(W), indexing='ij')
    mask = (torch.sqrt(fy**2 + fx**2) <= frac * 0.5).to(x.dtype)
    return torch.fft.ifft2(torch.fft.fft2(x) * mask).real.clamp(0, 1)

def degrade_pil(img: Image.Image, fam: str, lvl):
    """Returns a [0,1] tensor (3,H,W) of the degraded 224×224 image."""
    img = _RESIZE(img.convert('RGB'))
    if fam == 'down':
        s = max(2, round(IMG_SIZE / lvl))
        img = img.resize((s, s), Image.BICUBIC).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    elif fam == 'jpeg':
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=int(lvl)); buf.seek(0)
        img = Image.open(buf).convert('RGB')
    elif fam == 'lowpass_gray':
        img = img.convert('L').convert('RGB')
    x = transforms.functional.to_tensor(img)
    if fam in ('lowpass', 'lowpass_gray'):
        x = _lowpass(x, float(lvl))
    return x

class DegradedDataset(Dataset):
    def __init__(self, recs, fam='clean', lvl=None, gray=False):
        self.paths, self.fam, self.lvl, self.gray = [r['path'] for r in recs], fam, lvl, gray
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = degrade_pil(Image.open(self.paths[i]), self.fam, self.lvl)
        if self.gray:                                        # Gabor input: (1,H,W) in [0,1]
            return (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
        return _NORM(x)

def _loader(ds):
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=min(4, os.cpu_count() or 1))

@torch.no_grad()
def siglip_feats(enc, recs, fam='clean', lvl=None) -> np.ndarray:
    enc.eval()
    out = [enc(x.to(device)).float().cpu().numpy() for x in _loader(DegradedDataset(recs, fam, lvl))]
    return np.vstack(out)

@torch.no_grad()
def gabor_feats_raw(recs, fam='clean', lvl=None) -> np.ndarray:
    out = [gabor_descriptor(x.to(device)).cpu().numpy()
           for x in _loader(DegradedDataset(recs, fam, lvl, gray=True))]
    return np.vstack(out).astype(np.float32)

def rr_cross(qf, gf, labels, groups):
    """Query i (possibly degraded) vs the clean gallery of the same images.
    Own source group excluded; returns per-query reciprocal rank."""
    sim  = qf @ gf.T
    same = groups[:, None] == groups[None, :]
    pos  = (labels[:, None] == labels[None, :]) & ~same
    neg  =  labels[:, None] != labels[None, :]
    best = np.where(pos, sim, -np.inf).max(1)
    n_better = (np.where(neg, sim, -np.inf) > best[:, None]).sum(1)
    return np.where(pos.any(1), 1.0 / (1 + n_better), np.nan)

def robust_eval(feat_fn, recs, levels: dict) -> dict:
    """feat_fn(recs, fam, lvl) → L2-normalised features. Returns {cond: rr array}."""
    lab, grp = nitik_arrays(recs)
    g = feat_fn(recs, 'clean', None)
    out = {'clean': rr_cross(g, g, lab, grp)}
    for fam, lvls in levels.items():
        for j, l in enumerate(lvls):
            out[f'{fam}_{j}'] = rr_cross(feat_fn(recs, fam, l), g, lab, grp)
    return out

_mrr = lambda rr: float(np.nanmean(rr))

# ── Calibration (frozen SigLIP only, once; reused by every later session) ──
if LEVELS_FILE.exists():
    CAL = json.load(open(LEVELS_FILE))
    print(f"Loaded degradation levels from {LEVELS_FILE.name}")
else:
    print("Calibrating degradation strengths with frozen SigLIP (one-time, ~5 min) …")
    enc = build_encoder('frozen')
    curve = {fam: [] for fam in GRIDS}
    for k in range(N_FOLDS):                          # same per-fold galleries as evaluation
        te = fold_records(k)[2]
        r = robust_eval(lambda rc, f, l: siglip_feats(enc, rc, f, l), te, GRIDS)
        for fam, lv in GRIDS.items():
            curve[fam].append([r[f'{fam}_{j}'] for j in range(len(lv))])
    del enc; torch.cuda.empty_cache()
    CAL = {'grids': GRIDS, 'range': CALIB_RANGE, 'frozen_mrr': {}, 'levels': {}}
    for fam, lv in GRIDS.items():
        m = [_mrr(np.concatenate([curve[fam][k][j] for k in range(N_FOLDS)])) for j in range(len(lv))]
        CAL['frozen_mrr'][fam] = m
        inside = [j for j, v in enumerate(m) if CALIB_RANGE[0] <= v <= CALIB_RANGE[1]]
        if len(inside) > MAX_LEVELS:
            inside = [inside[round(i)] for i in np.linspace(0, len(inside) - 1, MAX_LEVELS)]
        if len(inside) < 3:
            below = [j for j, v in enumerate(m) if v < CALIB_RANGE[1]]
            if len(below) < 2:
                print(f"  ⚠ {fam}: frozen stays ≥ {CALIB_RANGE[1]} even at the strongest level — family dropped")
                continue
            inside = sorted(sorted(range(len(m)), key=lambda j: abs(m[j] - 0.65))[:3])
        CAL['levels'][fam] = [lv[j] for j in inside]
    json.dump(CAL, open(LEVELS_FILE, 'w'), indent=2)

LEVELS = CAL['levels']
VAL_FAMILY = next(f for f in VAL_FAMILY_PREFERENCE if f in LEVELS)
print("\nFrozen SigLIP MRR by strength (✓ = used):")
for fam, lv in CAL['grids'].items():
    used = set(LEVELS.get(fam, []))
    print(f"  {fam:13s} " + '  '.join(f"{l}:{v:.3f}{'✓' if l in used else ''}"
                                     for l, v in zip(lv, CAL['frozen_mrr'][fam])))
print(f"\nLevels used: {LEVELS}\nCheckpoint selection family: {VAL_FAMILY}")


# Training helper definitions from original notebook cell 19; execution loop is in train.py.
# ═══════════════════════════════════════════════════════════════════════════
#  [v4-2] TRAINING — retrain per class fold; select & evaluate on DEGRADED queries
#  Fail-safe: finished runs are skipped · a run in progress resumes from its
#  last 5-epoch checkpoint · no new run starts if it might not finish before
#  SESSION_BUDGET_H · mixed precision (fp16) on GPU, loss computed in fp32.
# ═══════════════════════════════════════════════════════════════════════════
CV_METHODS   = ['lora', 'dora', 'krona', 'full']
CV_FOURIER   = [False, True]
CV_SEEDS     = [0]            # 5 folds already = 5 independent runs per config
CV_EPOCHS    = 20
STEPS_PER_EPOCH = 100
LR_MAP_V3    = {'lora': 3e-4, 'dora': 3e-4, 'krona': 1e-3, 'full': 1e-5}

EXTRA_TRAIN_DATASETS = ['keraton', 'lasem', 'yogyakarta']
extra_train = [r for r in dedup_records if r['dataset'] in EXTRA_TRAIN_DATASETS]

BALANCED_SAMPLER = True       # P×K batches so Nitik classes get positive pairs
PK_P, PK_K = 8, 4             # 8 × 4 = 32 = BATCH_SIZE
SAVE_FULL_FT_WEIGHTS = False  # ~350 MB per run


class PKBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, P, K, n_batches, seed):
        self.by_cls = defaultdict(list)
        for i, l in enumerate(labels):
            self.by_cls[int(l)].append(i)
        self.classes = [c for c, v in self.by_cls.items() if len(v) >= 2]
        self.P, self.K, self.n, self.seed, self.epoch = P, K, n_batches, seed, 0
    def __len__(self): return self.n
    def __iter__(self):
        rng = np.random.default_rng(self.seed * 10_007 + self.epoch); self.epoch += 1
        for _ in range(self.n):
            batch = []
            for c in rng.choice(self.classes, self.P, replace=False):
                idx = self.by_cls[c]
                batch.extend(rng.choice(idx, self.K, replace=len(idx) < self.K).tolist())
            yield batch


def train_one_epoch_amp(enc, opt, loader, crit, scaler):
    enc.train()
    params = [p for p in enc.parameters() if p.requires_grad]
    total, n, bad = 0.0, 0, 0
    for imgs, labels, _, _ in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
            feats = enc(imgs)
        loss = crit(feats.float(), labels)   # fp32: SupCon's −1e9 mask overflows fp16
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            bad += 1; continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        total += loss.item(); n += 1
    return total / max(n, 1), bad


@torch.no_grad()
def nitik_val_mrr(enc, val_recs) -> float:
    """Checkpoint criterion: mean MRR over the VAL_FAMILY strengths (clean
    validation is saturated at 1.0 and cannot rank epochs)."""
    r = robust_eval(lambda rc, f, l: siglip_feats(enc, rc, f, l), val_recs,
                    {VAL_FAMILY: LEVELS[VAL_FAMILY]})
    return float(np.mean([_mrr(r[f'{VAL_FAMILY}_{j}']) for j in range(len(LEVELS[VAL_FAMILY]))]))


def run_cv(method: str, fourier: bool, fold: int, seed: int):
    tag = f"{method}_f{int(fourier)}_fold{fold}_seed{seed}"
    out = RUNS_DIR / f'{tag}.npz'
    if out.exists():
        return 'skip'
    resume = RUNS_DIR / f'{tag}_resume.pt'
    t0 = time.time()
    set_seed(1000 * seed + fold)

    nit_tr, nit_va, nit_te = fold_records(fold)
    train_recs = [dict(r) for r in nit_tr + extra_train]
    enc = build_encoder(method)
    ds = BatikDatasetWithFourier(train_recs, transform=get_transform('siglip', augment=True),
                                 use_fourier=fourier)
    g = torch.Generator(); g.manual_seed(1000 * seed + fold)
    common = dict(num_workers=min(4, os.cpu_count() or 1), pin_memory=True,
                  worker_init_fn=_seed_worker, generator=g, persistent_workers=True)
    if BALANCED_SAMPLER:
        labels = [r['global_class_id'] for r in ds.base.records]
        loader = DataLoader(ds, batch_sampler=PKBatchSampler(labels, PK_P, PK_K,
                            STEPS_PER_EPOCH, 1000 * seed + fold), **common)
    else:
        sampler = torch.utils.data.RandomSampler(ds, replacement=True, generator=g,
                                                 num_samples=STEPS_PER_EPOCH * BATCH_SIZE)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True, **common)

    trainable = [p for p in enc.parameters() if p.requires_grad]
    opt    = torch.optim.AdamW(trainable, lr=LR_MAP_V3[method], weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CV_EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    crit   = SupConLoss(temperature=0.07)
    best_mrr, best_state, best_ep, start, bad_total = -1.0, None, -1, 0, 0

    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        load_trainable_state(enc, ck['state']); opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
        best_mrr, best_state, best_ep = ck['best_mrr'], ck['best_state'], ck['best_ep']
        start, bad_total = ck['epoch'] + 1, ck.get('bad', 0)
        if BALANCED_SAMPLER: loader.batch_sampler.epoch = start
        print(f"  resumed {tag} at epoch {start}")

    for ep in range(start, CV_EPOCHS):
        loss, bad = train_one_epoch_amp(enc, opt, loader, crit, scaler)
        bad_total += bad
        sched.step()
        v = nitik_val_mrr(enc, nit_va)
        if np.isfinite(v) and v > best_mrr:
            best_mrr, best_state, best_ep = v, trainable_state(enc), ep
        print(f"    {tag} ep {ep+1:2d}/{CV_EPOCHS} loss={loss:.4f} val_MRR={v:.4f} "
              f"best={best_mrr:.4f}@{best_ep+1}" + (f"  ⚠ {bad} non-finite steps" if bad else ''))
        if (ep + 1) % 5 == 0 and ep + 1 < CV_EPOCHS:
            torch.save({'state': trainable_state(enc), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                        'epoch': ep, 'best_mrr': best_mrr, 'best_state': best_state,
                        'best_ep': best_ep, 'bad': bad_total}, resume)

    if best_state is None:
        raise RuntimeError(f"{tag}: validation never produced a finite MRR — training diverged.")
    load_trainable_state(enc, best_state)
    res = robust_eval(lambda rc, f, l: siglip_feats(enc, rc, f, l), nit_te, LEVELS)
    rr_i = res['clean']
    minutes = (time.time() - t0) / 60
    tmp = RUNS_DIR / f'{tag}.tmp.npz'                     # write-then-rename: never a half file
    np.savez(tmp, paths=np.array([r['path'] for r in nit_te]),
             **{f'rr__{c}': v for c, v in res.items()},
             best_val_mrr=best_mrr, best_epoch=best_ep + 1, minutes=minutes,
             nonfinite_steps=bad_total, amp=USE_AMP, balanced=BALANCED_SAMPLER)
    tmp.rename(out)
    if method != 'full' or SAVE_FULL_FT_WEIGHTS:
        torch.save(best_state, RUNS_DIR / f'{tag}_weights.pt')
    if resume.exists(): resume.unlink()
    rob = np.mean([_mrr(v) for c, v in res.items() if c != 'clean'])
    print(f"  ✔ {tag}: clean MRR={_mrr(rr_i):.4f}  robust MRR={rob:.4f}  "
          f"(val {best_mrr:.4f} @ep{best_ep+1}, {minutes:.1f} min)"
          + (f"  ⚠ {bad_total} non-finite steps skipped" if bad_total else ''))
    del enc, opt, ds, loader; torch.cuda.empty_cache()
    return minutes


def run_reference_cv():
    """Frozen SigLIP and Gabor on the same 5 test folds and degradations."""
    for name in ['frozen', 'gabor']:
        for k in range(N_FOLDS):
            out = RUNS_DIR / f'{name}_fold{k}.npz'
            if out.exists(): continue
            nit_tr, nit_va, nit_te = fold_records(k)
            if name == 'frozen':
                enc = build_encoder('frozen')
                fn = lambda rc, f, l: siglip_feats(enc, rc, f, l)
            else:                                           # z-score with clean non-test classes
                ref = gabor_feats_raw(nit_tr + nit_va)
                mu, sd = ref.mean(0), ref.std(0) + 1e-8
                def fn(rc, f, l):
                    z = (gabor_feats_raw(rc, f, l) - mu) / sd
                    return z / np.linalg.norm(z, axis=1, keepdims=True)
            res = robust_eval(fn, nit_te, LEVELS)
            if name == 'frozen': del enc
            tmp = RUNS_DIR / f'{name}_fold{k}.tmp.npz'
            np.savez(tmp, paths=np.array([r['path'] for r in nit_te]),
                     **{f'rr__{c}': v for c, v in res.items()})
            tmp.rename(out)
        print(f"  ✔ {name}: {N_FOLDS} folds")

# ═══════════════════════════════════════════════════════════════════════════
#  Shared control/v5 DEFINITIONS (moved here from controls.py so that the
#  train/evaluate/export/generalize stages can resolve these names without
#  executing the compute-heavy RQ2 controls stage first). Only *definitions*
#  live here; the executable RQ2 run blocks remain in controls.py. The
#  research logic is unchanged — this is a pure relocation.
# ═══════════════════════════════════════════════════════════════════════════


# ==================== Original notebook cell 12 (definitions) ====================
# ═══════════════════════════════════════════════════════════════════════════
#  Gabor texture baseline (hand-crafted, training-free)
#  Evaluated on the SAME global_test split + metrics as the SigLIP baseline
#  (cells "MRR" / "Cross Dataset Generalization") so numbers are comparable.
# ═══════════════════════════════════════════════════════════════════════════

# ── Gabor filter bank ───────────────────────────────────────────────────────
# Wavelength λ (px, on the 224×224 image) sets the scale. σ is tied to λ
# (σ ≈ 0.56 λ ⇒ ~1-octave bandwidth) and the kernel grows with σ (±3σ), so
# large scales are no longer truncated by a fixed 31×31 window.
GABOR_WAVELENGTHS  = [4, 8, 16, 32]
GABOR_ORIENTATIONS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GABOR_GAMMA        = 0.5      # spatial aspect ratio
GABOR_SIGMA_RATIO  = 0.56     # σ / λ
GABOR_IMG_SIZE     = IMG_SIZE # 224
GABOR_PAD          = GABOR_IMG_SIZE // 2   # reflect pad ≥ largest kernel radius

def make_gabor_bank(size=GABOR_IMG_SIZE, pad=GABOR_PAD):
    """
    Complex Gabor kernels (even = real, odd = imag part: a quadrature pair),
    returned as their FFTs on the padded grid → (K, P, P) complex64 tensor.
      • even part made zero-mean (no DC response → invariant to brightness)
      • each kernel L1-normalised so every scale has comparable gain
    """
    P = size + 2 * pad
    bank, meta = [], []
    for lam in GABOR_WAVELENGTHS:
        sigma = GABOR_SIGMA_RATIO * lam
        half  = int(np.ceil(3 * sigma))
        assert half < pad, f"kernel radius {half} exceeds padding {pad}"
        y, x  = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
        for theta in GABOR_ORIENTATIONS:
            xr =  x * np.cos(theta) + y * np.sin(theta)
            yr = -x * np.sin(theta) + y * np.cos(theta)
            env = np.exp(-(xr**2 + (GABOR_GAMMA * yr)**2) / (2 * sigma**2))
            g   = env * np.exp(1j * 2 * np.pi * xr / lam)
            g   = (g.real - env * g.real.sum() / env.sum()) + 1j * g.imag  # zero DC
            g  /= np.abs(g).sum()
            k = np.zeros((P, P), dtype=np.complex128)
            k[:2*half + 1, :2*half + 1] = g
            k = np.roll(k, (-half, -half), axis=(0, 1))   # centre kernel at origin
            bank.append(np.fft.fft2(k))
            meta.append((lam, theta))
    return torch.from_numpy(np.stack(bank)).to(torch.complex64), meta

GABOR_BANK, GABOR_META = make_gabor_bank()
GABOR_BANK = GABOR_BANK.to(device)
print(f"Gabor bank: {len(GABOR_META)} filters  →  {2*len(GABOR_META)}-dim descriptor")


@torch.no_grad()
def gabor_descriptor(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 1, H, W) grayscale in [0, 1].
    Returns (B, 2K): mean and std of the Gabor *magnitude* response per filter
    (Manjunath & Ma 1996). Magnitude of the quadrature pair is phase-invariant,
    unlike the raw real response whose spatial mean is ~0 for every filter.
    """
    H, W = x.shape[-2:]
    p = GABOR_PAD
    xp  = F.pad(x, (p, p, p, p), mode='reflect')                 # avoid FFT wrap-around
    X   = torch.fft.fft2(xp)                                      # (B, 1, P, P)
    R   = torch.fft.ifft2(X * GABOR_BANK.unsqueeze(0))            # (B, K, P, P)
    mag = R.abs()[..., p:p + H, p:p + W]
    return torch.cat([mag.mean(dim=(-2, -1)), mag.std(dim=(-2, -1))], dim=1)


class GrayImageDataset(Dataset):
    """Read-only grayscale loader — does NOT mutate records (BatikDataset rewrites
    r['global_class_id'] every time it is built, so we don't rely on it)."""
    def __init__(self, records):
        self.paths = [r['path'] for r in records]
        self.tf = transforms.Compose([
            transforms.Resize((GABOR_IMG_SIZE, GABOR_IMG_SIZE)),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('L'))


def extract_gabor(records, batch_size=BATCH_SIZE, cache_name=None):
    cache = OUT_DIR / cache_name if cache_name else None
    paths = [r['path'] for r in records]
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z['paths']) == paths:
            print(f"  loaded cached Gabor features: {cache.name}")
            return z['feats']
    loader = DataLoader(GrayImageDataset(records), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    out = [gabor_descriptor(x.to(device)).cpu().numpy()
           for x in tqdm(loader, desc='Gabor', leave=False)]
    feats = np.vstack(out).astype(np.float32)
    if cache is not None:
        np.savez(cache, feats=feats, paths=np.array(paths, dtype=object))
    return feats


def class_labels(records):
    """Integer labels from the stable key 'dataset::class_name'."""
    keys = [f"{r['dataset']}::{r['class_name']}" for r in records]
    k2i  = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([k2i[k] for k in keys], dtype=np.int64)


def loo_metrics(feats, labels) -> dict:
    """Leave-one-out instance retrieval: each image queries all *other* test
    images. Complements the centroid protocol (whose centroids include the query)."""
    sim = feats @ feats.T
    np.fill_diagonal(sim, -np.inf)
    order   = np.argsort(-sim, axis=1)
    correct = labels[order] == labels[:, None]
    correct[:, -1] = False                         # self sits last (−inf)
    has_pos = correct.any(1)
    first   = correct.argmax(1) + 1
    rr      = np.where(has_pos, 1.0 / first, 0.0)
    return {'mrr': float(rr.mean()),
            'r@1': float(correct[:, 0].mean()),
            'r@5': float(correct[:, :5].any(1).mean()),
            'hnr': float(1 - correct[:, 0].mean())}   # top-1 wrong (same def. as compute_hnr)


# ==================== Original notebook cell 22 (definitions) ====================
# ═══════════════════════════════════════════════════════════════════════════
#  [v5-1] Multi-encoder setup · augmentation variants · weight blending
#  Reuses the v4 levels, folds, metrics and the 40 finished SigLIP runs.
# ═══════════════════════════════════════════════════════════════════════════
import math, platform, sys, csv

# ── What v5 runs (edit to add/remove; finished runs are always skipped) ──
#   (encoder, method, augmentation) — augmentation ∈ 'none' | 'fourier' | 'blur'
V5_PLAN = (
    [('siglip', m, 'blur') for m in ['lora', 'dora']]                      # item 2: augmentation control
  + [(e, 'lora', a) for e in ['dinov2', 'clip'] for a in ['none', 'fourier', 'blur']]  # other encoders, incl. blur
)
V5_REF_ENCODERS = ['dinov2', 'clip']          # frozen references at the same degradation levels
WISE_ALPHAS     = [0.25, 0.5, 0.75]           # 0 = frozen, 1 = fine-tuned (already measured)
LR_V5           = {'lora': 3e-4, 'dora': 3e-4, 'krona': 1e-3, 'full': 1e-5}

# Correct input normalisation per encoder (v1 used ImageNet stats for CLIP).
NORM = {'siglip': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        'clip':   ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        'dinov2': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

# ── Run naming. v4 SigLIP runs are mapped in; new runs use the v5 scheme ──
#   v5: <enc>__<method>__<aug>__fold<k>__seed<s>.npz   ·  ref: <enc>__frozen__fold<k>.npz
def v5_tag(enc, method, aug, fold, seed=0):
    if enc == 'siglip' and aug in ('none', 'fourier'):          # finished v4 run
        return f"{method}_f{int(aug == 'fourier')}_fold{fold}_seed{seed}"
    return f"{enc}__{method}__{aug}__fold{fold}__seed{seed}"

def ref_tag(enc, fold):
    return f"frozen_fold{fold}" if enc == 'siglip' else f"{enc}__frozen__fold{fold}"

# ── Encoders: loaded on first use ──
def base_encoder(enc):
    if enc not in encoders:
        print(f"  loading {enc} …")
        encoders[enc] = ViTEncoder(enc).eval().to(device)
    return encoders[enc]

def _qkv_targets(vision):
    names = []
    for n, m in vision.named_modules():
        last = n.split('.')[-1]
        if isinstance(m, nn.Linear) and last in ('q_proj', 'k_proj', 'v_proj', 'qkv'):
            names.append(n)
    assert names, "no attention projections found"
    return names

def apply_lora_any(encoder, r=16, alpha=32, use_dora=False):
    """LoRA/DoRA on every attention q/k/v projection (fused 'qkv' for DINOv2)."""
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=_qkv_targets(encoder.vision),
                     lora_dropout=0.05, bias='none', use_dora=use_dora)
    return get_peft_model(encoder.vision, cfg)

_TRAIN_PREFIX = {'siglip': '_model.vision_model', 'clip': '_model.vision_model', 'dinov2': 'vision.'}

def build_encoder_v5(method, enc='siglip'):
    e = deepcopy(base_encoder(enc)).to(device)
    if method == 'frozen':
        for p in e.parameters(): p.requires_grad_(False)
    elif method in ('lora', 'dora'):
        (apply_lora_v2 if enc == 'siglip' else apply_lora_any)(e, r=16, alpha=32, use_dora=method == 'dora')
    elif method == 'krona':
        apply_krona(e.vision, k=8)
    elif method == 'full':
        for p in e.parameters(): p.requires_grad_(True)
    else:
        raise ValueError(method)
    for n, p in e.named_parameters():                 # only the vision tower may train
        if not n.startswith(_TRAIN_PREFIX[enc]):
            p.requires_grad_(False)
    assert method == 'frozen' or any(p.requires_grad for p in e.parameters()), "nothing to train"
    return e

# ── Transforms / degraded inputs with the right normalisation ──
def train_transform(enc, aug):
    t = [transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
         transforms.RandomHorizontalFlip(),
         transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)]
    if aug == 'blur':                                 # item 2: matched blur control
        t.append(transforms.RandomApply([transforms.GaussianBlur(23, sigma=(0.1, 4.0))], p=0.5))
    t += [transforms.ToTensor(), transforms.Normalize(*NORM[enc])]
    return transforms.Compose(t)

class DegradedDatasetE(Dataset):
    def __init__(self, recs, enc, fam='clean', lvl=None):
        self.paths, self.fam, self.lvl = [r['path'] for r in recs], fam, lvl
        self.norm = transforms.Normalize(*NORM[enc])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.norm(degrade_pil(Image.open(self.paths[i]), self.fam, self.lvl))

@torch.no_grad()
def enc_feats(model, enc, recs, fam='clean', lvl=None):
    model.eval()
    dl = DataLoader(DegradedDatasetE(recs, enc, fam, lvl), batch_size=64, shuffle=False,
                    num_workers=min(4, os.cpu_count() or 1))
    return np.vstack([model(x.to(device)).float().cpu().numpy() for x in dl])

def val_score(model, enc, val_recs):
    r = robust_eval(lambda rc, f, l: enc_feats(model, enc, rc, f, l), val_recs,
                    {VAL_FAMILY: LEVELS[VAL_FAMILY]})
    return float(np.mean([_mrr(r[f'{VAL_FAMILY}_{j}']) for j in range(len(LEVELS[VAL_FAMILY]))]))

def save_npz(path, **arrays):
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays); tmp.rename(path)

# ── Weight blending (WiSE-FT): θ(α) = (1−α)·θ_frozen + α·θ_fine-tuned ──
def blend_(model, alpha):
    """In place. LoRA/DoRA layers are merged, then their weight is interpolated
    with the frozen weight; KronA's update is linear in A1, so A1 is scaled."""
    from peft.tuners.lora import LoraLayer
    n = 0
    for m in model.modules():
        if isinstance(m, LoraLayer) and not getattr(m, 'merged', False):
            w0 = m.get_base_layer().weight.data.clone()
            m.merge()
            w1 = m.get_base_layer().weight.data
            m.get_base_layer().weight.data = (1 - alpha) * w0 + alpha * w1
            n += 1
    for m in model.modules():
        if isinstance(m, KroneckerAdapter):
            m.A1.data.mul_(alpha); n += 1
    assert n, "no adapter layers found to blend"
    return model

def time_left_ok(est_minutes):
    return (time.time() - SESSION_START) / 3600 + est_minutes / 60 <= SESSION_BUDGET_H

print("v5 plan:", len(V5_PLAN), "configs ×", N_FOLDS, "folds →",
      sum(not (RUNS_DIR / f"{v5_tag(e, m, a, k)}.npz").exists()
          for e, m, a in V5_PLAN for k in range(N_FOLDS)), "training runs to do")


# ==================== Original notebook cell 23 (definitions) ====================
# Frozen validation score per (encoder, fold) — the α = 0 point for blending.
# These are definitions only; the RQ2 execution that calls them lives in controls.py.
FROZEN_VAL_FILE = EVAL_DIR / 'frozen_val_scores.json'
FROZEN_VAL = json.load(open(FROZEN_VAL_FILE)) if FROZEN_VAL_FILE.exists() else {}
def frozen_val(enc, k):
    key = f'{enc}|{k}'
    if key not in FROZEN_VAL:
        m = build_encoder_v5('frozen', enc)
        FROZEN_VAL[key] = val_score(m, enc, fold_records(k)[1]); del m
        json.dump(FROZEN_VAL, open(FROZEN_VAL_FILE, 'w'), indent=1)
    return FROZEN_VAL[key]

# ── B. Weight blending on every saved adapter (no training) ──────────────
def wise_jobs():
    jobs = []
    for enc in ['siglip', 'dinov2', 'clip']:
        for method in ['lora', 'dora', 'krona']:
            for aug in ['none', 'fourier', 'blur']:
                for k in range(N_FOLDS):
                    tag = v5_tag(enc, method, aug, k)
                    w = RUNS_DIR / f'{tag}_weights.pt'
                    if w.exists() and not (RUNS_DIR / f'wise__{tag}.npz').exists():
                        jobs.append((enc, method, aug, k, tag, w))
    return jobs

def run_wise(enc, method, aug, k, tag, w):
    t0 = time.time()
    _, va, te = fold_records(k)
    run = np.load(RUNS_DIR / f'{tag}.npz')
    out = {'paths': np.array([r['path'] for r in te]),
           'alphas': np.array([0.0] + WISE_ALPHAS + [1.0]),
           'val': np.array([frozen_val(enc, k)] + [np.nan] * len(WISE_ALPHAS) + [float(run['best_val_mrr'])])}
    state = torch.load(w, map_location='cpu')
    for i, a in enumerate(WISE_ALPHAS, start=1):
        m = build_encoder_v5(method, enc); load_trainable_state(m, state); blend_(m, a)
        out['val'][i] = val_score(m, enc, va)
        res = robust_eval(lambda rc, f, l: enc_feats(m, enc, rc, f, l), te, LEVELS)
        for c, v in res.items(): out[f'rr__a{a}__{c}'] = v
        del m; torch.cuda.empty_cache()
    out['minutes'] = (time.time() - t0) / 60
    save_npz(RUNS_DIR / f'wise__{tag}.npz', **out)
    print(f"  ✔ blend {tag}: val by α {np.round(out['val'], 3).tolist()}  ({out['minutes']:.1f} min)")
    return out['minutes']

# ── C. Training (generalised v4 run_cv) ───────────────────────────────────
def run_cv_v5(enc, method, aug, fold, seed=0):
    tag = v5_tag(enc, method, aug, fold, seed)
    out = RUNS_DIR / f'{tag}.npz'
    if out.exists(): return None
    resume = RUNS_DIR / f'{tag}_resume.pt'
    t0 = time.time(); set_seed(1000 * seed + fold)
    nit_tr, nit_va, nit_te = fold_records(fold)
    model = build_encoder_v5(method, enc)
    ds = BatikDatasetWithFourier([dict(r) for r in nit_tr + extra_train],
                                 transform=train_transform(enc, aug), use_fourier=(aug == 'fourier'))
    labels = [r['global_class_id'] for r in ds.base.records]
    g = torch.Generator(); g.manual_seed(1000 * seed + fold)
    loader = DataLoader(ds, batch_sampler=PKBatchSampler(labels, PK_P, PK_K, STEPS_PER_EPOCH, 1000 * seed + fold),
                        num_workers=min(4, os.cpu_count() or 1), pin_memory=True,
                        worker_init_fn=_seed_worker, generator=g, persistent_workers=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.AdamW(params, lr=LR_V5[method], weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CV_EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    crit   = SupConLoss(temperature=0.07)
    best, best_state, best_ep, start, bad_total, log = -1.0, None, -1, 0, 0, []
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        load_trainable_state(model, ck['state']); opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
        best, best_state, best_ep, start = ck['best'], ck['best_state'], ck['best_ep'], ck['epoch'] + 1
        bad_total, log = ck['bad'], ck['log']; loader.batch_sampler.epoch = start
        print(f"  resumed {tag} at epoch {start}")
    for ep in range(start, CV_EPOCHS):
        loss, bad = train_one_epoch_amp(model, opt, loader, crit, scaler)
        bad_total += bad; sched.step()
        v = val_score(model, enc, nit_va)
        log.append((ep + 1, loss, v))
        if np.isfinite(v) and v > best:
            best, best_state, best_ep = v, trainable_state(model), ep
        print(f"    {tag} ep {ep+1:2d}/{CV_EPOCHS} loss={loss:.4f} val={v:.4f} best={best:.4f}@{best_ep+1}"
              + (f"  ⚠ {bad} non-finite" if bad else ''))
        if (ep + 1) % 5 == 0 and ep + 1 < CV_EPOCHS:
            torch.save({'state': trainable_state(model), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'scaler': scaler.state_dict(), 'epoch': ep, 'best': best, 'best_state': best_state,
                        'best_ep': best_ep, 'bad': bad_total, 'log': log}, resume)
    if best_state is None:
        raise RuntimeError(f"{tag}: validation never finite — training diverged")
    load_trainable_state(model, best_state)
    res = robust_eval(lambda rc, f, l: enc_feats(model, enc, rc, f, l), nit_te, LEVELS)
    minutes = (time.time() - t0) / 60
    save_npz(out, paths=np.array([r['path'] for r in nit_te]), **{f'rr__{c}': v for c, v in res.items()},
             best_val_mrr=best, best_epoch=best_ep + 1, minutes=minutes, nonfinite_steps=bad_total,
             val_log=np.array(log, dtype=float), amp=USE_AMP)
    if method != 'full' or SAVE_FULL_FT_WEIGHTS:
        torch.save(best_state, RUNS_DIR / f'{tag}_weights.pt')
    if resume.exists(): resume.unlink()
    rob = np.mean([_mrr(v) for c, v in res.items() if c != 'clean'])
    print(f"  ✔ {tag}: clean={_mrr(res['clean']):.4f} robust={rob:.4f} (val {best:.4f} @ep{best_ep+1}, {minutes:.1f} min)")
    del model, opt, ds, loader; torch.cuda.empty_cache()
    return minutes

# ── Go: blending of existing adapters first (cheap, results early), then training,
#        then blending of the adapters that training just produced ────────
def _go():
    hist_w = [float(np.load(p)['minutes']) for p in RUNS_DIR.glob('wise__*.npz')]
    hist_t = [float(np.load(p)['minutes']) for p in RUNS_DIR.glob('*.npz')
              if not p.stem.startswith(('wise__', 'frozen', 'gabor')) and '__frozen__' not in p.stem
              and 'minutes' in np.load(p).files]
    for job in wise_jobs():
        if not time_left_ok(1.5 * max(hist_w or [8])): return False
        hist_w.append(run_wise(*job))
    queue = [(e, m, a, k) for k in range(N_FOLDS) for e, m, a in V5_PLAN
             if not (RUNS_DIR / f"{v5_tag(e, m, a, k)}.npz").exists()]
    print(f"\n{len(queue)} v5 training runs left.")
    for e, m, a, k in queue:
        if not time_left_ok(1.3 * max(hist_t or [45])): return False
        hist_t.append(run_cv_v5(e, m, a, k))
        tag = v5_tag(e, m, a, k)                       # blend it right away while it's fresh
        for job in wise_jobs():
            if job[4] == tag and time_left_ok(1.5 * max(hist_w or [8])):
                hist_w.append(run_wise(*job))
    for job in wise_jobs():
        if not time_left_ok(1.5 * max(hist_w or [8])): return False
        hist_w.append(run_wise(*job))
    return True
