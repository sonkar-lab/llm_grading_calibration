import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable, List


def compute_features_cache_key(
    *,
    predictions_path: Path,
    smoke_n: Optional[int],
    enable_perplexity: bool,
    perplexity_model: str,
    perplexity_max_length: int,
    perplexity_min_tokens: int,
    feature_version: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "predictions_file": predictions_path.name,
        "smoke_n": smoke_n,
        "enable_perplexity": bool(enable_perplexity),
        "perplexity_model": perplexity_model,
        "perplexity_max_length": int(perplexity_max_length),
        "perplexity_min_tokens": int(perplexity_min_tokens),
        "feature_version": feature_version,
    }
    if extra:
        payload.update(extra)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return digest, payload


def build_features_cache_path(
    cache_dir: Optional[str],
    *,
    dataset: str,
    method: str,
    model: str,
    feature_version: str,
    cache_key: str,
) -> Optional[Path]:
    if not cache_dir:
        return None
    safe_model = model.replace("/", "_")
    filename = f"features_{safe_model}_{method}_{feature_version}_{cache_key}.csv"
    return Path(cache_dir) / dataset / method / filename


def cache_metadata_path(cache_path: Path) -> Path:
    return Path(str(cache_path) + ".meta.json")


def write_features_cache_metadata(cache_path: Path, metadata: Dict[str, Any]) -> None:
    meta_path = cache_metadata_path(cache_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_features_cache_metadata(meta_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_compatible_feature_caches(
    cache_dir: Optional[str],
    *,
    dataset: str,
    method: str,
    model: str,
    feature_version: str,
    target_meta: Dict[str, Any],
    ignore_keys: Optional[Iterable[str]] = None,
) -> List[Path]:
    if not cache_dir:
        return []
    ignore = set(ignore_keys or [])
    base = Path(cache_dir) / dataset / method
    if not base.exists():
        return []
    safe_model = model.replace("/", "_")
    pattern = f"features_{safe_model}_{method}_{feature_version}_*.csv.meta.json"
    matches: List[Path] = []
    for meta_path in base.glob(pattern):
        meta = load_features_cache_metadata(meta_path)
        if not meta:
            continue
        ok = True
        for key, val in target_meta.items():
            if key in ignore:
                continue
            if meta.get(key) != val:
                ok = False
                break
        if not ok:
            continue
        csv_path = Path(str(meta_path)[:-len(".meta.json")])
        if csv_path.exists():
            matches.append(csv_path)
    return matches
