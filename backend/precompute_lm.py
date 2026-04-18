"""
Pre-compute LM concept results for TCE datasets.
Run once with: OPENAI_API_KEY=sk-... python precompute_lm.py

Generates backend/lm_results/results_{dataset}_{concept}.json
"""

import os
import sys

# Block tensorflow before umap loads
sys.modules.setdefault("tensorflow", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import json
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "lm_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONCEPT_CACHE_FILE = os.path.join(BASE_DIR, "concept_directions.json")
UMAP_CACHE_FILE = os.path.join(BASE_DIR, "aligned_umap_embeddings.json")

# Default: compute ML concept for vispub
JOBS = [
    {"dataset": "vispub",   "concept": "machine_learning",       "factors": [-1, 2, 5]},
    {"dataset": "physics",  "concept": "machine_learning",       "factors": [-1, 2, 5]},
    {"dataset": "physics",  "concept": "experimental",            "factors": [-1, 2, 5]},
    {"dataset": "physics",  "concept": "many_body_physics",       "factors": [-1, 2, 5]},
    {"dataset": "physics",  "concept": "noise_and_disorder",      "factors": [-1, 2, 5]},
    {"dataset": "physics",  "concept": "nonequilibrium_dynamics", "factors": [-1, 2, 5]},
    {"dataset": "physics",  "concept": "pedagogical",             "factors": [-1, 2, 5, 10]},
]


# ── Concept direction cache ────────────────────────────────────────────────

def _load_concept_cache():
    if os.path.exists(CONCEPT_CACHE_FILE):
        with open(CONCEPT_CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_concept_cache(cache):
    with open(CONCEPT_CACHE_FILE, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── AlignedUMAP embeddings cache ──────────────────────────────────────────

def _direction_hash(direction_vec):
    """Short hash of the concept direction vector for cache invalidation."""
    data = ",".join(f"{v:.5f}" for v in direction_vec)
    return hashlib.sha256(data.encode()).hexdigest()[:10]


def _umap_cache_key(dataset_name, concept_name, factors, direction_vec):
    factors_str = ",".join(str(f) for f in sorted(factors))
    dir_hash = _direction_hash(direction_vec)
    return f"{dataset_name}__{concept_name}__{factors_str}__{dir_hash}"


def _load_umap_cache():
    if os.path.exists(UMAP_CACHE_FILE):
        with open(UMAP_CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_umap_cache(cache):
    with open(UMAP_CACHE_FILE, "w") as f:
        json.dump(cache, f, ensure_ascii=False)


# ── Helpers ────────────────────────────────────────────────────────────────

def load_datasets():
    path = os.path.join(BASE_DIR, "datasets.json")
    print(f"Loading datasets.json …")
    with open(path) as f:
        return json.load(f)


def generate_concept_direction(client, concept_name, dataset_name):
    cache_key = f"{dataset_name}__{concept_name}"
    cache = _load_concept_cache()
    if cache_key in cache:
        print(f"  Using cached concept direction for '{concept_name}' (dataset: {dataset_name})")
        entry = cache[cache_key]
        vec = entry["direction"] if isinstance(entry, dict) else entry
        return np.array(vec)

    print(f"  Generating concept direction for '{concept_name}' …")
    from sentence_transformers import SentenceTransformer

    DATASET_CTX = {
        "physics": (
            "The texts are physics research paper abstracts from arXiv. "
            "Generate texts that read like physics research abstracts — technical, concise, and domain-specific. "
        ),
        "frankenstein": (
            "The texts are passages from Mary Shelley's novel Frankenstein. "
            "Generate texts that match the Gothic, Romantic, 19th-century literary style of the novel. "
        ),
    }
    ctx = DATASET_CTX.get(dataset_name, "")
    prompt = (
        f"{ctx}Generate 20 short texts (50-100 words each) about '{concept_name}' "
        f"in the context of: {dataset_name}. "
        f"Separate each text with the exact string <SEP> on its own line."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
    )
    raw = resp.choices[0].message.content.strip()
    for sep in ["<SEP>", "---", "\n\n"]:
        parts = [t.strip() for t in raw.split(sep) if t.strip() and len(t.strip()) > 20]
        if len(parts) >= 5:
            concept_texts = parts
            break
    else:
        concept_texts = [t.strip() for t in raw.split("\n\n") if t.strip() and len(t.strip()) > 20]
    print(f"    Got {len(concept_texts)} concept texts")
    if not concept_texts:
        print(f"    Raw response: {raw[:300]}")
        raise ValueError("No concept texts generated")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(concept_texts, show_progress_bar=False)
    direction = embs.mean(axis=0)

    cache[cache_key] = {"texts": concept_texts, "direction": direction.tolist()}
    _save_concept_cache(cache)
    print(f"    Saved to concept_directions.json")
    return direction


def compute_aligned_umap(dataset_name, concept_name, factors, embeddings, concept_dir):
    import umap as umap_lib

    n = embeddings.shape[0]
    concept_dir_norm = concept_dir / np.linalg.norm(concept_dir)
    projections = embeddings @ concept_dir_norm

    # Build per-factor embedding matrices, sorted by factor value
    all_factor_embs = [(0, embeddings)]
    for factor in factors:
        if factor == -1:
            emb_proj = embeddings - np.outer(projections, concept_dir_norm)
        else:
            emb_proj = embeddings + factor * np.outer(projections, concept_dir_norm)
        all_factor_embs.append((factor, emb_proj))
    all_factor_embs.sort(key=lambda x: x[0])
    all_factors_sorted = [f for f, _ in all_factor_embs]
    all_embs_sorted = [e for _, e in all_factor_embs]

    cache_key = _umap_cache_key(dataset_name, concept_name, factors, concept_dir)
    umap_cache = _load_umap_cache()

    if cache_key in umap_cache:
        print(f"  Using cached AlignedUMAP embeddings …")
        cached = umap_cache[cache_key]
        factor_to_2d = {int(k) if k != "0" else 0: np.array(v) for k, v in cached.items()}
        # Ensure all required factors are present
        if all(f in factor_to_2d for f in all_factors_sorted):
            return factor_to_2d, concept_dir_norm, projections
        print(f"  Cache incomplete, recomputing …")

    print(f"  AlignedUMAP on {len(all_embs_sorted)} snapshots ({n} points each) …")
    relations = [{i: i for i in range(n)} for _ in range(len(all_embs_sorted) - 1)]
    aligned_mapper = umap_lib.AlignedUMAP(n_neighbors=15, min_dist=0.1, alignment_regularisation=1e-3, random_state=42)
    aligned_mapper.fit(all_embs_sorted, relations=relations)
    all_2d = aligned_mapper.embeddings_

    factor_to_2d = {f: all_2d[j] for j, f in enumerate(all_factors_sorted)}

    umap_cache[cache_key] = {str(f): emb.tolist() for f, emb in factor_to_2d.items()}
    _save_umap_cache(umap_cache)
    print(f"  Saved AlignedUMAP embeddings to aligned_umap_embeddings.json")

    return factor_to_2d, concept_dir_norm, projections


def generate_topic_labels(client, texts, embeddings, cluster_labels, n_examples=5):
    cluster_topics = {}
    unique = sorted(set(cluster_labels))
    print(f"  Generating labels for {len([c for c in unique if c != -1])} clusters …")
    for cid in unique:
        if cid == -1:
            continue
        mask = cluster_labels == cid
        c_embs = embeddings[mask]
        centroid = c_embs.mean(axis=0).reshape(1, -1)
        indices = np.where(mask)[0]
        dists = np.linalg.norm(c_embs - centroid, axis=1)
        nearest = indices[np.argsort(dists)[:n_examples]]
        examples = [texts[i] for i in nearest]
        example_block = "\n".join(f"{j+1}. {ex}" for j, ex in enumerate(examples))
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": (
                    f"These texts are representative examples from a cluster:\n\n{example_block}\n\n"
                    "Write a 2-5 word label that broadly describes what most texts in this cluster are about. "
                    "Use plain, everyday language — avoid jargon. Reply with ONLY the label."
                )}],
                temperature=0.3,
                max_tokens=20,
            )
            cluster_topics[cid] = resp.choices[0].message.content.strip()
        except Exception as e:
            cluster_topics[cid] = f"Cluster {cid}"
    return cluster_topics


def cluster_purity(embeddings, cluster_labels):
    from sklearn.metrics.pairwise import cosine_similarity as csim
    unique = [c for c in set(cluster_labels) if c != -1]
    if not unique:
        return {"overall_purity": 0.0, "cluster_purities": {}}
    purities, sizes = {}, {}
    for cid in unique:
        mask = cluster_labels == cid
        c_embs = embeddings[mask]
        sizes[cid] = c_embs.shape[0]
        if sizes[cid] == 1:
            purities[cid] = 1.0
        else:
            centroid = c_embs.mean(axis=0).reshape(1, -1)
            purities[cid] = float(csim(c_embs, centroid).flatten().mean())
    total = sum(sizes.values())
    overall = sum(purities[c] * sizes[c] for c in unique) / total
    return {"overall_purity": float(overall),
            "cluster_purities": {int(k): float(v) for k, v in purities.items()}}


# ── Main job ───────────────────────────────────────────────────────────────

def run_job(client, datasets, dataset_name, concept_name, factors):
    import hdbscan
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    output_file = os.path.join(RESULTS_DIR, f"results_{dataset_name}_{concept_name}.json")
    if os.path.exists(output_file):
        print(f"  {output_file} already exists, skipping.")
        return

    print(f"\n=== {dataset_name} / {concept_name} ===")
    ds = datasets[dataset_name]
    texts = ds["texts"]
    embeddings = np.array(ds["embeddings"])
    n = len(texts)
    print(f"  {n} texts, embeddings shape: {embeddings.shape}")

    concept_dir = generate_concept_direction(client, concept_name.replace("_", " "), dataset_name)
    cosine_sims = cosine_similarity(embeddings, concept_dir.reshape(1, -1)).flatten()

    factor_to_2d, concept_dir_norm, projections = compute_aligned_umap(
        dataset_name, concept_name, factors, embeddings, concept_dir
    )

    emb_2d = factor_to_2d[0]

    print("  HDBSCAN on original …")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
    orig_labels = clusterer.fit_predict(emb_2d)
    orig_n_cl = int(len(set(orig_labels)) - (1 if -1 in orig_labels else 0))
    orig_n_no = int(list(orig_labels).count(-1))
    print(f"    {orig_n_cl} clusters, {orig_n_no} noise")

    orig_topics = generate_topic_labels(client, texts, embeddings, orig_labels)

    orig_sil = None
    nn = orig_labels != -1
    if np.sum(nn) > 1 and orig_n_cl > 1:
        try:
            orig_sil = float(silhouette_score(embeddings[nn], orig_labels[nn], metric='cosine'))
        except Exception:
            pass

    results = {
        "dataset": dataset_name,
        "concept_direction": concept_name.replace("_", " ").title(),
        "concept_direction_vector": concept_dir.tolist(),
        "projection_factors": factors,
        "cosine_similarities": cosine_sims.tolist(),
        "original": {
            "embeddings_2d": emb_2d.tolist(),
            "cluster_labels": orig_labels.tolist(),
            "cluster_topics": {int(k): str(v) for k, v in orig_topics.items()},
            "n_clusters": orig_n_cl,
            "n_noise": orig_n_no,
            "silhouette_score": orig_sil,
            "purity_metrics": cluster_purity(embeddings, orig_labels),
        },
        "projected": {},
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "concept_direction": concept_name,
            "total_data_points": len(texts),
            "embedding_dimension": int(embeddings.shape[1]),
        },
    }

    for factor in factors:
        label = "removed" if factor == -1 else f"{factor}X_amplified"
        print(f"  Factor {factor} ({label}) …")

        emb_2d_proj = factor_to_2d[factor]

        if factor == -1:
            emb_proj = embeddings - np.outer(projections, concept_dir_norm)
        else:
            emb_proj = embeddings + factor * np.outer(projections, concept_dir_norm)

        c2 = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=3, metric='euclidean')
        cl = c2.fit_predict(emb_2d_proj)
        n_cl = int(len(set(cl)) - (1 if -1 in cl else 0))
        n_no = int(list(cl).count(-1))
        print(f"    {n_cl} clusters, {n_no} noise")

        topics = generate_topic_labels(client, texts, embeddings, cl)

        sil = None
        nm = cl != -1
        if np.sum(nm) > 1 and n_cl > 1:
            try:
                sil = float(silhouette_score(emb_proj[nm], cl[nm], metric='cosine'))
            except Exception:
                pass

        results["projected"][label] = {
            "factor": float(factor),
            "factor_label": label,
            "embeddings_2d": emb_2d_proj.tolist(),
            "cluster_labels": cl.tolist(),
            "cluster_topics": {int(k): str(v) for k, v in topics.items()},
            "n_clusters": n_cl,
            "n_noise": n_no,
            "silhouette_score": sil,
            "purity_metrics": cluster_purity(emb_proj, cl),
        }

    print(f"  Saving to {output_file} …")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"  Done.")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    all_datasets = load_datasets()

    for job in JOBS:
        run_job(client, all_datasets, job["dataset"], job["concept"], job["factors"])

    print("\nAll done. Results saved to lm_results/")


if __name__ == "__main__":
    main()
