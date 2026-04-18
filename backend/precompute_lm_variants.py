"""
Generate 3 comparison variants of physics/machine_learning embeddings.

  1. machine_learning_independent  — separate UMAP per factor (no alignment)
  2. machine_learning              — existing file (aligned UMAP, default reg) — skipped if present
  3. machine_learning_loose_aligned — AlignedUMAP with alignment_regularisation=1e-3

Run: OPENAI_API_KEY=sk-... python precompute_lm_variants.py
"""

import os, sys, json, numpy as np
from datetime import datetime

sys.modules.setdefault("tensorflow", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "lm_results")
CONCEPT_CACHE_FILE = os.path.join(BASE_DIR, "concept_directions.json")

DATASET = "physics"
CONCEPT_KEY = "machine_learning"
CONCEPT_DISPLAY = "Machine Learning"
FACTORS = [-1, 2, 3, 5]
FACTORS_LOOSE = [-1, 2, 3, 5]


# ── Helpers ────────────────────────────────────────────────────────────────

def load_concept_direction():
    with open(CONCEPT_CACHE_FILE) as f:
        cache = json.load(f)
    key = f"{DATASET}__{CONCEPT_KEY.replace('_', ' ')}"
    entry = cache[key]
    vec = entry["direction"] if isinstance(entry, dict) else entry
    return np.array(vec)


def compute_projected_embeddings(embeddings, factors, concept_dir):
    concept_dir_norm = concept_dir / np.linalg.norm(concept_dir)
    projections = embeddings @ concept_dir_norm

    factor_to_emb = {0: embeddings}
    for f in factors:
        if f == -1:
            factor_to_emb[f] = embeddings - np.outer(projections, concept_dir_norm)
        else:
            factor_to_emb[f] = embeddings + f * np.outer(projections, concept_dir_norm)

    return factor_to_emb, concept_dir_norm, projections


def umap_independent(factor_to_emb, factors):
    """Fit a separate UMAP for each snapshot independently."""
    import umap as umap_lib

    all_factors_sorted = sorted([0] + factors)
    factor_to_2d = {}
    for f in all_factors_sorted:
        print(f"    Independent UMAP for factor {f} …")
        mapper = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        factor_to_2d[f] = mapper.fit_transform(factor_to_emb[f])
    return factor_to_2d


def umap_aligned(factor_to_emb, factors, alignment_regularisation=1e-2):
    """Fit AlignedUMAP across all snapshots."""
    import umap as umap_lib

    all_factors_sorted = sorted([0] + factors)
    all_embs_sorted = [factor_to_emb[f] for f in all_factors_sorted]
    n = all_embs_sorted[0].shape[0]

    relations = [{i: i for i in range(n)} for _ in range(len(all_embs_sorted) - 1)]
    print(f"    AlignedUMAP on {len(all_embs_sorted)} snapshots, reg={alignment_regularisation} …")
    mapper = umap_lib.AlignedUMAP(
        n_neighbors=15, min_dist=0.1,
        alignment_regularisation=alignment_regularisation,
        random_state=42,
    )
    mapper.fit(all_embs_sorted, relations=relations)

    return {f: mapper.embeddings_[j] for j, f in enumerate(all_factors_sorted)}


def normalise_coords(factor_to_2d, factors):
    """Globally normalise all snapshots to [0,1] so they share a coordinate space."""
    all_factors_sorted = sorted([0] + factors)
    all_coords = np.vstack([factor_to_2d[f] for f in all_factors_sorted])
    mnx, mxx = all_coords[:, 0].min(), all_coords[:, 0].max()
    mny, mxy = all_coords[:, 1].min(), all_coords[:, 1].max()
    rx = mxx - mnx if mxx != mnx else 1.0
    ry = mxy - mny if mxy != mny else 1.0
    return {f: [[float((x - mnx) / rx), float((y - mny) / ry)] for x, y in factor_to_2d[f]]
            for f in all_factors_sorted}


def run_hdbscan_and_label(client, texts, embeddings, emb_2d_list, factors):
    """Run HDBSCAN + GPT labels for original and each factor snapshot."""
    import hdbscan
    from sklearn.metrics import silhouette_score

    emb_2d_orig = np.array(emb_2d_list[0])
    print("  HDBSCAN on original …")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
    orig_labels = clusterer.fit_predict(emb_2d_orig)
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

    projected_results = {}
    for i, factor in enumerate(factors):
        label = "removed" if factor == -1 else f"{factor}X_amplified"
        print(f"  Factor {factor} ({label}) …")
        emb_2d_f = np.array(emb_2d_list[i + 1])  # +1 because index 0 is original
        c2 = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=3, metric='euclidean')
        cl = c2.fit_predict(emb_2d_f)
        n_cl = int(len(set(cl)) - (1 if -1 in cl else 0))
        n_no = int(list(cl).count(-1))
        print(f"    {n_cl} clusters, {n_no} noise")
        topics = generate_topic_labels(client, texts, embeddings, cl)
        projected_results[label] = {
            "factor": float(factor),
            "factor_label": label,
            "embeddings_2d": emb_2d_f.tolist(),
            "cluster_labels": cl.tolist(),
            "cluster_topics": {int(k): str(v) for k, v in topics.items()},
            "n_clusters": n_cl,
            "n_noise": n_no,
            "silhouette_score": None,
            "purity_metrics": {},
        }

    return orig_labels, orig_topics, orig_n_cl, orig_n_no, orig_sil, projected_results


def generate_topic_labels(client, texts, embeddings, cluster_labels, n_examples=5):
    unique = sorted(set(cluster_labels))
    print(f"  Generating labels for {len([c for c in unique if c != -1])} clusters …")
    cluster_topics = {}
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
        except Exception:
            cluster_topics[cid] = f"Cluster {cid}"
    return cluster_topics


def build_and_save(client, texts, embeddings, concept_dir, cosine_sims,
                   factors, factor_to_2d_norm, display_name, output_filename):
    """Run HDBSCAN + labels on pre-computed normalised 2D coords and save JSON."""
    all_factors_sorted = sorted([0] + factors)

    # Build ordered list: original first, then factors in FACTORS order
    emb_2d_orig = factor_to_2d_norm[0]
    emb_2d_by_factor = [factor_to_2d_norm[f] for f in factors]
    all_emb_2d_ordered = [emb_2d_orig] + emb_2d_by_factor

    orig_labels, orig_topics, orig_n_cl, orig_n_no, orig_sil, projected = run_hdbscan_and_label(
        client, texts, embeddings, all_emb_2d_ordered, factors
    )

    concept_dir_norm = concept_dir / np.linalg.norm(concept_dir)
    results = {
        "dataset": DATASET,
        "concept_direction": display_name,
        "concept_direction_vector": concept_dir.tolist(),
        "projection_factors": factors,
        "cosine_similarities": cosine_sims.tolist(),
        "original": {
            "embeddings_2d": emb_2d_orig if isinstance(emb_2d_orig, list) else emb_2d_orig.tolist(),
            "cluster_labels": orig_labels.tolist(),
            "cluster_topics": {int(k): str(v) for k, v in orig_topics.items()},
            "n_clusters": orig_n_cl,
            "n_noise": orig_n_no,
            "silhouette_score": orig_sil,
            "purity_metrics": {},
        },
        "projected": projected,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": DATASET,
            "concept_direction": output_filename,
            "total_data_points": len(texts),
            "embedding_dimension": int(embeddings.shape[1]),
        },
    }
    out = os.path.join(RESULTS_DIR, f"results_{DATASET}_{output_filename}.json")
    print(f"  Saving to {out} …")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print("  Done.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    from openai import OpenAI
    from sklearn.metrics.pairwise import cosine_similarity

    client = OpenAI(api_key=api_key)

    print("Loading datasets.json …")
    with open(os.path.join(BASE_DIR, "datasets.json")) as f:
        all_datasets = json.load(f)

    ds = all_datasets[DATASET]
    texts = ds["texts"]
    embeddings = np.array(ds["embeddings"])
    print(f"  {len(texts)} texts, shape: {embeddings.shape}")

    print("Loading concept direction from cache …")
    concept_dir = load_concept_direction()
    cosine_sims = cosine_similarity(embeddings, concept_dir.reshape(1, -1)).flatten()

    factor_to_emb, concept_dir_norm, projections = compute_projected_embeddings(
        embeddings, FACTORS, concept_dir
    )

    # ── Variant 1: Independent UMAP ────────────────────────────────────────
    out1 = os.path.join(RESULTS_DIR, f"results_{DATASET}_machine_learning_independent.json")
    if os.path.exists(out1):
        print(f"\nVariant 1 already exists, skipping: {out1}")
    else:
        print("\n=== Variant 1: Independent UMAP ===")
        factor_to_2d_raw = umap_independent(factor_to_emb, FACTORS)
        factor_to_2d_norm = normalise_coords(factor_to_2d_raw, FACTORS)
        build_and_save(
            client, texts, embeddings, concept_dir, cosine_sims,
            FACTORS, factor_to_2d_norm,
            display_name="Machine Learning (Independent)",
            output_filename="machine_learning_independent",
        )

    # ── Variant 3: AlignedUMAP loose (1e-3) ───────────────────────────────
    out3 = os.path.join(RESULTS_DIR, f"results_{DATASET}_machine_learning_loose_aligned.json")
    if os.path.exists(out3):
        print(f"\nVariant 3 already exists, skipping: {out3}")
    else:
        print("\n=== Variant 3: AlignedUMAP (alignment_regularisation=1e-3, factors=-1,2,3,5) ===")
        factor_to_emb_loose, _, _ = compute_projected_embeddings(embeddings, FACTORS_LOOSE, concept_dir)
        factor_to_2d_raw = umap_aligned(factor_to_emb_loose, FACTORS_LOOSE, alignment_regularisation=1e-3)
        factor_to_2d_norm = normalise_coords(factor_to_2d_raw, FACTORS_LOOSE)
        build_and_save(
            client, texts, embeddings, concept_dir, cosine_sims,
            FACTORS_LOOSE, factor_to_2d_norm,
            display_name="Machine Learning (Loose Aligned)",
            output_filename="machine_learning_loose_aligned",
        )

    print("\nAll variants done.")


if __name__ == "__main__":
    main()
