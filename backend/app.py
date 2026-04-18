import os
import sys

# Block tensorflow before umap loads it via umap.parametric_umap.
# TF segfaults on this machine; umap catches ImportError and skips ParametricUMAP.
sys.modules.setdefault("tensorflow", None)  # type: ignore[assignment]

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import glob
import json
import pickle
import random
import re
import threading
import uuid
from datetime import datetime
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# ---------------------------------------------------------------------------
# Startup: load datasets, UMAP models, SentenceBERT model
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Latent Manipulator config
# ---------------------------------------------------------------------------
# Results are stored inside this project under backend/lm_results/
LM_RESULTS_DIR = os.path.join(BASE_DIR, "lm_results")
os.makedirs(LM_RESULTS_DIR, exist_ok=True)

# Background jobs for concept generation
lm_jobs = {}

# In-memory cache: (dataset, concept_filename) -> visualization payload
lm_viz_cache = {}

datasets = {}
umap_models = {}
sentence_model = None
openai_client = None

def load_resources():
    global datasets, umap_models, sentence_model, openai_client

    # Load datasets.json
    datasets_path = os.path.join(BASE_DIR, "datasets.json")
    if os.path.exists(datasets_path):
        with open(datasets_path, "r") as f:
            datasets = json.load(f)
        print(f"[startup] Loaded datasets: {list(datasets.keys())}")
    else:
        print("[startup] WARNING: datasets.json not found. Run generate_sample_data.py first.")

    # Load UMAP pickle files
    for name in ["vispub", "frankenstein", "physics"]:
        pkl_path = os.path.join(BASE_DIR, f"{name}_umap.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                umap_models[name] = pickle.load(f)
            print(f"[startup] Loaded UMAP model: {name}")
        else:
            print(f"[startup] WARNING: {name}_umap.pkl not found.")

    # Load SentenceBERT model
    try:
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[startup] SentenceBERT model loaded.")
    except Exception as e:
        print(f"[startup] WARNING: Could not load SentenceBERT: {e}")

    # OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        print("[startup] OpenAI client initialized.")
    else:
        print("[startup] WARNING: OPENAI_API_KEY not set.")

    # Pre-warm LM visualization cache for all existing result files
    result_files = glob.glob(os.path.join(LM_RESULTS_DIR, "results_*.json"))
    for fp in result_files:
        m = re.match(r'results_(.+?)_(.+)\.json', os.path.basename(fp))
        if not m:
            continue
        ds_name, concept_fn = m.group(1), m.group(2)
        try:
            lm_load_visualization(ds_name, concept_fn)
            print(f"[startup] Cached LM viz: {ds_name}/{concept_fn}")
        except Exception as e:
            print(f"[startup] WARNING: Could not cache {fp}: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    """Return list of available dataset names."""
    return jsonify(["vispub", "frankenstein", "physics"])


@app.route("/api/dataset/<name>", methods=["GET"])
def get_dataset(name):
    """Return texts, coords, labels, semantic_labels for a dataset (no raw embeddings)."""
    if name not in datasets:
        return jsonify({"error": f"Dataset '{name}' not found."}), 404

    ds = datasets[name]

    # cluster_labels is a dict {"0": "name", "1": "name", ...}
    # Convert to an array ordered by cluster index for the frontend.
    cluster_labels_dict = ds.get("cluster_labels") or ds.get("semantic_labels") or {}
    if isinstance(cluster_labels_dict, dict):
        max_idx = max((int(k) for k in cluster_labels_dict), default=-1)
        semantic_labels = [
            cluster_labels_dict.get(str(i), f"Cluster {i}")
            for i in range(max_idx + 1)
        ]
    else:
        semantic_labels = cluster_labels_dict  # already a list

    payload = {
        "texts": ds["texts"],
        "coords": ds["coords"],
        "labels": ds["labels"],
        "semantic_labels": semantic_labels,
    }
    return jsonify(payload)


@app.route("/api/explain", methods=["POST"])
def explain():
    """Call OpenAI to explain why selected texts cluster together."""
    if openai_client is None:
        return jsonify({"error": "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."}), 503

    body = request.get_json(force=True, silent=True) or {}
    dataset_name = body.get("dataset", "")
    selected_texts = body.get("selected_texts", [])
    all_texts_sample = body.get("all_texts_sample", [])
    model = body.get("model", "gpt-5-nano-2025-08-07")
    prompt_template = body.get("prompt_template", None)

    if not selected_texts:
        return jsonify({"error": "No texts selected."}), 400

    selected_block = "\n".join(f"- {t}" for t in selected_texts[:30])
    other_block = "\n".join(f"- {t}" for t in all_texts_sample[:40])

    if prompt_template:
        # User-supplied template: substitute placeholders and skip dataset context
        prompt = prompt_template.replace("{selected_texts}", selected_block).replace("{other_texts}", other_block)
    else:
        DATASET_CONTEXT = {
            "frankenstein": "All texts are passages from Mary Shelley's novel Frankenstein. ",
            "physics": "All texts are physics research paper abstracts from arXiv. ",
        }
        context = DATASET_CONTEXT.get(dataset_name, "")
        prompt = (
            f"You are an expert at analyzing clusters of text. {context}"
            "Given the following texts that are clustered together in an embedding visualization, "
            "return a JSON object with two fields:\n"
            "- \"cluster_label\": a short 2-4 word title for this cluster (title case, no punctuation)\n"
            "- \"explanation\": a single sentence explaining what these texts have in common and what "
            "distinguishes them from the rest of the corpus. Be direct and specific — no preamble, no filler.\n\n"
            f"Selected texts:\n{selected_block}\n\n"
            f"Other texts (for contrast):\n{other_block}"
        )

    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "cluster_explanation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "cluster_label": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["cluster_label", "explanation"],
                "additionalProperties": False,
            },
        },
    }

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=json_schema,
        )
        result = json.loads(response.choices[0].message.content or "{}")
        explanation = result.get("explanation", "").strip()
        cluster_label = result.get("cluster_label", "").strip()
        return jsonify({"explanation": explanation, "cluster_label": cluster_label})
    except Exception as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 502


@app.route("/api/reproject", methods=["POST"])
def reproject():
    """Embed a new text with SentenceBERT and project it with the pre-fitted UMAP model."""
    body = request.get_json(force=True, silent=True) or {}
    dataset_name = body.get("dataset", "vispub")
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    if sentence_model is None:
        return jsonify({"error": "SentenceBERT model not loaded."}), 503

    if dataset_name not in umap_models:
        return jsonify({"error": f"UMAP model for '{dataset_name}' not loaded. Run generate_sample_data.py first."}), 503

    try:
        embedding = sentence_model.encode([text])
        umap_model = umap_models[dataset_name]
        coord = umap_model.transform(embedding)  # shape (1, 2)
        x, y = float(coord[0][0]), float(coord[0][1])
        return jsonify({"coord": [x, y], "text": text})
    except Exception as e:
        return jsonify({"error": f"Reprojection error: {str(e)}"}), 500


@app.route("/api/phrasecloud", methods=["POST"])
def phrasecloud():
    """Train a LinearSVC on bag-of-phrases and return top distinctive phrases."""
    body = request.get_json(force=True, silent=True) or {}
    dataset_name = body.get("dataset", "vispub")
    selected_indices = body.get("selected_indices", [])

    if dataset_name not in datasets:
        return jsonify({"error": f"Dataset '{dataset_name}' not found."}), 404

    texts = datasets[dataset_name]["texts"]

    if len(selected_indices) < 3:
        return jsonify({"error": "Selection too small for phrasecloud (minimum 3 texts)"}), 400

    selected_set = set(selected_indices)
    in_cluster = [texts[i] for i in selected_indices if i < len(texts)]
    out_cluster = [texts[i] for i in range(len(texts)) if i not in selected_set]

    if len(in_cluster) < 3:
        return jsonify({"error": "Selection too small for phrasecloud (minimum 3 texts)"}), 400

    all_texts = in_cluster + out_cluster
    labels = [1] * len(in_cluster) + [-1] * len(out_cluster)

    try:
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=10000)
        X = vectorizer.fit_transform(all_texts)
        clf = LinearSVC(class_weight="balanced", max_iter=10000, C=1.0)
        clf.fit(X, labels)
        feature_names = vectorizer.get_feature_names_out()
        weights = clf.coef_[0]
        top_indices = np.argsort(weights)[::-1][:50]
        phrases = [
            {"text": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_indices
            if weights[i] > 0
        ][:30]
        return jsonify({"phrases": phrases})
    except Exception as e:
        return jsonify({"error": f"Phrasecloud error: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Latent Manipulator helpers
# ---------------------------------------------------------------------------

def lm_get_datasets():
    return sorted(datasets.keys())


def lm_get_concepts(dataset_name):
    pattern = os.path.join(LM_RESULTS_DIR, f"results_{dataset_name}_*.json")
    concepts = []
    for fp in glob.glob(pattern):
        m = re.match(rf'results_{re.escape(dataset_name)}_(.+)\.json', os.path.basename(fp))
        if not m:
            continue
        concept_filename = m.group(1)
        try:
            with open(fp) as f:
                data = json.load(f)
            concepts.append({
                "filename": concept_filename,
                "name": data.get("concept_direction", concept_filename),
                "data_points": data.get("metadata", {}).get("total_data_points", 0),
                "projection_factors": data.get("projection_factors", []),
            })
        except Exception as e:
            print(f"[lm] Error loading {fp}: {e}")
    return concepts


def lm_load_text_data(dataset_name):
    if dataset_name in datasets:
        return datasets[dataset_name]["texts"]
    return None


def lm_load_visualization(dataset_name, concept_filename):
    cache_key = (dataset_name, concept_filename)
    if cache_key in lm_viz_cache:
        return lm_viz_cache[cache_key]

    fp = os.path.join(LM_RESULTS_DIR, f"results_{dataset_name}_{concept_filename}.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        full = json.load(f)

    # Read cosine similarities from file (pre-computed, not dependent on projection factor)
    cosine_sims = full.get("cosine_similarities", None)

    texts = lm_load_text_data(dataset_name)

    result = {
        "dataset": full["dataset"],
        "concept_direction": full["concept_direction"],
        "metadata": full.get("metadata", {}),
        "projection_factors": full.get("projection_factors", []),
        "cosine_similarities": cosine_sims,
        "texts": texts,
        "snapshots": [],
    }

    # Global normalization: compute min/max across ALL snapshots so the shared
    # UMAP coordinate space is preserved when displayed.
    orig = full.get("original", {})
    all_raw_coords = []
    if orig.get("embeddings_2d"):
        all_raw_coords.extend(orig["embeddings_2d"])
    for pdata in full.get("projected", {}).values():
        all_raw_coords.extend(pdata.get("embeddings_2d", []))

    if all_raw_coords:
        arr_all = np.array(all_raw_coords)
        g_mnx, g_mxx = float(arr_all[:, 0].min()), float(arr_all[:, 0].max())
        g_mny, g_mxy = float(arr_all[:, 1].min()), float(arr_all[:, 1].max())
        g_rx = g_mxx - g_mnx if g_mxx != g_mnx else 1.0
        g_ry = g_mxy - g_mny if g_mxy != g_mny else 1.0
    else:
        g_mnx, g_mny, g_rx, g_ry = 0.0, 0.0, 1.0, 1.0

    def norm(coords):
        return [[(x - g_mnx) / g_rx, (y - g_mny) / g_ry] for x, y in coords]

    if orig.get("embeddings_2d"):
        result["snapshots"].append({
            "factor": 0, "factor_label": "original",
            "embeddings_2d": norm(orig["embeddings_2d"]),
            "cluster_labels": orig.get("cluster_labels", []),
            "cluster_topics": orig.get("cluster_topics", {}),
            "n_clusters": orig.get("n_clusters", 0),
            "n_noise": orig.get("n_noise", 0),
            "silhouette_score": orig.get("silhouette_score"),
            "purity_metrics": orig.get("purity_metrics", {}),
        })

    for _, pdata in full.get("projected", {}).items():
        result["snapshots"].append({
            "factor": pdata["factor"], "factor_label": pdata.get("factor_label", ""),
            "embeddings_2d": norm(pdata["embeddings_2d"]),
            "cluster_labels": pdata["cluster_labels"],
            "cluster_topics": pdata.get("cluster_topics", {}),
            "n_clusters": pdata.get("n_clusters", 0),
            "n_noise": pdata.get("n_noise", 0),
            "silhouette_score": pdata.get("silhouette_score"),
            "purity_metrics": pdata.get("purity_metrics", {}),
        })

    result["snapshots"].sort(key=lambda x: x["factor"])
    lm_viz_cache[cache_key] = result
    return result


class TCEConceptWorker:
    """
    Concept projection worker that uses pre-loaded TCE embeddings.
    Skips re-embedding; uses datasets.json embeddings directly.
    """

    def __init__(self, job_id, dataset_name, embeddings, texts, concept_name,
                 projection_factors, output_dir, openai_client):
        self.job_id = job_id
        self.dataset_name = dataset_name
        self.embeddings = embeddings  # list of lists (N x 384)
        self.texts = texts
        self.concept_name = concept_name
        self.projection_factors = projection_factors
        self.output_dir = output_dir
        self.client = openai_client
        self.concept_filename = concept_name.lower().replace(' ', '_')
        self.output_file = os.path.join(
            output_dir,
            f"results_{dataset_name}_{self.concept_filename}.json"
        )

    def run(self, progress_callback=None):
        def progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)
            print(f"[{self.job_id}] {pct}%: {msg}")

        import numpy as np
        import umap
        import hdbscan
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer

        progress(5, "Preparing embeddings…")
        embeddings = np.array(self.embeddings)
        texts = self.texts

        progress(15, f"Generating '{self.concept_name}' direction vector…")
        concept_direction = self._generate_concept_direction()

        concept_dir_norm = concept_direction / np.linalg.norm(concept_direction)
        projections = embeddings @ concept_dir_norm
        cosine_sims = cosine_similarity(embeddings, concept_direction.reshape(1, -1)).flatten()

        n = len(texts)

        # Build per-factor projected embedding matrices
        factor_to_proj_emb = {}
        all_factor_embs = [(0, embeddings)]
        for factor in self.projection_factors:
            if factor == -1:
                pe = embeddings - np.outer(projections, concept_dir_norm)
            else:
                pe = embeddings + factor * np.outer(projections, concept_dir_norm)
            factor_to_proj_emb[factor] = pe
            all_factor_embs.append((factor, pe))
        all_factor_embs.sort(key=lambda x: x[0])
        all_factors_sorted = [f for f, _ in all_factor_embs]
        all_embs_sorted = [e for _, e in all_factor_embs]

        # AlignedUMAP with caching
        factor_to_2d = self._compute_aligned_umap(
            concept_direction, all_factors_sorted, all_embs_sorted, n, progress
        )

        embeddings_2d = factor_to_2d[0]

        progress(40, "Clustering original embeddings…")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
        orig_labels = clusterer.fit_predict(embeddings_2d)

        progress(48, "Generating topic labels for original clusters…")
        orig_topics = self._generate_topic_labels(texts, embeddings, orig_labels)

        orig_n_clusters = int(len(set(orig_labels)) - (1 if -1 in orig_labels else 0))
        orig_n_noise = int(list(orig_labels).count(-1))
        orig_sil = None
        nn_mask = orig_labels != -1
        if np.sum(nn_mask) > 1 and orig_n_clusters > 1:
            try:
                orig_sil = float(silhouette_score(embeddings[nn_mask], orig_labels[nn_mask], metric='cosine'))
            except Exception:
                pass

        results = {
            "dataset": self.dataset_name,
            "concept_direction": self.concept_name,
            "concept_direction_vector": concept_direction.tolist(),
            "projection_factors": self.projection_factors,
            "cosine_similarities": cosine_sims.tolist(),
            "original": {
                "embeddings_2d": embeddings_2d.tolist(),
                "cluster_labels": orig_labels.tolist(),
                "cluster_topics": {int(k): str(v) for k, v in orig_topics.items()},
                "n_clusters": orig_n_clusters,
                "n_noise": orig_n_noise,
                "silhouette_score": orig_sil,
                "purity_metrics": self._cluster_purity(embeddings, orig_labels),
            },
            "projected": {},
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset_name": self.dataset_name,
                "concept_direction": self.concept_name,
                "total_data_points": len(texts),
                "embedding_dimension": embeddings.shape[1],
            },
        }

        n_factors = len(self.projection_factors)
        for i, factor in enumerate(self.projection_factors):
            base = 55 + (i * 40 // n_factors)
            label = "removed" if factor == -1 else f"{factor}X_amplified"
            emb_2d = factor_to_2d[factor]
            emb_proj = factor_to_proj_emb[factor]

            progress(base, f"Clustering {label}…")
            c2 = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=3, metric='euclidean')
            cl = c2.fit_predict(emb_2d)

            n_cl = int(len(set(cl)) - (1 if -1 in cl else 0))
            n_no = int(list(cl).count(-1))

            progress(base + 5, f"Topic labels for {label}…")
            topics = self._generate_topic_labels(texts, embeddings, cl)

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
                "embeddings_2d": emb_2d.tolist(),
                "cluster_labels": cl.tolist(),
                "cluster_topics": {int(k): str(v) for k, v in topics.items()},
                "n_clusters": n_cl,
                "n_noise": n_no,
                "silhouette_score": sil,
                "purity_metrics": self._cluster_purity(emb_proj, cl),
            }

        progress(97, "Saving results…")
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

        progress(100, f"Done — saved to {self.output_file}")
        return results

    def _compute_aligned_umap(self, concept_direction, all_factors_sorted, all_embs_sorted, n, progress):
        import hashlib
        import json as _json
        import umap

        cache_file = os.path.join(BASE_DIR, "aligned_umap_embeddings.json")
        factors_str = ",".join(str(f) for f in sorted(self.projection_factors))
        dir_data = ",".join(f"{v:.5f}" for v in concept_direction)
        dir_hash = hashlib.sha256(dir_data.encode()).hexdigest()[:10]
        cache_key = f"{self.dataset_name}__{self.concept_filename}__{factors_str}__{dir_hash}"

        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cache = _json.load(f)
            except Exception:
                cache = {}

        if cache_key in cache:
            progress(35, "Using cached AlignedUMAP embeddings…")
            cached = cache[cache_key]
            factor_to_2d = {}
            for k, v in cached.items():
                fk = int(k) if k != "0" else 0
                factor_to_2d[fk] = np.array(v)
            if all(f in factor_to_2d for f in all_factors_sorted):
                return factor_to_2d

        progress(25, f"AlignedUMAP on {len(all_embs_sorted)} snapshots ({n} points each)…")
        relations = [{i: i for i in range(n)} for _ in range(len(all_embs_sorted) - 1)]
        aligned_mapper = umap.AlignedUMAP(n_neighbors=15, min_dist=0.1, alignment_regularisation=1e-3, random_state=42)
        aligned_mapper.fit(all_embs_sorted, relations=relations)
        all_2d = aligned_mapper.embeddings_

        factor_to_2d = {f: all_2d[j] for j, f in enumerate(all_factors_sorted)}

        cache[cache_key] = {str(f): emb.tolist() for f, emb in factor_to_2d.items()}
        try:
            with open(cache_file, "w") as f:
                _json.dump(cache, f, ensure_ascii=False)
        except Exception as e:
            print(f"[{self.job_id}] WARNING: Could not save UMAP cache: {e}")

        return factor_to_2d

    def _generate_concept_direction(self):
        import numpy as np
        import json as _json
        from sentence_transformers import SentenceTransformer

        cache_file = os.path.join(BASE_DIR, "concept_directions.json")
        cache_key = f"{self.dataset_name}__{self.concept_name}"

        # Load cache
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cache = _json.load(f)
            except Exception:
                cache = {}

        if cache_key in cache:
            print(f"[{self.job_id}] Using cached concept direction for '{self.concept_name}'")
            entry = cache[cache_key]
            vec = entry["direction"] if isinstance(entry, dict) else entry
            return np.array(vec)

        DATASET_CONCEPT_CONTEXT = {
            "physics": (
                "The texts are physics research paper abstracts. "
                "Generate texts that are representative of physics research abstracts — "
                "technical, concise, and domain-specific. "
            ),
            "frankenstein": (
                "The texts are passages from Mary Shelley's novel Frankenstein. "
                "Generate texts that match the Gothic, Romantic, and 19th-century literary style of the novel. "
            ),
        }
        ctx = DATASET_CONCEPT_CONTEXT.get(self.dataset_name, "")
        prompt = (
            f"{ctx}Generate 20 diverse texts (150-200 words each) strongly focused on the concept of "
            f"'{self.concept_name}'. Return ONLY the texts separated by '---'."
        )
        resp = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
        )
        raw = resp.choices[0].message.content.strip()
        concept_texts = [t.strip() for t in raw.split("---") if t.strip()]
        model = SentenceTransformer("all-MiniLM-L6-v2")
        concept_embs = model.encode(concept_texts, show_progress_bar=False)
        direction = concept_embs.mean(axis=0)

        cache[cache_key] = {
            "texts": concept_texts,
            "direction": direction.tolist(),
        }
        try:
            with open(cache_file, "w") as f:
                _json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[{self.job_id}] WARNING: Could not save concept direction cache: {e}")

        return direction

    def _generate_topic_labels(self, texts, embeddings, cluster_labels, n_examples=5):
        import numpy as np

        cluster_topics = {}
        for cid in sorted(set(cluster_labels)):
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
                resp = self.client.chat.completions.create(
                    model="gpt-4",
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

    def _cluster_purity(self, embeddings, cluster_labels):
        import numpy as np
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
        return {"overall_purity": float(overall), "cluster_purities": {int(k): float(v) for k, v in purities.items()}}


def _run_lm_worker(job_id, worker):
    try:
        lm_jobs[job_id]["status"] = "running"

        def cb(pct, msg):
            lm_jobs[job_id]["progress"] = pct
            lm_jobs[job_id]["message"] = msg

        worker.run(cb)
        lm_jobs[job_id].update({"status": "completed", "progress": 100,
                                  "message": "Completed", "completed_at": datetime.now().isoformat()})
    except Exception as e:
        lm_jobs[job_id].update({"status": "failed", "error": str(e), "message": f"Failed: {e}"})


# ---------------------------------------------------------------------------
# Latent Manipulator API Routes  (/api/lm/*)
# ---------------------------------------------------------------------------

@app.route("/api/lm/datasets", methods=["GET"])
def lm_api_datasets():
    return jsonify({"datasets": lm_get_datasets()})


@app.route("/api/lm/concepts/<dataset>", methods=["GET"])
def lm_api_concepts(dataset):
    return jsonify({"concepts": lm_get_concepts(dataset)})


@app.route("/api/lm/results/<dataset>/<concept>", methods=["GET"])
def lm_api_results(dataset, concept):
    data = lm_load_visualization(dataset, concept)
    if data is None:
        return jsonify({"error": "Results not found"}), 404
    return jsonify(data)


@app.route("/api/lm/concepts/<dataset>", methods=["POST"])
def lm_api_create_concept(dataset):
    body = request.get_json(force=True, silent=True) or {}
    concept_name = body.get("concept_name", "").strip()
    projection_factors = body.get("projection_factors", [-1, 2, 5, 10])
    if not concept_name:
        return jsonify({"error": "concept_name is required"}), 400

    if dataset not in datasets:
        return jsonify({"error": f"Dataset '{dataset}' not found"}), 404

    if openai_client is None:
        return jsonify({"error": "OpenAI API key not configured"}), 503

    job_id = str(uuid.uuid4())[:8]
    worker = TCEConceptWorker(
        job_id=job_id, dataset_name=dataset,
        embeddings=datasets[dataset]["embeddings"],
        texts=datasets[dataset]["texts"],
        concept_name=concept_name,
        projection_factors=projection_factors,
        output_dir=LM_RESULTS_DIR,
        openai_client=openai_client,
    )
    lm_jobs[job_id] = {
        "status": "starting", "dataset": dataset, "concept_name": concept_name,
        "started_at": datetime.now().isoformat(), "progress": 0, "message": "Initializing...",
    }
    t = threading.Thread(target=_run_lm_worker, args=(job_id, worker))
    t.daemon = True
    t.start()
    return jsonify({"job_id": job_id, "status": "started"})


@app.route("/api/lm/worker/<job_id>/status", methods=["GET"])
def lm_api_worker_status(job_id):
    if job_id not in lm_jobs:
        return jsonify({"error": "Job not found"}), 404
    j = lm_jobs[job_id]
    return jsonify({
        "job_id": job_id, "status": j["status"],
        "progress": j.get("progress", 0), "message": j.get("message", ""),
        "dataset": j.get("dataset"), "concept_name": j.get("concept_name"),
        "started_at": j.get("started_at"), "completed_at": j.get("completed_at"),
        "error": j.get("error"),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_resources()
    app.run(host="0.0.0.0", port=5001, debug=False)
