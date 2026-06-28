"""
cluster_occurrences.py
======================
Step 2 of the diachronic concept graph pipeline.

For each concept with enough occurrences, clusters them by semantic sense
using evidence quote embeddings. Then calls an LLM to label each cluster.

Pipeline:
  1. Load occurrences.json (from extract_occurrences.py)
  2. For each concept with >= MIN_OCCURRENCES:
     a. Embed each occurrence evidence quote (Sentence-BERT)
     b. Cluster embeddings (KMeans, auto-select k via silhouette score)
     c. Label each cluster via Groq LLM (given 5 representative passages)
  3. Save enriched occurrences with cluster_id and sense_label
  4. Print summary showing sense breakdown per concept

Install:
    pip install sentence-transformers scikit-learn --break-system-packages

Run:
    python3 cluster_occurrences.py
    python3 cluster_occurrences.py --concept maya
    python3 cluster_occurrences.py --min_occurrences 20 --max_k 6
"""

import json, argparse, os, time
from collections import defaultdict, Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_OCCURRENCES = 15   # skip concepts with fewer occurrences
MAX_K           = 5    # maximum number of sense clusters to try
MIN_K           = 2    # minimum clusters (1 = no disambiguation needed)
GROQ_MODEL      = "llama-3.3-70b-versatile"
EMBED_MODEL     = "all-MiniLM-L6-v2"  # fast, good for short texts
REPRESENTATIVE_N = 5   # passages per cluster to show LLM

def load_occurrences(path="occurrences.json"):
    with open(path) as f:
        data = json.load(f)
    return data

def get_embeddings(texts, model):
    """Embed a list of texts. Returns numpy array (n, dim)."""
    print(f"    Embedding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=False,
                               batch_size=64, normalize_embeddings=True)
    return embeddings

def best_k(embeddings, max_k=MAX_K, min_k=MIN_K):
    """
    Find the best number of clusters using silhouette score.
    Returns k=1 if all silhouette scores are low (no meaningful clustering).
    """
    n = len(embeddings)
    if n < min_k * 3:
        return 1   # too few to cluster

    best_score = -1
    best_k_val = 1
    for k in range(min_k, min(max_k + 1, n // 3 + 1)):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k_val = k

    # If best silhouette < threshold, don't split — one sense is fine
    if best_score < 0.12:
        return 1
    return best_k_val

def label_cluster_with_llm(concept, cluster_id, passages, groq_client):
    """Ask Groq LLM to give a short philosophical sense label for this cluster."""
    passages_text = "\n\n".join(
        f"[{i+1}] ({p['school']}, {p['relation']}): {p['evidence'][:300]}"
        for i, p in enumerate(passages)
    )

    prompt = f"""You are a specialist in Indian philosophy.

I have clustered all occurrences of the concept "{concept}" in a philosophical text corpus.
Below are {len(passages)} representative passages from ONE cluster.

Your task: give a SHORT label (3-7 words) capturing the PHILOSOPHICAL SENSE
of "{concept}" as used in these passages.

Be specific. For example:
- "maya as cosmic illusion obscuring brahman" (not just "illusion")
- "karma as ritual sacrificial action" (not just "karma")
- "dharma as phenomenon in Pali canon" (not just "dharma")

Passages:
{passages_text}

Reply with ONLY the short label. No explanation. No preamble."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.1,
        )
        label = response.choices[0].message.content.strip()
        # Clean up
        label = label.strip('"\'').strip()
        return label
    except Exception as e:
        print(f"    LLM error: {e}")
        return f"cluster_{cluster_id}"

def cluster_concept(concept, occurrences, embed_model, groq_client,
                    max_k=MAX_K, verbose=True):
    """
    Cluster all occurrences of one concept.
    Returns occurrences enriched with cluster_id and sense_label.
    """
    texts = [o["evidence"] for o in occurrences]

    # Embed
    embeddings = get_embeddings(texts, embed_model)

    # Find best k
    k = best_k(embeddings, max_k=max_k)
    if verbose:
        print(f"    Best k={k} for {len(occurrences)} occurrences of '{concept}'")

    if k == 1:
        # No meaningful clustering — all one sense
        for o in occurrences:
            o["cluster_id"] = 0
            o["sense_label"] = f"{concept} (undifferentiated)"
        return occurrences, {0: f"{concept} (undifferentiated)"}

    # Cluster
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    for o, label in zip(occurrences, labels):
        o["cluster_id"] = int(label)

    # For each cluster, get representative passages and label
    cluster_labels = {}
    for cluster_id in range(k):
        cluster_occs = [o for o in occurrences if o["cluster_id"] == cluster_id]

        # Find representatives: closest to cluster centroid
        cluster_embs = embeddings[labels == cluster_id]
        centroid = cluster_embs.mean(axis=0)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        rep_indices = np.argsort(dists)[:REPRESENTATIVE_N]
        representatives = [cluster_occs[i] for i in rep_indices]

        # School breakdown for this cluster
        schools = Counter(o["school"] for o in cluster_occs)
        top_schools = ", ".join(f"{s}({n})" for s, n in schools.most_common(3))

        if verbose:
            print(f"    Cluster {cluster_id}: {len(cluster_occs)} occurrences"
                  f" | {top_schools}")
            for rep in representatives[:2]:
                ev = rep["evidence"][:100].replace("\n", " ")
                print(f"      [{rep['school']}] {ev}...")

        # Label via LLM
        if groq_client:
            sense_label = label_cluster_with_llm(
                concept, cluster_id, representatives, groq_client
            )
        else:
            sense_label = f"{concept}_cluster_{cluster_id}"

        cluster_labels[cluster_id] = sense_label
        if verbose:
            print(f"    -> Label: '{sense_label}'")

    # Apply labels
    for o in occurrences:
        o["sense_label"] = cluster_labels[o["cluster_id"]]

    return occurrences, cluster_labels

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Cluster occurrences into philosophical senses"
    )
    parser.add_argument("--input",  type=str, default="occurrences.json")
    parser.add_argument("--output", type=str, default="occurrences_clustered.json")
    parser.add_argument("--concept", type=str, default=None,
                        help="Cluster only this concept (for testing)")
    parser.add_argument("--min_occurrences", type=int, default=MIN_OCCURRENCES)
    parser.add_argument("--max_k", type=int, default=MAX_K)
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM labelling (use cluster_N labels)")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Step 2: Clustering occurrences into philosophical senses")
    print("="*65 + "\n")

    # Load occurrences
    data = load_occurrences(args.input)
    concepts_data = data["concepts"]

    # Load embedding model
    print("  Loading embedding model (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(EMBED_MODEL)
        print("  Model loaded.")
    except ImportError:
        print("  ERROR: sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers --break-system-packages")
        return

    # Load Groq client
    groq_client = None
    if not args.no_llm:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if groq_api_key:
            try:
                from groq import Groq
                groq_client = Groq(api_key=groq_api_key)
                print(f"  Groq client loaded (model: {GROQ_MODEL})")
            except ImportError:
                print("  WARNING: groq package not installed. Using cluster_N labels.")
        else:
            print("  WARNING: GROQ_API_KEY not set. Using cluster_N labels.")
            print("  Set it with: export GROQ_API_KEY=your_key_here")

    # Select concepts to process
    if args.concept:
        concepts_to_process = [args.concept]
    else:
        concepts_to_process = [
            c for c, d in concepts_data.items()
            if d["count"] >= args.min_occurrences
        ]
        print(f"  Processing {len(concepts_to_process)} concepts "
              f"with >= {args.min_occurrences} occurrences\n")

    # Cluster each concept
    all_sense_labels = {}  # concept -> {cluster_id -> label}
    for concept in concepts_to_process:
        if concept not in concepts_data:
            print(f"  SKIP: '{concept}' not in occurrences file")
            continue

        occs = concepts_data[concept]["occurrences"]
        n = len(occs)
        print(f"\n  [{concept}] — {n} occurrences")

        enriched, labels = cluster_concept(
            concept, occs, embed_model, groq_client,
            max_k=args.max_k, verbose=True
        )
        concepts_data[concept]["occurrences"] = enriched
        concepts_data[concept]["sense_labels"] = labels
        all_sense_labels[concept] = labels

        # Small delay to avoid Groq rate limits
        if groq_client:
            time.sleep(0.5)

    # Summary
    print(f"\n{'='*65}")
    print("  SENSE DISAMBIGUATION SUMMARY")
    print(f"{'='*65}")
    for concept, labels in all_sense_labels.items():
        if len(labels) == 1:
            print(f"\n  {concept}: 1 sense (undifferentiated)")
        else:
            print(f"\n  {concept}: {len(labels)} senses")
            for cid, label in labels.items():
                n = sum(1 for o in concepts_data[concept]["occurrences"]
                        if o["cluster_id"] == cid)
                print(f"    [{cid}] {label} ({n} occurrences)")

    # Save
    data["concepts"] = concepts_data
    data["metadata"]["pipeline_step"] = "2_clustering"
    data["metadata"]["next_step"] = "python3 build_sense_graph.py"

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {args.output}")
    print(f"  Next: python3 build_sense_graph.py --input {args.output}")

if __name__ == "__main__":
    main()
