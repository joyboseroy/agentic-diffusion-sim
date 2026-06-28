"""
cluster_occurrences_v2.py
=========================
Step 2 — no sentence-transformers dependency.
Uses TF-IDF + SVD for embeddings (pure sklearn, no torch/numpy conflict).
Then labels clusters with Groq LLM.

Install: pip install scikit-learn --break-system-packages  (already have it)

Run:
    python3 cluster_occurrences_v2.py --concept maya
    python3 cluster_occurrences_v2.py  # all concepts >= 15 occurrences
"""

import json, argparse, os, time, re
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

MIN_OCCURRENCES = 15
MAX_K           = 5
MIN_K           = 2
GROQ_MODEL      = "llama-3.3-70b-versatile"
REPRESENTATIVE_N = 5

def load_occurrences(path):
    with open(path) as f:
        return json.load(f)

def embed_tfidf(texts, n_components=40):
    """
    TF-IDF + SVD embedding. No torch, no numpy conflicts.
    Returns normalised (n, n_components) array.
    """
    # Clean texts
    clean = [re.sub(r'\s+', ' ', t.strip().lower()) for t in texts]
    
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents='unicode',
    )
    X = tfidf.fit_transform(clean)
    
    # SVD to dense
    n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    if n_comp < 2:
        return normalize(X.toarray())
    
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_dense = svd.fit_transform(X)
    return normalize(X_dense)

def best_k(embeddings, max_k=MAX_K, min_k=MIN_K):
    n = len(embeddings)
    if n < min_k * 3:
        return 1
    best_score = -1
    best_k_val = 1
    for k in range(min_k, min(max_k + 1, n // 3 + 1)):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k_val = k
        except Exception:
            continue
    if best_score < 0.08:   # lower threshold for TF-IDF
        return 1
    return best_k_val

def label_cluster_with_groq(concept, cluster_id, passages, groq_client):
    passages_text = "\n\n".join(
        f"[{i+1}] (school={p['school']}, relation={p['relation']}):\n{p['evidence'][:300]}"
        for i, p in enumerate(passages)
    )
    prompt = f"""You are a specialist in Indian philosophy comparing Buddhist, Jain, and Hindu traditions.

I clustered all occurrences of the concept "{concept}" in a philosophical corpus.
Below are {len(passages)} representative passages from ONE cluster.

Give a SHORT label (4-8 words) capturing the PHILOSOPHICAL SENSE of "{concept}" in these passages.
Be specific about tradition and meaning, e.g.:
- "maya as Vedic magical power of gods"
- "maya as cosmic illusion in Advaita"  
- "karma as ritual sacrificial action in Vedas"
- "karma as intentional action in Buddhist ethics"
- "dharma as universal law in Upanishads"
- "dharma as phenomena in Pali Buddhism"

Passages:
{passages_text}

Reply with ONLY the short label. Nothing else."""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=25,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip().strip('"\'')
    except Exception as e:
        print(f"    Groq error: {e}")
        return f"{concept}_sense_{cluster_id}"

def cluster_concept(concept, occurrences, groq_client, max_k=MAX_K, verbose=True):
    texts = [o["evidence"] for o in occurrences]
    
    print(f"    Embedding {len(texts)} passages (TF-IDF/SVD)...")
    embeddings = embed_tfidf(texts)
    
    k = best_k(embeddings, max_k=max_k)
    print(f"    Best k={k}")
    
    if k == 1:
        for o in occurrences:
            o["cluster_id"] = 0
            o["sense_label"] = f"{concept} (single sense in corpus)"
        return occurrences, {0: f"{concept} (single sense in corpus)"}
    
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    for o, lbl in zip(occurrences, labels):
        o["cluster_id"] = int(lbl)
    
    cluster_labels = {}
    for cid in range(k):
        cluster_occs = [o for o in occurrences if o["cluster_id"] == cid]
        
        # Representatives: closest to centroid
        cidx = [i for i, o in enumerate(occurrences) if o["cluster_id"] == cid]
        cemb = embeddings[cidx]
        centroid = cemb.mean(axis=0)
        dists = np.linalg.norm(cemb - centroid, axis=1)
        rep_idx = np.argsort(dists)[:REPRESENTATIVE_N]
        reps = [cluster_occs[i] for i in rep_idx]
        
        schools = Counter(o["school"] for o in cluster_occs)
        top_schools = ", ".join(f"{s}({n})" for s, n in schools.most_common(3))
        
        print(f"\n    Cluster {cid}: {len(cluster_occs)} occ | {top_schools}")
        for r in reps[:2]:
            print(f"      [{r['school']}|{r['relation']}] "
                  f"{r['evidence'][:100].replace(chr(10),' ')}...")
        
        if groq_client:
            label = label_cluster_with_groq(concept, cid, reps, groq_client)
            time.sleep(0.3)
        else:
            label = f"{concept}_cluster_{cid}"
        
        print(f"    -> '{label}'")
        cluster_labels[cid] = label
    
    for o in occurrences:
        o["sense_label"] = cluster_labels[o["cluster_id"]]
    
    return occurrences, cluster_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="occurrences.json")
    parser.add_argument("--output", default="occurrences_clustered.json")
    parser.add_argument("--concept", default=None)
    parser.add_argument("--min_occurrences", type=int, default=MIN_OCCURRENCES)
    parser.add_argument("--max_k", type=int, default=MAX_K)
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Step 2 v2: Clustering (TF-IDF/SVD, no torch dependency)")
    print("="*65 + "\n")

    data = load_occurrences(args.input)
    concepts_data = data["concepts"]

    groq_client = None
    if not args.no_llm:
        key = os.environ.get("GROQ_API_KEY","")
        if key:
            try:
                from groq import Groq
                groq_client = Groq(api_key=key)
                print(f"  Groq ready ({GROQ_MODEL})\n")
            except ImportError:
                print("  groq package missing — using cluster_N labels")
        else:
            print("  GROQ_API_KEY not set — using cluster_N labels")
            print("  Set: export GROQ_API_KEY=your_key\n")

    if args.concept:
        to_process = [args.concept]
    else:
        to_process = [c for c, d in concepts_data.items()
                      if d["count"] >= args.min_occurrences]
    
    print(f"  Processing {len(to_process)} concepts\n")
    
    all_labels = {}
    for concept in to_process:
        if concept not in concepts_data:
            print(f"  SKIP: '{concept}' not found"); continue
        
        occs = concepts_data[concept]["occurrences"]
        print(f"\n{'='*55}")
        print(f"  [{concept}] — {len(occs)} occurrences")
        
        enriched, labels = cluster_concept(
            concept, occs, groq_client, max_k=args.max_k)
        
        concepts_data[concept]["occurrences"] = enriched
        concepts_data[concept]["sense_labels"] = labels
        all_labels[concept] = labels

    # Summary
    print(f"\n{'='*65}")
    print("  SENSE SUMMARY")
    print(f"{'='*65}")
    for concept, labels in all_labels.items():
        print(f"\n  {concept}: {len(labels)} sense(s)")
        for cid, label in sorted(labels.items()):
            n = sum(1 for o in concepts_data[concept]["occurrences"]
                    if o["cluster_id"] == cid)
            print(f"    [{cid}] {label}  ({n} occ)")

    data["concepts"] = concepts_data
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output}")

if __name__ == "__main__":
    main()
