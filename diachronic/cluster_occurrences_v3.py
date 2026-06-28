"""
cluster_occurrences_v3.py
=========================
Fixed version with two key changes:

1. Feature vector = school (one-hot) + relation type (one-hot) + TF-IDF text
   School label is the strongest signal for sense disambiguation.
   A maya occurrence in advaita school is almost certainly cosmic illusion.
   A maya occurrence in general school from early text is different.

2. Silhouette threshold lowered to 0.05 — TF-IDF scores are low,
   we should still split if there is ANY structure.

3. cluster_id assignment bug fixed — properly writes back to occurrences.

Run:
    python3 cluster_occurrences_v3.py --concept maya
    python3 cluster_occurrences_v3.py  # all >= 15 occurrences
"""

import json, argparse, os, time, re
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize, LabelEncoder
from scipy.sparse import hstack, csr_matrix

MIN_OCCURRENCES = 15
MAX_K           = 5
GROQ_MODEL      = "llama-3.3-70b-versatile"
REPRESENTATIVE_N = 5
SILHOUETTE_THRESHOLD = 0.04   # lower than v2

def load_occurrences(path):
    with open(path) as f:
        return json.load(f)

def make_features(occurrences):
    """
    Build feature matrix combining:
    - School one-hot (strongest signal)
    - Relation type one-hot
    - TF-IDF of evidence text (40 SVD dimensions)
    """
    texts    = [o["evidence"] for o in occurrences]
    schools  = [o["school"] for o in occurrences]
    relations = [o["relation"] for o in occurrences]

    n = len(texts)

    # --- School one-hot ---
    all_schools = sorted(set(schools))
    school_idx  = {s:i for i,s in enumerate(all_schools)}
    school_mat  = np.zeros((n, len(all_schools)))
    for i, s in enumerate(schools):
        school_mat[i, school_idx[s]] = 3.0  # weight schools 3x

    # --- Relation one-hot ---
    all_rels = sorted(set(relations))
    rel_idx  = {r:i for i,r in enumerate(all_rels)}
    rel_mat  = np.zeros((n, len(all_rels)))
    for i, r in enumerate(relations):
        rel_mat[i, rel_idx[r]] = 1.5

    # --- TF-IDF text ---
    clean = [re.sub(r'\s+', ' ', t.strip().lower()) for t in texts]
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1,2),
        min_df=1,
        sublinear_tf=True,
    )
    try:
        X_text = tfidf.fit_transform(clean)
        n_comp = min(30, X_text.shape[1]-1, n-1)
        if n_comp >= 2:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            text_mat = svd.fit_transform(X_text)
        else:
            text_mat = X_text.toarray()
    except Exception:
        text_mat = np.zeros((n, 1))

    # Combine
    combined = np.hstack([school_mat, rel_mat, text_mat])
    return normalize(combined)

def best_k(embeddings, n, max_k=MAX_K):
    if n < 6:
        return 1
    best_score = -1
    best_k_val = 1
    for k in range(2, min(max_k+1, n//2+1)):
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
    if best_score < SILHOUETTE_THRESHOLD:
        return 1
    print(f"    silhouette={best_score:.3f} at k={best_k_val}")
    return best_k_val

def label_with_groq(concept, cluster_id, reps, groq_client):
    passages = "\n\n".join(
        f"[{i+1}] school={p['school']} | relation={p['relation']}:\n{p['evidence'][:250]}"
        for i,p in enumerate(reps)
    )
    prompt = f"""You are a specialist in Indian philosophy.

I clustered all corpus occurrences of "{concept}". 
Below are {len(reps)} passages from ONE cluster.

Give a SHORT label (4-8 words) for the PHILOSOPHICAL SENSE of "{concept}" in these passages.
Be specific about tradition and meaning:
- "maya as Vedic magical creative power"
- "maya as cosmic illusion in Advaita after Buddhism"
- "karma as Vedic ritual sacrificial action"
- "karma as intentional action in Buddhist ethics"
- "dharma as cosmic order in Vedas"
- "dharma as phenomena in Pali Buddhism"

Passages:
{passages}

Reply with ONLY the short label."""
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=25, temperature=0.1)
        return r.choices[0].message.content.strip().strip('"\'')
    except Exception as e:
        print(f"    Groq error: {e}")
        return f"{concept}_cluster_{cluster_id}"

def cluster_concept(concept, occurrences, groq_client, max_k=MAX_K):
    n = len(occurrences)
    print(f"    Building features for {n} occurrences...")
    embeddings = make_features(occurrences)

    k = best_k(embeddings, n, max_k=max_k)
    print(f"    k={k}")

    if k == 1:
        for o in occurrences:
            o["cluster_id"] = 0
            o["sense_label"] = f"{concept} (undifferentiated in this corpus)"
        return occurrences, {0: f"{concept} (undifferentiated in this corpus)"}

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(embeddings)

    # Write cluster assignments
    for i, o in enumerate(occurrences):
        o["cluster_id"] = int(km_labels[i])

    cluster_labels = {}
    for cid in range(k):
        cluster_occs  = [o for i,o in enumerate(occurrences) if km_labels[i]==cid]
        cluster_embs  = embeddings[km_labels==cid]
        centroid      = cluster_embs.mean(axis=0)
        dists         = np.linalg.norm(cluster_embs - centroid, axis=1)
        rep_idx       = np.argsort(dists)[:REPRESENTATIVE_N]
        reps          = [cluster_occs[i] for i in rep_idx]

        schools = Counter(o["school"] for o in cluster_occs)
        top_s   = ", ".join(f"{s}({n})" for s,n in schools.most_common(3))
        print(f"\n    Cluster {cid}: {len(cluster_occs)} occ | {top_s}")
        for r in reps[:2]:
            print(f"      [{r['school']}|{r['relation']}] "
                  f"{r['evidence'][:90].replace(chr(10),' ')}...")

        if groq_client:
            label = label_with_groq(concept, cid, reps, groq_client)
            time.sleep(0.4)
        else:
            label = f"{concept}_cluster_{cid}_schools_{top_s[:30]}"
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
    print("  Step 2 v3: Clustering (school+relation+text features)")
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
                print("  groq missing — using cluster_N labels")
        else:
            print("  No GROQ_API_KEY — using cluster_N labels\n")

    if args.concept:
        to_process = [args.concept]
    else:
        to_process = [c for c,d in concepts_data.items()
                      if d["count"] >= args.min_occurrences]
    print(f"  Processing {len(to_process)} concepts\n")

    all_labels = {}
    for concept in to_process:
        if concept not in concepts_data:
            print(f"  SKIP '{concept}'"); continue
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
                    if o.get("cluster_id")==cid)
            print(f"    [{cid}] {label}  ({n} occ)")

    data["concepts"] = concepts_data
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output}")

if __name__ == "__main__":
    main()
