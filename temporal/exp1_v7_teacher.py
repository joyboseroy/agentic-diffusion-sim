"""
Experiment 1 v7: Structural Homologues — teacher's vocabulary included.

Additions from reviewer meditation teacher:
- nirodha, nivarana, asrava, bodhyanga, dhyana, maitri, karuna, avarana
- Pali-Sanskrit aliases: kamma/karma, dhamma/dharma, jhana/dhyana,
  metta/maitri, avijja/avidya, dosa/dvesha, bojjhanga/bodhyanga
- citta, vedana, cetana, tanha, upadana, moha, lobha, vipassana

These are now in temporal_source_layer.json after running add_teacher_terms.py.
"""

import json, urllib.request, unicodedata, argparse
from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/joyboseroy/darshana-graph"
    "/resolve/main/darshana_graph.jsonl"
)
DIRTY_SCHOOLS = {"dvaitadvaita","jain_digambara","jain_common"}

TRADITION_GROUP = {
    "Vedic":                     "Vedic",
    "Vedic/early Upanishadic":   "Vedic",
    "Cross-tradition note":       None,
    "Jain":                      "Jain",
    "Gaudiya Vaishnavism":       "Vaishnava",
    "Theravada Buddhist":        "Buddhist/Theravada",
    "Madhyamaka Buddhist":       "Buddhist/Madhyamaka",
    "Yogacara Buddhist":         "Buddhist/Yogacara",
    "Mahayana Buddhist":         "Buddhist/Mahayana",
    "Vajrayana Buddhist":        "Buddhist/Vajrayana",
    "Chan / Zen Buddhist":       "Buddhist/Zen",
    "Samkhya":                   "Hindu/Samkhya",
    "Yoga":                      "Hindu/Yoga",
    "Advaita Vedanta":           "Hindu/Advaita",
    "Pre-Shankara Advaita":      "Hindu/Advaita",
    "Sufi Islam":                "Sufi",
    "Sikh":                      "Sikh",
    "Sant tradition":            "Sant",
}

ALIASES = {
    # Compound temporal layer names -> graph node names
    "sunyata as philosophical system":         "sunyata",
    "pratityasamutpada as sunyata":            "pratityasamutpada",
    "svabhava critique":                       "svabhava",
    "two truths doctrine systematic":          "two truths",
    "shunyata of shunyata":                    "sunyata",
    "critique of atman":                       "anatta",
    "dharma as dhamma":                        "dharma",
    "karma as intentional action":             "karma",
    "karma as ethical causation":              "karma",
    "karma as substance":                      "karma",
    "buddha nature":                           "tathagatagarbha",
    "bodhisattva ideal":                       "bodhisattva ideal",
    "moksha as liberation from karma-matter":  "moksha",
    "moksha as liberation from rebirth":       "moksha",
    "brahman as absolute":                     "brahman",
    "atman as self":                           "atman",
    "ahimsa in vedic tradition":               "ahimsa",
    "maya as systematic advaita term":         "maya",
    "maya as systematic metaphysics":          "maya",
    "jiva-brahman identity":                   "jiva",
    "nirguna brahman systematic":              "brahman",
    "lila as ultimate reality":                "lila",
    "saguna brahman as ultimate":              "brahman",
    "shabda as inner sound":                   "shabda",
    "nirguna bhakti":                          "bhakti",
    # Pali -> Sanskrit (graph uses Sanskrit forms primarily)
    "kamma":        "karma",
    "dhamma":       "dharma",
    "avijja":       "avidya",
    "metta":        "compassion",   # graph has 'compassion' not 'maitri'
    "karuna":       "compassion",   # graph uses 'compassion' for both
    "bojjhanga":    "bodhyanga",
    "dosa":         "dvesha",
    "jhana":        "dhyana",
    "sacca":        "satya",
    # Teacher's terms -> graph node names
    "nirodha":      "nirodha",      # confirmed 24 edges
    "nivarana":     "hindrances",   # confirmed 15 edges
    "asrava":       "asrava",       # confirmed 19 edges
    "bodhyanga":    "bodhyanga",    # via bojjhanga, 3 edges (low — may not qualify)
    "dhyana":       "dhyana",       # confirmed 26 edges
    "avarana":      "avarana",      # confirmed 1 edge (low)
    "tanha":        "tanha",        # confirmed 54 edges
    "vedana":       "vedana",       # confirmed 51 edges
    "cetana":       "cetana",       # confirmed 4 edges
    "upadana":      "upadana",      # confirmed 13 edges
    "citta":        "citta",        # confirmed 35 edges
    "moha":         "moha",         # confirmed 22 edges
    "lobha":        "lobha",        # confirmed 10 edges
    "vipassana":    "vipassana",    # confirmed 1 edge (low)
}

KNOWN_PAIRS = [
    ("sunyata",          "maya",          "Buddhist/Madhyamaka", "Vedic"),
    ("sunyata",          "brahman",       "Buddhist/Madhyamaka", "Vedic"),
    ("nibbana",          "moksha",        "Buddhist/Theravada",  "Jain"),
    ("anatta",           "atman",         "Buddhist/Theravada",  "Vedic"),
    ("pratityasamutpada","karma",         "Buddhist/Madhyamaka", "Vedic"),
    ("dhyana",           "samadhi",       "Buddhist/Theravada",  "Hindu/Yoga"),
    ("dhyana",           "turiya",        "Buddhist/Theravada",  "Hindu/Advaita"),
    ("nirodha",          "moksha",        "Buddhist/Theravada",  "Jain"),
    ("nirodha",          "nirvana",       "Buddhist/Theravada",  "Buddhist/Theravada"),
    ("tanha",            "karma",         "Buddhist/Theravada",  "Vedic"),
    ("avidya",           "maya",          "Hindu/Advaita",       "Vedic"),
    ("purusha",          "jiva",          "Hindu/Samkhya",       "Jain"),
    ("prakriti",         "maya",          "Hindu/Samkhya",       "Vedic"),
    ("dukkha",           "samsara",       "Buddhist/Theravada",  "Vedic"),
    ("sila",             "tapas",         "Buddhist/Theravada",  "Jain"),
    ("karuna",           "ahimsa",        "Buddhist/Theravada",  "Jain"),
    ("moha",             "avidya",        "Buddhist/Theravada",  "Hindu/Advaita"),
    ("vedana",           "tanha",         "Buddhist/Theravada",  "Buddhist/Theravada"),
]

def normalise(s):
    s = unicodedata.normalize("NFC", s.strip().lower())
    for a,b in [("ṛ","r"),("ā","a"),("ī","i"),("ū","u"),("ś","s"),
                ("ṣ","s"),("ṭ","t"),("ḍ","d"),("ṇ","n"),("ñ","n"),
                ("ḥ","h"),("ṃ","m")]:
        s = s.replace(a,b)
    return s

def load_temporal_labels(path="temporal_source_layer.json"):
    temporal = {}
    with open(path) as f:
        sources = json.load(f)
    for source in sources:
        trad_raw = source.get("tradition","")
        group    = TRADITION_GROUP.get(trad_raw)
        if group is None: continue
        date = source.get("date_early_ce", 9999)
        for entry in source.get("concepts_introduced",[]):
            raw   = entry["concept"].strip().lower()
            alias = ALIASES.get(raw)
            c     = normalise(alias if alias else raw)
            if c not in temporal or date < temporal[c]["date"]:
                temporal[c] = {
                    "date":      date,
                    "tradition": group,
                    "trad_raw":  trad_raw,
                    "text":      source.get("text",""),
                    "note":      entry.get("note",""),
                }
    print(f"  Temporal labels: {len(temporal)} resolved concepts")
    by_t = defaultdict(list)
    for c,info in temporal.items():
        by_t[info["tradition"]].append(c)
    for t,cs in sorted(by_t.items(), key=lambda x:-len(x[1])):
        print(f"    {t:<28}: {len(cs):>3}  — {', '.join(cs[:6])}")
    return temporal

def load_graph(max_rows=None):
    print("\n  Downloading darshana_graph.jsonl...")
    rows = []
    with urllib.request.urlopen(HUGGINGFACE_URL, timeout=60) as f:
        for i,line in enumerate(f):
            if max_rows and i >= max_rows: break
            rows.append(json.loads(line.decode()))
    rows = [r for r in rows if r.get("school","") not in DIRTY_SCHOOLS]
    G = nx.MultiGraph()
    rels = set()
    for row in rows:
        ca = normalise(row["concept_a"])
        cb = normalise(row["concept_b"])
        rel = row.get("relation","UNKNOWN")
        if ca and cb and ca != cb:
            G.add_edge(ca, cb, relation=rel)
            rels.add(rel)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G, sorted(rels)

def compute_features(G, temporal, relation_types, bc, min_degree=3):
    trad_list = sorted(set(v["tradition"] for v in temporal.values()))
    n_rel  = len(relation_types)
    n_trad = len(trad_list)
    rel_idx  = {r:i for i,r in enumerate(relation_types)}
    trad_idx = {t:i for i,t in enumerate(trad_list)}
    deg = dict(G.degree())
    max_deg = max(deg.values()) if deg else 1
    vectors = {}
    missing = []

    for node, info in temporal.items():
        if node not in G:
            missing.append(node)
            continue
        if deg.get(node,0) < min_degree:
            continue
        vec = np.zeros(3 + n_rel + n_trad)
        vec[0] = bc.get(node, 0.0)
        vec[1] = deg.get(node,0) / max_deg
        try:    vec[2] = nx.clustering(G, node)
        except: vec[2] = 0.0
        for _,_,d in G.edges(node, data=True):
            rel = d.get("relation","UNKNOWN")
            if rel in rel_idx:
                vec[3+rel_idx[rel]] += 1
        s = vec[3:3+n_rel].sum()
        if s > 0: vec[3:3+n_rel] /= s
        for nb in G.neighbors(node):
            t = temporal.get(nb,{}).get("tradition","")
            if t in trad_idx:
                vec[3+n_rel+trad_idx[t]] += 1
        s = vec[3+n_rel:].sum()
        if s > 0: vec[3+n_rel:] /= s
        vectors[node] = vec

    print(f"\n  Concepts with features (degree>={min_degree}): {len(vectors)}")
    by_t = defaultdict(list)
    for n in vectors:
        by_t[temporal[n]["tradition"]].append(n)
    for t,ns in sorted(by_t.items(), key=lambda x:-len(x[1])):
        print(f"    {t:<28}: {len(ns):>3}  — {', '.join(ns[:7])}")
    if missing[:6]:
        print(f"\n  Not in graph: {missing[:8]}")
    return vectors, trad_list, by_t

def find_homologues(vectors, temporal, by_trad, top_n=60):
    trad_list = sorted(by_trad.keys())
    results = []
    for i in range(len(trad_list)):
        for j in range(i+1, len(trad_list)):
            ta,tb = trad_list[i], trad_list[j]
            na = [n for n in by_trad[ta] if n in vectors]
            nb = [n for n in by_trad[tb] if n in vectors]
            if not na or not nb: continue
            va = np.array([vectors[n] for n in na])
            vb = np.array([vectors[n] for n in nb])
            sim = cosine_similarity(va, vb)
            for ii,na_ in enumerate(na):
                for jj,nb_ in enumerate(nb):
                    results.append((float(sim[ii,jj]), na_, ta, nb_, tb))
    results.sort(reverse=True)
    return results[:top_n*5]

def validate(results, temporal, known_pairs):
    idx = {}
    for rank,(sim,ca,ta,cb,tb) in enumerate(results,1):
        idx[(ca,cb)] = (rank,sim,ta,tb)
        idx[(cb,ca)] = (rank,sim,tb,ta)
    print(f"\n{'='*75}")
    print("  VALIDATION: Known scholarly correspondences")
    print(f"{'='*75}")
    found = 0
    for ca,cb,ea,eb in known_pairs:
        can,cbn = normalise(ca), normalise(cb)
        if (can,cbn) in idx:
            r,s,ta,tb = idx[(can,cbn)]
            found += 1
            print(f"  FOUND  {ca:<24} ~ {cb:<24}  rank {r:>4}  sim {s:.4f}")
        else:
            ta_ = temporal.get(can,{}).get("tradition","?")
            tb_ = temporal.get(cbn,{}).get("tradition","?")
            same = (ta_==tb_ and ta_!="?")
            note = "same tradition" if same else "[not in graph]"
            print(f"  MISS   {ca:<24} ~ {cb:<24}  [{ea}|{eb}]  {note}")
    print(f"\n  Recovery: {found}/{len(known_pairs)}")
    return found

def print_results(results, temporal, known_pairs, top_n=50):
    known_set = set()
    for ca,cb,_,_ in known_pairs:
        known_set.add((normalise(ca),normalise(cb)))
        known_set.add((normalise(cb),normalise(ca)))
    print(f"\n{'='*100}")
    print("  TOP CROSS-TRADITION STRUCTURAL HOMOLOGUES")
    print(f"{'='*100}")
    print(f"  {'#':<4} {'Concept A':<22} {'Tradition A':<26} "
          f"{'Concept B':<22} {'Tradition B':<26} {'Sim':>7}  Note")
    print(f"  {'-'*4} {'-'*22} {'-'*26} {'-'*22} {'-'*26} {'-'*7}  {'-'*5}")
    shown = 0
    for rank,(sim,ca,ta,cb,tb) in enumerate(results,1):
        if shown >= top_n: break
        flag = "KNOWN" if (ca,cb) in known_set or (cb,ca) in known_set else "new"
        print(f"  {rank:<4} {ca:<22} {ta:<26} {cb:<22} {tb:<26} {sim:>7.4f}  {flag}")
        shown += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows",   type=int, default=None)
    parser.add_argument("--top_n",      type=int, default=50)
    parser.add_argument("--min_degree", type=int, default=3)
    parser.add_argument("--output",     type=str, default="homologues_v7.json")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Experiment 1 v7: Structural Homologues")
    print("  Teacher's vocabulary added: nirodha, dhyana, karuna,")
    print("  tanha, vedana, moha + Pali-Sanskrit aliases")
    print("="*65 + "\n")

    temporal = load_temporal_labels("temporal_source_layer.json")
    G, rels  = load_graph(args.max_rows)
    print("\n  Computing betweenness centrality...")
    bc = nx.betweenness_centrality(G, normalized=True)
    vectors, trad_list, by_trad = compute_features(
        G, temporal, rels, bc, min_degree=args.min_degree)
    print("\n  Computing cross-tradition similarities...")
    results = find_homologues(vectors, temporal, by_trad, top_n=args.top_n)
    validate(results, temporal, KNOWN_PAIRS)
    print_results(results, temporal, KNOWN_PAIRS, top_n=args.top_n)
    out = [{"rank":r+1,"sim":s,"concept_a":ca,"tradition_a":ta,
            "concept_b":cb,"tradition_b":tb}
           for r,(s,ca,ta,cb,tb) in enumerate(results[:200])]
    with open(args.output,"w") as f: json.dump(out,f,indent=2)
    print(f"\n  Saved: {args.output}")

if __name__ == "__main__":
    main()
