"""
darshana_diffusion_standalone.py

Practical Application: Philosophical Concept Diffusion across Hindu/Buddhist/Jain Traditions
Based on: Krishnan (2025) Ch5 ABM methodology, applied to joyboseroy/darshana-graph

STANDALONE — no local imports required. Single file, runs anywhere.

Usage:
  python3 darshana_diffusion_standalone.py --seed_school advaita --steps 25
  python3 darshana_diffusion_standalone.py --compare
  python3 darshana_diffusion_standalone.py --max_rows 5000 --compare
"""

import argparse
import os
import sys
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Inlined: BeliefAgent
# ---------------------------------------------------------------------------

STAGE_LABELS = {0: "Unaware", 1: "Aware", 2: "Action", 3: "Habit"}

class BeliefAgent:
    def __init__(self, agent_id, persona="neutral", intrinsic_benefit=None):
        self.id = agent_id
        self.persona = persona
        self.intrinsic_benefit = intrinsic_benefit or random.uniform(0.1, 0.9)
        self.belief = 0.0
        self.stage = 0
        self.behavior = -1
        self.memory = []
        persona_params = {
            "champion": {"openness": 0.9, "conformity": 0.7},
            "skeptic":  {"openness": 0.2, "conformity": 0.2},
            "neutral":  {"openness": 0.5, "conformity": 0.5},
            "laggard":  {"openness": 0.1, "conformity": 0.4},
        }
        p = persona_params.get(persona, persona_params["neutral"])
        self.openness = p["openness"]
        self.conformity = p["conformity"]

    def receive_message(self, sender_id, sender_belief, sender_stage):
        self.memory.append({"sender": sender_id, "belief": sender_belief, "stage": sender_stage})
        if len(self.memory) > 5:
            self.memory.pop(0)

    def update(self, cost=0.3):
        if not self.memory:
            return
        avg_neighbor_belief = sum(m["belief"] for m in self.memory) / len(self.memory)
        peer_pressure = self.conformity * (avg_neighbor_belief - self.belief)
        net_utility = (self.intrinsic_benefit - cost) + peer_pressure
        delta = self.openness * net_utility * 0.2
        self.belief = max(0.0, min(1.0, self.belief + delta))
        if self.belief > 0.75 and self.stage < 3:
            self.stage = 3
        elif self.belief > 0.5 and self.stage < 2:
            self.stage = 2
        elif self.belief > 0.25 and self.stage < 1:
            self.stage = 1
        self.behavior = 1 if self.belief > 0.5 else -1

# ---------------------------------------------------------------------------
# Inlined: centrality helpers
# ---------------------------------------------------------------------------

def compute_centralities(G):
    bc = nx.betweenness_centrality(G)
    dc = nx.degree_centrality(G)
    # Percolation approx: betweenness * (1 - clustering)
    cc = nx.clustering(G)
    perc_raw = {n: bc[n] * (1 - cc.get(n, 0)) for n in G.nodes()}
    max_p = max(perc_raw.values()) if perc_raw else 1.0
    perc = {k: v / max_p for k, v in perc_raw.items()} if max_p > 0 else perc_raw
    return {
        "betweenness": bc,
        "degree": dc,
        "percolation": perc,
        "random": {n: random.random() for n in G.nodes()},
    }

# ---------------------------------------------------------------------------
# Data and network constants
# ---------------------------------------------------------------------------

HUGGINGFACE_URL = (
    "https://huggingface.co/datasets/joyboseroy/darshana-graph"
    "/resolve/main/darshana_graph.jsonl"
)

KNOWN_SCHOOLS = [
    "advaita", "vishishtadvaita", "dvaita", "nyaya", "vaisheshika",
    "samkhya", "yoga", "mimamsa", "purva_mimamsa",
    "madhyamaka", "yogacara", "theravada", "vajrayana", "zen",
    "jainism", "carvaka", "ajivika",
]

TRADITION_MAP = {
    "advaita": "Hindu", "vishishtadvaita": "Hindu", "dvaita": "Hindu",
    "nyaya": "Hindu", "vaisheshika": "Hindu", "samkhya": "Hindu",
    "yoga": "Hindu", "mimamsa": "Hindu", "purva_mimamsa": "Hindu",
    "madhyamaka": "Buddhist", "yogacara": "Buddhist",
    "theravada": "Buddhist", "vajrayana": "Buddhist", "zen": "Buddhist",
    "jainism": "Jain", "carvaka": "Heterodox", "ajivika": "Heterodox",
}

TRADITION_COLORS = {
    "Hindu": "#F4A261",
    "Buddhist": "#2A9D8F",
    "Jain": "#E63946",
    "Heterodox": "#ADB5BD",
}

STRATEGY_COLORS = {
    "betweenness": "#E63946",
    "percolation": "#F4A261",
    "degree": "#2A9D8F",
    "random": "#ADB5BD",
}

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_darshana_edges(max_rows=None):
    try:
        import urllib.request
        print("  Downloading darshana_graph.jsonl from HuggingFace...")
        rows = []
        with urllib.request.urlopen(HUGGINGFACE_URL, timeout=30) as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                rows.append(json.loads(line.decode("utf-8")))
        print(f"  Loaded {len(rows):,} edges from HuggingFace.")
        return rows
    except Exception as e:
        print(f"  HuggingFace unavailable ({e}). Using synthetic sample.")
        return _synthetic_sample()


def _synthetic_sample():
    relations = ["IS_IDENTICAL_TO", "NEGATES", "SUBSUMES", "DERIVES_FROM",
                 "IS_COMPATIBLE_WITH", "CRITIQUES", "PARALLELS"]
    concepts_per_school = {
        "advaita":         ["atman", "brahman", "maya", "jiva", "moksha", "nirguna"],
        "madhyamaka":      ["sunyata", "pratityasamutpada", "nirvana", "dharma", "madhyama"],
        "theravada":       ["anatta", "anicca", "dukkha", "nibbana", "sila", "panna"],
        "vishishtadvaita": ["brahman", "jiva", "jagat", "bhakti", "moksha"],
        "dvaita":          ["vishnu", "jiva", "maya", "bhakti", "moksha"],
        "yogacara":        ["alayavijnana", "vijnapti", "dharma", "nirvana"],
        "jainism":         ["jiva", "karma", "moksha", "ahimsa", "anekantavada"],
        "nyaya":           ["pramana", "anumana", "pratyaksha", "tarka"],
        "vaisheshika":     ["dravya", "guna", "karma", "samanya"],
        "samkhya":         ["purusha", "prakriti", "guna", "viveka", "moksha"],
        "yoga":            ["purusha", "prakriti", "samadhi", "chitta"],
        "vajrayana":       ["sunyata", "tantra", "rigpa", "buddha_nature"],
        "zen":             ["sunyata", "buddha_nature", "mu", "zazen"],
        "carvaka":         ["materialism", "perception", "lokayata"],
        "ajivika":         ["niyati", "materialism", "fatalism"],
    }
    rows = []
    edge_id = 0
    schools = list(concepts_per_school.keys())
    for i, school_a in enumerate(schools):
        for school_b in schools[i+1:]:
            shared = set(concepts_per_school[school_a]) & set(concepts_per_school[school_b])
            for concept in shared:
                for _ in range(random.randint(1, 4)):
                    rows.append({
                        "edge_id": f"edge_{edge_id:06d}",
                        "concept_a": concept,
                        "concept_b": random.choice(list(concepts_per_school[school_b])),
                        "relation": random.choice(relations),
                        "school": school_a,
                        "confidence": random.choice(["high", "medium", "low"]),
                        "evidence_quote": f"Synthetic: {concept} in {school_a}",
                        "source_text": f"{school_a}_texts",
                        "tagged_from_file": f"{school_a}.jsonl",
                    })
                    edge_id += 1
    print(f"  Generated {len(rows):,} synthetic edges across {len(schools)} schools.")
    return rows

# ---------------------------------------------------------------------------
# 2. Build school co-presence network
# ---------------------------------------------------------------------------

def build_school_network(edges):
    concept_schools = defaultdict(set)
    for row in edges:
        school = row.get("school", "").strip().lower().replace(" ", "_")
        if not school:
            continue
        concept_schools[row["concept_a"]].add(school)
        concept_schools[row["concept_b"]].add(school)

    school_pair_weight = defaultdict(int)
    for concept, schools in concept_schools.items():
        school_list = sorted(schools)
        for i, sa in enumerate(school_list):
            for sb in school_list[i+1:]:
                school_pair_weight[(sa, sb)] += 1

    G = nx.Graph()
    for (sa, sb), weight in school_pair_weight.items():
        if weight >= 2:
            G.add_edge(sa, sb, weight=weight)

    for school in KNOWN_SCHOOLS:
        if school not in G:
            G.add_node(school)

    print(f"\n  School network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# ---------------------------------------------------------------------------
# 3. Diffusion simulation
# ---------------------------------------------------------------------------

def run_darshana_diffusion(G, strategy="betweenness", seed_school=None,
                           steps=20, cost=0.25, seed=42):
    random.seed(seed)
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    agents = {}
    for school in nodes:
        tradition = TRADITION_MAP.get(school, "Heterodox")
        persona = ("champion" if tradition == "Buddhist"
                   else "skeptic" if tradition == "Heterodox"
                   else "neutral")
        agents[school] = BeliefAgent(agent_id=node_idx[school], persona=persona)

    centralities = compute_centralities(G)
    if seed_school and seed_school in agents:
        seeds = [seed_school]
    else:
        scores = centralities.get(strategy, {})
        k = max(1, int(num_nodes * 0.15))
        seeds = sorted(G.nodes(), key=lambda n: scores.get(n, 0), reverse=True)[:k]

    print(f"\n  Strategy: {strategy} | Seeds: {seeds}")
    for s in seeds:
        if s in agents:
            agents[s].belief = 0.90
            agents[s].stage = 2
            agents[s].behavior = 1

    adoption_history = []
    stage_history = []

    for step in range(steps):
        for school in G.nodes():
            for neighbor in G.neighbors(school):
                weight = G[school][neighbor].get("weight", 1)
                scaled_belief = agents[school].belief * min(1.0, weight / 10.0)
                agents[neighbor].receive_message(
                    sender_id=node_idx[school],
                    sender_belief=scaled_belief,
                    sender_stage=agents[school].stage,
                )
        for school in G.nodes():
            agents[school].update(cost=cost)

        adopters = sum(1 for a in agents.values() if a.behavior == 1)
        adoption_rate = adopters / num_nodes
        stage_counts = {STAGE_LABELS[s]: 0 for s in range(4)}
        for a in agents.values():
            stage_counts[STAGE_LABELS[a.stage]] += 1

        adoption_history.append(adoption_rate)
        stage_history.append(stage_counts)

        bar = "█" * int(adoption_rate * 25)
        print(f"  Step {step+1:02d} | {bar:<25} {adoption_rate:.0%} | "
              f"Habit:{stage_counts['Habit']:2d} | Aware:{stage_counts['Aware']:2d}")

    return agents, adoption_history, stage_history, seeds, centralities

# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------

def plot_network(G, agents, seeds, output_dir="."):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    pos = nx.spring_layout(G, seed=42, weight="weight", k=1.2)
    max_w = max((G[u][v].get("weight", 1) for u, v in G.edges()), default=1)
    edge_widths = [G[u][v].get("weight", 1) / max_w * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color="#5c6370", width=edge_widths)

    for school in G.nodes():
        tradition = TRADITION_MAP.get(school, "Heterodox")
        color = TRADITION_COLORS.get(tradition, "#adb5bd")
        belief = agents[school].belief if school in agents else 0.0
        size = 200 + belief * 600
        border = "#FFD700" if school in seeds else "#2d3142"
        nx.draw_networkx_nodes(G, pos, nodelist=[school], ax=ax,
                               node_color=[color], node_size=[size],
                               edgecolors=[border], linewidths=2.0)

    nx.draw_networkx_labels(G, pos,
                            labels={s: s.replace("_", "\n") for s in G.nodes()},
                            ax=ax, font_size=7, font_color="#e9ecef")

    patches = [mpatches.Patch(color=c, label=t) for t, c in TRADITION_COLORS.items()]
    patches.append(mpatches.Patch(color="#FFD700", label="Seed school"))
    ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#2d3142",
              labelcolor="#f8f9fa", fontsize=9, loc="lower left")
    ax.set_title(
        "Philosophical Concept Diffusion — Darshana Graph\n"
        "(Node size = final belief | Gold = seed school)",
        color="#f8f9fa", fontsize=13, pad=12)
    ax.axis("off")

    path = os.path.join(output_dir, "darshana_diffusion_network.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path}")


def plot_comparison(results, steps, output_dir="."):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for strategy, history in results.items():
        ax.plot(range(1, steps + 1), [h * 100 for h in history],
                label=strategy.capitalize(),
                color=STRATEGY_COLORS.get(strategy, "#fff"),
                linewidth=2.5, marker="o", markersize=3)

    ax.set_xlabel("Timestep", color="#adb5bd", fontsize=11)
    ax.set_ylabel("Adoption Rate (%)", color="#adb5bd", fontsize=11)
    ax.set_title(
        "Concept Diffusion by Seeding Strategy — Darshana Graph\n"
        "(joyboseroy/darshana-graph | Krishnan 2025 Ch5 methodology)",
        color="#f8f9fa", fontsize=12, pad=12)
    ax.tick_params(colors="#adb5bd")
    ax.spines[:].set_color("#2d3142")
    ax.grid(color="#2d3142", linewidth=0.5, linestyle="--")
    ax.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#f8f9fa", fontsize=10)

    path = os.path.join(output_dir, "darshana_strategy_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_summary(G, agents, centralities):
    bc = centralities["betweenness"]
    print("\n" + "="*68)
    print(f"  {'School':<22} {'Tradition':<12} {'Belief':>10} {'Stage':<10} {'Betw':>6}")
    print("="*68)
    for school, agent in sorted(agents.items(), key=lambda x: -x[1].belief):
        tradition = TRADITION_MAP.get(school, "?")
        stage = STAGE_LABELS[agent.stage]
        bar = "█" * int(agent.belief * 10)
        bval = bc.get(school, 0.0)
        print(f"  {school:<22} {tradition:<12} {bar:<10} {stage:<10} {bval:>6.3f}")
    print("="*68)

# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Darshana Graph Diffusion Simulator")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cost", type=float, default=0.25)
    parser.add_argument("--seed_school", type=str, default=None)
    parser.add_argument("--strategy", default="betweenness",
                        choices=["betweenness", "percolation", "degree", "random"])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    print("\n" + "="*68)
    print("  Darshana Diffusion — Philosophical Concept Spread Simulator")
    print("  Data: joyboseroy/darshana-graph (28,322 edges)")
    print("  Method: Krishnan (2025) Ch5 ABM + weighted co-presence network")
    print("="*68)

    edges = load_darshana_edges(max_rows=args.max_rows)
    G = build_school_network(edges)

    if args.compare:
        comparison_results = {}
        for strategy in ["betweenness", "percolation", "degree", "random"]:
            print(f"\n--- Strategy: {strategy} ---")
            agents, history, _, seeds, centralities = run_darshana_diffusion(
                G, strategy=strategy, steps=args.steps, cost=args.cost)
            comparison_results[strategy] = history

        plot_comparison(comparison_results, args.steps, args.output_dir)
        print("\nFinal adoption rates:")
        for s, h in sorted(comparison_results.items(), key=lambda x: -x[1][-1]):
            print(f"  {s:12s}: {h[-1]:.0%}")
    else:
        agents, history, stage_history, seeds, centralities = run_darshana_diffusion(
            G, strategy=args.strategy, seed_school=args.seed_school,
            steps=args.steps, cost=args.cost)
        print_summary(G, agents, centralities)
        plot_network(G, agents, seeds, args.output_dir)


if __name__ == "__main__":
    main()
