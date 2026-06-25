"""
notebooks/darshana_diffusion.py

Practical Application: Philosophical Concept Diffusion across Hindu/Buddhist/Jain Traditions
=============================================================================================

Scenario
--------
A new syncretic concept enters the darshana graph — for example, a neo-Advaita
teaching that blends Vedantic atman-brahman identity with Buddhist sunyata.

Question (directly from Krishnan 2025 Ch5):
  Which tradition acts as the structural bridge that spreads this concept furthest?
  Does seeding high-betweenness philosophical nodes produce faster diffusion
  than seeding the most popular (degree-central) schools?

Data
----
Downloads darshana_graph.jsonl from HuggingFace (joyboseroy/darshana-graph).
Schema: concept_a, concept_b, relation, school, confidence, evidence_quote,
        source_text, tagged_from_file

Network construction
--------------------
Nodes = philosophical schools (advaita, madhyamaka, theravada, jainism, etc.)
Edges = co-occurrence of concepts across schools (an edge exists between
        school A and school B if a shared concept appears in both traditions).
Edge weight = number of shared concepts (stronger conceptual bridge = higher weight).

This is the same co-presence logic used in the digital-buddhism repo,
applied to philosophical schools rather than subreddits.

Usage
-----
# Full run (downloads ~10 MB from HuggingFace)
python notebooks/darshana_diffusion.py

# Limit rows for fast testing
python notebooks/darshana_diffusion.py --max_rows 5000

# Seed a specific school
python notebooks/darshana_diffusion.py --seed_school advaita

# Compare all seeding strategies
python notebooks/darshana_diffusion.py --compare
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.belief_agent import BeliefAgent, STAGE_LABELS
from network.generator import compute_centralities

# ---------------------------------------------------------------------------
# 1. Load data
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


def load_darshana_edges(max_rows: Optional[int] = None) -> List[dict]:
    """
    Download darshana_graph.jsonl from HuggingFace and return as list of dicts.
    Falls back to a small synthetic sample if network is unavailable.
    """
    try:
        import urllib.request
        print(f"  Downloading darshana_graph.jsonl from HuggingFace...")
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


def _synthetic_sample() -> List[dict]:
    """
    Fallback: small synthetic dataset matching the real schema.
    Covers the same schools and relation types as the actual darshana-graph.
    Useful for offline testing or CI.
    """
    relations = [
        "IS_IDENTICAL_TO", "NEGATES", "SUBSUMES", "DERIVES_FROM",
        "IS_COMPATIBLE_WITH", "CRITIQUES", "PARALLELS", "RESPONDS_TO",
    ]
    concepts_per_school = {
        "advaita":          ["atman", "brahman", "maya", "jiva", "moksha", "nirguna"],
        "madhyamaka":       ["sunyata", "pratityasamutpada", "madhyama", "nirvana", "dharma"],
        "theravada":        ["anatta", "anicca", "dukkha", "nibbana", "sila", "panna"],
        "vishishtadvaita":  ["brahman", "jiva", "jagat", "bhakti", "moksha"],
        "dvaita":           ["vishnu", "jiva", "maya", "bhakti", "moksha"],
        "yogacara":         ["alayavijnana", "vijnapti", "dharma", "nirvana"],
        "jainism":          ["jiva", "karma", "moksha", "ahimsa", "anekantavada"],
        "nyaya":            ["pramana", "anumana", "pratyaksha", "God", "tarka"],
        "vaisheshika":      ["dravya", "guna", "karma", "samanya", "vishesha"],
        "samkhya":          ["purusha", "prakriti", "guna", "viveka", "moksha"],
        "yoga":             ["purusha", "prakriti", "samadhi", "chitta", "vritti"],
        "vajrayana":        ["sunyata", "tantra", "rigpa", "buddha_nature", "mandala"],
        "zen":              ["sunyata", "buddha_nature", "mu", "zazen", "kensho"],
        "carvaka":          ["materialism", "perception", "lokayata"],
        "jainism":          ["jiva", "karma", "anekantavada", "syadvada", "ahimsa"],
    }

    rows = []
    edge_id = 0
    schools = list(concepts_per_school.keys())
    for i, school_a in enumerate(schools):
        for school_b in schools[i+1:]:
            shared = set(concepts_per_school[school_a]) & set(concepts_per_school[school_b])
            for concept in shared:
                for _ in range(random.randint(1, 3)):
                    rows.append({
                        "edge_id": f"edge_{edge_id:06d}",
                        "concept_a": concept,
                        "concept_b": random.choice(
                            [c for c in concepts_per_school[school_b] if c != concept]
                            or [concept]
                        ),
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

def build_school_network(edges: List[dict]) -> Tuple[nx.Graph, Dict]:
    """
    Nodes = schools. Edge weight = number of shared concepts between schools.
    Also returns concept_to_schools mapping for seeding by concept.
    """
    # Map concepts to schools
    concept_schools = defaultdict(set)
    for row in edges:
        school = row.get("school", "").strip().lower().replace(" ", "_")
        if not school:
            continue
        concept_schools[row["concept_a"]].add(school)
        concept_schools[row["concept_b"]].add(school)

    # Build co-occurrence counts
    school_pair_weight = defaultdict(int)
    for concept, schools in concept_schools.items():
        school_list = sorted(schools)
        for i, sa in enumerate(school_list):
            for sb in school_list[i+1:]:
                school_pair_weight[(sa, sb)] += 1

    G = nx.Graph()
    for (sa, sb), weight in school_pair_weight.items():
        if weight >= 2:  # minimum edge threshold
            G.add_edge(sa, sb, weight=weight)

    # Ensure all known schools appear as nodes
    for school in KNOWN_SCHOOLS:
        if school not in G:
            G.add_node(school)

    print(f"\n  School network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, concept_schools


# ---------------------------------------------------------------------------
# 3. Run diffusion on the real school graph
# ---------------------------------------------------------------------------

def run_darshana_diffusion(
    G: nx.Graph,
    strategy: str = "betweenness",
    seed_school: Optional[str] = None,
    steps: int = 20,
    cost: float = 0.25,
    seed: int = 42,
) -> Tuple[Dict[str, BeliefAgent], List[float], List[Dict]]:
    """
    Run diffusion on the school graph.
    Returns agents dict, adoption_history, stage_history.
    """
    random.seed(seed)
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Initialise agents — school persona based on tradition
    agents: Dict[str, BeliefAgent] = {}
    for school in nodes:
        tradition = TRADITION_MAP.get(school, "Heterodox")
        # Buddhist schools are more syncretic (higher openness)
        persona = "champion" if tradition == "Buddhist" else \
                  "skeptic" if tradition == "Heterodox" else "neutral"
        agents[school] = BeliefAgent(
            agent_id=node_idx[school],
            persona=persona,
        )

    # Seed nodes
    if seed_school:
        seeds = [seed_school] if seed_school in agents else []
    else:
        centralities = compute_centralities(G)
        scores = centralities.get(strategy, {})
        k = max(1, int(num_nodes * 0.15))
        seeds = sorted([n for n in G.nodes()], key=lambda n: scores.get(n, 0), reverse=True)[:k]

    print(f"\n  Strategy: {strategy} | Seeds: {seeds}")
    for s in seeds:
        if s in agents:
            agents[s].belief = 0.90
            agents[s].stage = 2
            agents[s].behavior = 1

    adoption_history = []
    stage_history = []

    for step in range(steps):
        # Message passing over weighted edges
        for school in G.nodes():
            for neighbor in G.neighbors(school):
                weight = G[school][neighbor].get("weight", 1)
                # Weight scales influence: stronger conceptual bridge = more persuasion
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
        habit_count = stage_counts["Habit"]
        print(f"  Step {step+1:02d} | {bar:<25} {adoption_rate:.0%} | "
              f"Habit:{habit_count:2d} | "
              f"Aware:{stage_counts['Aware']:2d}")

    return agents, adoption_history, stage_history


# ---------------------------------------------------------------------------
# 4. Visualize
# ---------------------------------------------------------------------------

def plot_darshana_network(
    G: nx.Graph,
    agents: Dict[str, BeliefAgent],
    seeds: List[str],
    output_dir: str = "/mnt/user-data/outputs",
    filename: str = "darshana_diffusion_network.png",
):
    """Network plot coloured by tradition, sized by final belief."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    pos = nx.spring_layout(G, seed=42, weight="weight", k=1.2)

    # Edge widths proportional to shared concept count
    max_w = max((G[u][v].get("weight", 1) for u, v in G.edges()), default=1)
    edge_widths = [G[u][v].get("weight", 1) / max_w * 3 for u, v in G.edges()]

    nx.draw_networkx_edges(
        G, pos, ax=ax, alpha=0.2, edge_color="#5c6370", width=edge_widths
    )

    for school in G.nodes():
        tradition = TRADITION_MAP.get(school, "Heterodox")
        color = TRADITION_COLORS.get(tradition, "#adb5bd")
        belief = agents[school].belief if school in agents else 0.0
        # Size = belief strength
        size = 200 + belief * 600
        border = "#FFD700" if school in seeds else "#2d3142"
        nx.draw_networkx_nodes(
            G, pos, nodelist=[school], ax=ax,
            node_color=[color], node_size=[size],
            edgecolors=[border], linewidths=2.0,
        )

    # Labels only for nodes with some belief
    label_nodes = {s: s.replace("_", "\n") for s in G.nodes()
                   if school in agents and agents[s].belief > 0.1}
    nx.draw_networkx_labels(
        G, pos, labels={s: s.replace("_", "\n") for s in G.nodes()},
        ax=ax, font_size=7, font_color="#e9ecef",
    )

    # Legend
    patches = [
        mpatches.Patch(color=c, label=t)
        for t, c in TRADITION_COLORS.items()
    ]
    patches.append(mpatches.Patch(color="#FFD700", label="Seed school"))
    ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#2d3142",
              labelcolor="#f8f9fa", fontsize=9, loc="lower left")
    ax.set_title(
        "Philosophical Concept Diffusion across Darshana Traditions\n"
        "(Node size = final belief strength | Gold border = seed school)",
        color="#f8f9fa", fontsize=13, pad=12,
    )
    ax.axis("off")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved network plot: {path}")
    return path


def plot_strategy_comparison(
    results: Dict[str, List[float]],
    steps: int,
    output_dir: str = "/mnt/user-data/outputs",
    filename: str = "darshana_strategy_comparison.png",
):
    """Line chart comparing seeding strategies on the darshana network."""
    COLORS = {
        "betweenness": "#E63946",
        "percolation": "#F4A261",
        "degree":      "#2A9D8F",
        "random":      "#ADB5BD",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for strategy, history in results.items():
        ax.plot(range(1, steps + 1), [h * 100 for h in history],
                label=strategy.capitalize(),
                color=COLORS.get(strategy, "#fff"),
                linewidth=2.5, marker="o", markersize=3)

    ax.set_xlabel("Timestep", color="#adb5bd", fontsize=11)
    ax.set_ylabel("Adoption Rate (%)", color="#adb5bd", fontsize=11)
    ax.set_title(
        "Concept Diffusion by Seeding Strategy — Darshana Graph\n"
        "(Real joyboseroy/darshana-graph | Krishnan 2025 Ch5 methodology)",
        color="#f8f9fa", fontsize=12, pad=12,
    )
    ax.tick_params(colors="#adb5bd")
    ax.spines[:].set_color("#2d3142")
    ax.grid(color="#2d3142", linewidth=0.5, linestyle="--")
    ax.legend(facecolor="#1a1d27", edgecolor="#2d3142",
              labelcolor="#f8f9fa", fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison chart: {path}")
    return path


def print_summary(G: nx.Graph, agents: Dict[str, BeliefAgent]):
    """Print a ranked table of schools by final belief."""
    centralities = compute_centralities(G)
    bc = centralities["betweenness"]

    print("\n" + "="*65)
    print(f"  {'School':<22} {'Tradition':<12} {'Belief':>7} {'Stage':<10} {'Betw.Cent':>9}")
    print("="*65)
    ranked = sorted(agents.items(), key=lambda x: -x[1].belief)
    for school, agent in ranked:
        tradition = TRADITION_MAP.get(school, "?")
        stage = STAGE_LABELS[agent.stage]
        bval = bc.get(school, 0.0)
        bar = "█" * int(agent.belief * 10)
        print(f"  {school:<22} {tradition:<12} {bar:<10} {stage:<10} {bval:>9.3f}")
    print("="*65)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Darshana Graph Diffusion — practical example for agentic-diffusion-sim"
    )
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Limit JSONL rows loaded (None = all 28k)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cost", type=float, default=0.25)
    parser.add_argument("--seed_school", type=str, default=None,
                        help="Manually seed a specific school (e.g. advaita)")
    parser.add_argument("--strategy", default="betweenness",
                        choices=["betweenness", "percolation", "degree", "random"])
    parser.add_argument("--compare", action="store_true",
                        help="Compare all 4 strategies")
    parser.add_argument("--output_dir", default="/mnt/user-data/outputs")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Darshana Diffusion — Philosophical Concept Spread Simulator")
    print("  Data: joyboseroy/darshana-graph (28,322 edges)")
    print("  Method: Krishnan (2025) Ch5 ABM + weighted co-presence network")
    print("="*65)

    # Load and build
    edges = load_darshana_edges(max_rows=args.max_rows)
    G, concept_schools = build_school_network(edges)

    if args.compare:
        comparison_results = {}
        for strategy in ["betweenness", "percolation", "degree", "random"]:
            print(f"\n--- Strategy: {strategy} ---")
            agents, history, _ = run_darshana_diffusion(
                G, strategy=strategy, steps=args.steps, cost=args.cost
            )
            comparison_results[strategy] = history

        plot_strategy_comparison(comparison_results, args.steps, args.output_dir)
        print("\nFinal adoption rates:")
        for s, h in sorted(comparison_results.items(), key=lambda x: -x[1][-1]):
            print(f"  {s:12s}: {h[-1]:.0%}")
    else:
        agents, history, stage_history = run_darshana_diffusion(
            G,
            strategy=args.strategy,
            seed_school=args.seed_school,
            steps=args.steps,
            cost=args.cost,
        )

        # Determine actual seeds for plot
        centralities = compute_centralities(G)
        scores = centralities.get(args.strategy, {})
        k = max(1, int(G.number_of_nodes() * 0.15))
        seeds = (
            [args.seed_school] if args.seed_school
            else sorted(G.nodes(), key=lambda n: scores.get(n, 0), reverse=True)[:k]
        )

        print_summary(G, agents)
        plot_darshana_network(G, agents, seeds, args.output_dir)


if __name__ == "__main__":
    main()
