"""
simulation/diffusion.py — Core simulation loop for agentic behavior diffusion.

Each timestep:
  1. Every agent broadcasts its belief to neighbors (message passing)
  2. Every agent updates belief using Krishnan myopic utility rule
  3. Adoption stats are recorded

Supports both math mode (fast) and LLM mode (Groq, slower but narrative).
"""

import networkx as nx
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from agents.belief_agent import BeliefAgent, STAGE_LABELS
from network.generator import build_graph, compute_centralities, get_seed_nodes


PERSONAS = ["champion", "skeptic", "neutral", "laggard"]
PERSONA_WEIGHTS = [0.15, 0.20, 0.50, 0.15]   # realistic enterprise mix


@dataclass
class SimulationResult:
    strategy: str
    topology: str
    steps: int
    adoption_history: List[float] = field(default_factory=list)
    stage_history: List[Dict] = field(default_factory=list)
    final_adopters: int = 0
    seed_nodes: List[int] = field(default_factory=list)
    llm_narratives: List[str] = field(default_factory=list)


def run_simulation(
    num_nodes: int = 60,
    topology: str = "erdos_renyi",
    edge_prob: float = 0.12,
    steps: int = 25,
    strategy: str = "betweenness",
    seed_fraction: float = 0.10,
    cost: float = 0.3,
    use_llm: bool = False,
    verbose: bool = True,
    seed: int = 42,
) -> SimulationResult:
    """Run one full diffusion simulation and return results."""
    random.seed(seed)

    # 1. Build graph
    G = build_graph(num_nodes=num_nodes, topology=topology,
                    edge_prob=edge_prob, seed=seed)

    # 2. Initialise agents with mixed personas
    agents: Dict[int, BeliefAgent] = {}
    for node in G.nodes():
        persona = random.choices(PERSONAS, PERSONA_WEIGHTS)[0]
        agents[node] = BeliefAgent(
            agent_id=node, persona=persona, use_llm=use_llm
        )

    # 3. Seed intervention nodes
    centralities = compute_centralities(G)
    seeds = get_seed_nodes(centralities, strategy, seed_fraction, num_nodes)
    for s in seeds:
        agents[s].belief = 0.85
        agents[s].stage = 2
        agents[s].behavior = 1

    result = SimulationResult(
        strategy=strategy, topology=topology, steps=steps, seed_nodes=seeds
    )

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Agentic Diffusion Sim | {topology} | {strategy} seeding")
        print(f"  Nodes: {num_nodes} | Seeds: {len(seeds)} | Steps: {steps}")
        print(f"{'='*55}")

    # 4. Simulation loop
    for step in range(steps):
        # Message passing: each agent broadcasts to neighbors
        for node in G.nodes():
            for neighbor in G.neighbors(node):
                agents[neighbor].receive_message(
                    sender_id=node,
                    sender_belief=agents[node].belief,
                    sender_stage=agents[node].stage,
                )

        # Belief updating
        step_narratives = []
        for node in G.nodes():
            narrative = agents[node].update(cost=cost)
            if narrative and use_llm:
                step_narratives.append(f"  Agent {node} ({agents[node].persona}): {narrative}")

        # Stats
        adopters = sum(1 for a in agents.values() if a.behavior == 1)
        adoption_rate = adopters / num_nodes
        stage_counts = {STAGE_LABELS[s]: 0 for s in range(4)}
        for a in agents.values():
            stage_counts[STAGE_LABELS[a.stage]] += 1

        result.adoption_history.append(adoption_rate)
        result.stage_history.append(stage_counts)
        if step_narratives:
            result.llm_narratives.extend(step_narratives)

        if verbose:
            bar = "█" * int(adoption_rate * 30)
            print(f"  Step {step+1:02d} | {bar:<30} {adoption_rate:.0%} adopted | "
                  f"Habit:{stage_counts['Habit']:3d}")

    result.final_adopters = sum(1 for a in agents.values() if a.behavior == 1)

    if verbose:
        print(f"\n  Final adoption: {result.final_adopters}/{num_nodes} "
              f"({result.final_adopters/num_nodes:.0%})")

    return result


def compare_strategies(
    strategies: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, SimulationResult]:
    """Run the same simulation across multiple seeding strategies for comparison."""
    if strategies is None:
        strategies = ["betweenness", "percolation", "degree", "random"]

    results = {}
    for strat in strategies:
        results[strat] = run_simulation(strategy=strat, **kwargs)
    return results
