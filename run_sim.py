"""
run_sim.py — Command-line entry point for Agentic Diffusion Simulator.

Usage examples:

  # Fast math mode, compare all strategies
  python run_sim.py

  # Specific strategy, different topology
  python run_sim.py --strategy betweenness --topology barabasi_albert --nodes 80

  # LLM mode (needs GROQ_API_KEY)
  python run_sim.py --strategy betweenness --llm --nodes 20 --steps 5

  # Enterprise scenario (single run, verbose)
  python run_sim.py --scenario enterprise --nodes 100 --steps 30
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.diffusion import run_simulation, compare_strategies
from visualization.animate import plot_adoption_curves, plot_stage_flow, plot_network_snapshot


SCENARIOS = {
    "enterprise": {
        "topology": "barabasi_albert",
        "num_nodes": 80,
        "cost": 0.35,
        "steps": 30,
        "description": "Enterprise AI adoption — scale-free org network",
    },
    "dharma": {
        "topology": "small_world",
        "num_nodes": 60,
        "cost": 0.20,
        "steps": 25,
        "description": "Dharma practice diffusion — small-world sangha network",
    },
    "demonetization": {
        "topology": "erdos_renyi",
        "num_nodes": 100,
        "edge_prob": 0.08,
        "cost": 0.50,
        "steps": 20,
        "description": "Krishnan 2025 baseline — digital payment adoption under shock",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Diffusion Simulator — based on Krishnan (2025)"
    )
    parser.add_argument("--nodes", type=int, default=60)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--topology", choices=["erdos_renyi", "barabasi_albert", "small_world"],
                        default="erdos_renyi")
    parser.add_argument("--strategy", choices=["betweenness", "percolation", "degree", "random", "all"],
                        default="all")
    parser.add_argument("--cost", type=float, default=0.30)
    parser.add_argument("--seed_frac", type=float, default=0.10)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default=None)
    parser.add_argument("--llm", action="store_true", help="Use Groq LLM for belief updates")
    parser.add_argument("--output_dir", default="/mnt/user-data/outputs")
    args = parser.parse_args()

    # Scenario overrides
    scenario_params = {}
    if args.scenario:
        s = SCENARIOS[args.scenario]
        print(f"\n  Scenario: {s['description']}")
        scenario_params = {k: v for k, v in s.items() if k != "description"}

    params = {
        "num_nodes": scenario_params.get("num_nodes", args.nodes),
        "steps": scenario_params.get("steps", args.steps),
        "topology": scenario_params.get("topology", args.topology),
        "cost": scenario_params.get("cost", args.cost),
        "edge_prob": scenario_params.get("edge_prob", 0.12),
        "seed_fraction": args.seed_frac,
        "use_llm": args.llm,
    }

    if args.strategy == "all":
        print("\nRunning all 4 seeding strategies for comparison...\n")
        results = compare_strategies(**params)

        print("\nGenerating visualizations...")
        plot_adoption_curves(results, output_dir=args.output_dir)
        best_strat = max(results, key=lambda k: results[k].final_adopters)
        plot_stage_flow(results[best_strat], output_dir=args.output_dir)
        plot_network_snapshot(
            results[best_strat],
            num_nodes=params["num_nodes"],
            topology=params["topology"],
            output_dir=args.output_dir,
        )

        print("\n--- Summary ---")
        for strat, res in sorted(results.items(), key=lambda x: -x[1].final_adopters):
            pct = res.final_adopters / params["num_nodes"] * 100
            print(f"  {strat:12s}: {res.final_adopters:3d}/{params['num_nodes']} adopted ({pct:.0f}%)")
    else:
        result = run_simulation(strategy=args.strategy, **params)
        plot_stage_flow(result, output_dir=args.output_dir)
        plot_network_snapshot(
            result,
            num_nodes=params["num_nodes"],
            topology=params["topology"],
            output_dir=args.output_dir,
        )

        if args.llm and result.llm_narratives:
            print("\n--- LLM Agent Narratives (sample) ---")
            for line in result.llm_narratives[:10]:
                print(line)


if __name__ == "__main__":
    main()
