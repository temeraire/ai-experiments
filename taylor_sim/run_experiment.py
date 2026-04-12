"""
Main experiment runner: trains all three agents and runs the test battery.

Usage:
    python -m taylor_sim.run_experiment              # full run
    python -m taylor_sim.run_experiment --quick       # quick test (5K steps)
    python -m taylor_sim.run_experiment --test-only   # just run tests on saved models
"""

import argparse
import pathlib
from taylor_sim.agents.train_staged import train_stage1, train_stage2
from taylor_sim.agents.train_baselines import train_all_at_once, train_feature_fusion
from taylor_sim.tests.test_battery import run_all_tests

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser(description="Taylor Perception Simulation")
    parser.add_argument("--quick", action="store_true", help="Quick run (5K steps)")
    parser.add_argument("--test-only", action="store_true", help="Skip training, run tests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    steps_stage1 = 5_000 if args.quick else 50_000
    steps_stage2 = 5_000 if args.quick else 50_000
    steps_baseline = 10_000 if args.quick else 100_000

    if not args.test_only:
        print("=" * 60)
        print("TAYLOR PERCEPTION SIMULATION EXPERIMENT")
        print("=" * 60)
        print(f"Stage 1 steps: {steps_stage1}")
        print(f"Stage 2 steps: {steps_stage2}")
        print(f"Baseline steps: {steps_baseline}")
        print()

        # Train staged (Taylor) agent
        train_stage1(total_timesteps=steps_stage1, seed=args.seed)
        train_stage2(total_timesteps=steps_stage2, seed=args.seed)

        # Train baselines
        train_all_at_once(total_timesteps=steps_baseline, seed=args.seed)
        train_feature_fusion(total_timesteps=steps_baseline, seed=args.seed)

    # Run test battery
    print("\n\nRunning test battery...\n")
    results = run_all_tests(
        staged_path=str(RESULTS_DIR / "stage2_checkpoint"),
        allatonce_path=str(RESULTS_DIR / "allatonce_checkpoint"),
        fusion_path=str(RESULTS_DIR / "fusion_checkpoint"),
    )

    print("\n\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
