"""
Main experiment runner: trains all three agents and runs the test battery.

Usage:
    python -m alien_baby.run_experiment              # full run (50K/50K/100K steps)
    python -m alien_baby.run_experiment --quick       # quick test (5K steps)
    python -m alien_baby.run_experiment --long        # 4x training (200K/200K/400K steps)
    python -m alien_baby.run_experiment --test-only   # just run tests on saved models
"""

import argparse
import pathlib
from alien_baby.agents.train_staged import train_stage1, train_stage2
from alien_baby.agents.train_baselines import train_all_at_once, train_feature_fusion
from alien_baby.tests.test_battery import run_all_tests

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser(description="Alien Baby perception simulation")
    parser.add_argument("--quick", action="store_true", help="Quick run (5K steps)")
    parser.add_argument("--long", action="store_true", help="4x training run (200K/200K/400K steps)")
    parser.add_argument("--test-only", action="store_true", help="Skip training, run tests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick and args.long:
        parser.error("--quick and --long are mutually exclusive")

    if args.quick:
        steps_stage1, steps_stage2, steps_baseline = 5_000, 5_000, 10_000
    elif args.long:
        steps_stage1, steps_stage2, steps_baseline = 200_000, 200_000, 400_000
    else:
        steps_stage1, steps_stage2, steps_baseline = 50_000, 50_000, 100_000

    if not args.test_only:
        print("=" * 60)
        print("ALIEN BABY PERCEPTION SIMULATION EXPERIMENT")
        print("=" * 60)
        print(f"Stage 1 steps: {steps_stage1}")
        print(f"Stage 2 steps: {steps_stage2}")
        print(f"Baseline steps: {steps_baseline}")
        print()

        # Train staged agent
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
