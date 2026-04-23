#!/usr/bin/env python3
"""Experiment D: 10k no BB -- FDDC, 1217 gens, pop 100.
Matches Thomas's roulette gen count. Expected: ~24 hours."""
from shared import *

if __name__ == "__main__":
    print("=" * 50)
    print("EXPERIMENT D: 10k no BB | FDDC | 1217 gens | pop 100")
    print(f"Thomas roulette got: cost=105.5283 in 1217 gens")
    print(f"Started: {datetime.now()}")
    print("=" * 50)

    r = run_fddc("10k", gen_count=1217, pop_size=100, impl="new")
    save_and_report(r, "d_10k_fddc_1217gen.json")

    print(f"\nDone. Total: {datetime.now()}")