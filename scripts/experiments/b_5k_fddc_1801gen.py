#!/usr/bin/env python3
"""Experiment B: 5k no BB -- FDDC, 1801 gens, pop 100.
Matches Thomas's roulette gen count. Expected: ~24 hours."""
from shared import *

if __name__ == "__main__":
    print("=" * 50)
    print("EXPERIMENT B: 5k no BB | FDDC | 1801 gens | pop 100")
    print(f"Thomas roulette got: cost=13.3815 in 1801 gens")
    print(f"Started: {datetime.now()}")
    print("=" * 50)

    r = run_fddc("5k", gen_count=1801, pop_size=100, impl="new")
    save_and_report(r, "b_5k_fddc_1801gen.json")

    print(f"\nDone. Total: {datetime.now()}")