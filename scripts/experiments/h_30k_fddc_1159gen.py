#!/usr/bin/env python3
"""Experiment H: 30k no BB -- FDDC, 1159 gens, pop 100.
Matches Thomas's roulette gen count. Expected: ~24 hours."""
from shared import *

if __name__ == "__main__":
    print("=" * 50)
    print("EXPERIMENT H: 30k no BB | FDDC | 1159 gens | pop 100")
    print(f"Thomas roulette got: cost=285.9435 in 1159 gens")
    print(f"Started: {datetime.now()}")
    print("=" * 50)

    r = run_fddc("30k", gen_count=1159, pop_size=100, impl="new")
    save_and_report(r, "h_30k_fddc_1159gen.json")

    print(f"\nDone. Total: {datetime.now()}")