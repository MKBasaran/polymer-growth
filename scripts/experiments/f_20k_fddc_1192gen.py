#!/usr/bin/env python3
"""Experiment F: 20k no BB -- FDDC, 1192 gens, pop 100.
Matches Thomas's roulette gen count. Expected: ~24 hours."""
from shared import *

if __name__ == "__main__":
    print("=" * 50)
    print("EXPERIMENT F: 20k no BB | FDDC | 1192 gens | pop 100")
    print(f"Thomas roulette got: cost=129.7423 in 1192 gens")
    print(f"Started: {datetime.now()}")
    print("=" * 50)

    r = run_fddc("20k", gen_count=1192, pop_size=100, impl="new")
    save_and_report(r, "f_20k_fddc_1192gen.json")

    print(f"\nDone. Total: {datetime.now()}")