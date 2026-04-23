#!/usr/bin/env python3
"""Experiment C: 10k no BB -- FDDC, 56 gens, pop 100.
Matches Thomas's FDDC run exactly. Expected: ~2-3 hours."""
from shared import *

if __name__ == "__main__":
    print("=" * 50)
    print("EXPERIMENT C: 10k no BB | FDDC | 56 gens | pop 100")
    print(f"Thomas got: cost=108.5352 in 56 gens")
    print(f"Started: {datetime.now()}")
    print("=" * 50)

    r = run_fddc("10k", gen_count=56, pop_size=100, impl="new")
    save_and_report(r, "c_10k_fddc_56gen.json")

    r2 = run_fddc("10k", gen_count=56, pop_size=100, impl="thomas")
    save_and_report(r2, "c_10k_fddc_56gen_thomas.json")

    print(f"\nDone. Total: {datetime.now()}")