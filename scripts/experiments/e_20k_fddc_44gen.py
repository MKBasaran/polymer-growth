#!/usr/bin/env python3
"""Experiment E: 20k no BB -- FDDC, 44 gens, pop 100.
Matches Thomas's FDDC run exactly. Expected: ~2-3 hours."""
from shared import *

if __name__ == "__main__":
    print("=" * 50)
    print("EXPERIMENT E: 20k no BB | FDDC | 44 gens | pop 100")
    print(f"Thomas got: cost=131.7344 in 44 gens")
    print(f"Started: {datetime.now()}")
    print("=" * 50)

    r = run_fddc("20k", gen_count=44, pop_size=100, impl="new")
    save_and_report(r, "e_20k_fddc_44gen.json")

    r2 = run_fddc("20k", gen_count=44, pop_size=100, impl="thomas")
    save_and_report(r2, "e_20k_fddc_44gen_thomas.json")

    print(f"\nDone. Total: {datetime.now()}")