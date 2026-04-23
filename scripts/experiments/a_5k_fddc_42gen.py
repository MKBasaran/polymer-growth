#!/usr/bin/env python3
"""Experiment A: 5k no BB -- FDDC, 42 gens, pop 100.
Matches Thomas's FDDC run exactly. Expected: ~1-2 hours."""
from shared import *

if __name__ == "__main__":
    print("=" * 50)
    print("EXPERIMENT A: 5k no BB | FDDC | 42 gens | pop 100")
    print(f"Thomas got: cost=21.7358 in 42 gens")
    print(f"Started: {datetime.now()}")
    print("=" * 50)

    run_simulation_equivalence(30)

    r = run_fddc("5k", gen_count=42, pop_size=100, impl="new")
    save_and_report(r, "a_5k_fddc_42gen.json")

    r2 = run_fddc("5k", gen_count=42, pop_size=100, impl="thomas")
    save_and_report(r2, "a_5k_fddc_42gen_thomas.json")

    print(f"\nDone. Total: {datetime.now()}")