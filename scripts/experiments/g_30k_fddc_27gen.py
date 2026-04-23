#!/usr/bin/env python3
"""Experiment G: 30k no BB -- FDDC, 27 gens, pop 100.
Matches Thomas's FDDC run exactly. Expected: ~1-2 hours."""
from shared import *

if __name__ == "__main__":
    print("=" * 50)
    print("EXPERIMENT G: 30k no BB | FDDC | 27 gens | pop 100")
    print(f"Thomas got: cost=340.2589 in 27 gens")
    print(f"Started: {datetime.now()}")
    print("=" * 50)

    r = run_fddc("30k", gen_count=27, pop_size=100, impl="new")
    save_and_report(r, "g_30k_fddc_27gen.json")

    r2 = run_fddc("30k", gen_count=27, pop_size=100, impl="thomas")
    save_and_report(r2, "g_30k_fddc_27gen_thomas.json")

    print(f"\nDone. Total: {datetime.now()}")