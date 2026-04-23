#!/usr/bin/env python3
"""Run all remaining short experiments (C, E, G) back to back.
Each runs both impls. Total estimated: ~8-10 hours."""
from shared import *

if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING C, E, G (all short, both impls)")
    print(f"Started: {datetime.now()}")
    print("=" * 50)

    run_simulation_equivalence(30)

    # C: 10k, 56 gens
    print("\n\n>>> EXPERIMENT C: 10k no BB | 56 gens <<<\n")
    r = run_fddc("10k", gen_count=56, pop_size=100, impl="new")
    save_and_report(r, "c_10k_fddc_56gen.json")
    r2 = run_fddc("10k", gen_count=56, pop_size=100, impl="thomas")
    save_and_report(r2, "c_10k_fddc_56gen_thomas.json")

    # E: 20k, 44 gens
    print("\n\n>>> EXPERIMENT E: 20k no BB | 44 gens <<<\n")
    r = run_fddc("20k", gen_count=44, pop_size=100, impl="new")
    save_and_report(r, "e_20k_fddc_44gen.json")
    r2 = run_fddc("20k", gen_count=44, pop_size=100, impl="thomas")
    save_and_report(r2, "e_20k_fddc_44gen_thomas.json")

    # G: 30k, 27 gens
    print("\n\n>>> EXPERIMENT G: 30k no BB | 27 gens <<<\n")
    r = run_fddc("30k", gen_count=27, pop_size=100, impl="new")
    save_and_report(r, "g_30k_fddc_27gen.json")
    r2 = run_fddc("30k", gen_count=27, pop_size=100, impl="thomas")
    save_and_report(r2, "g_30k_fddc_27gen_thomas.json")

    print(f"\n\nALL DONE: {datetime.now()}")