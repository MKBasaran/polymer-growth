#!/usr/bin/env python3
"""Experiment 6: Software engineering metrics report.

Generates quantitative SWE comparison between Thomas's original
code and our refactored codebase. Not a statistical experiment --
a structured metrics report for the thesis.

Usage:
    /usr/local/bin/python3 scripts/experiments/generate_swe_report.py
"""
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
THOMAS_DIR = PROJECT_ROOT / "program code"
OUR_DIR = PROJECT_ROOT / "src" / "polymer_growth"
TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUT_DIR = PROJECT_ROOT / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def count_lines(directory, extensions=('.py',)):
    """Count lines of code, blank lines, and comment lines."""
    total_lines = 0
    code_lines = 0
    blank_lines = 0
    comment_lines = 0
    file_count = 0
    file_stats = []

    for root, _, files in os.walk(directory):
        # Skip __pycache__, .venv, Data directories
        if any(skip in root for skip in ['__pycache__', '.venv', 'Data', '.git']):
            continue
        for fname in sorted(files):
            if not any(fname.endswith(ext) for ext in extensions):
                continue
            fpath = Path(root) / fname
            try:
                lines = fpath.read_text(encoding='utf-8', errors='ignore').splitlines()
            except Exception:
                continue
            file_count += 1
            n_total = len(lines)
            n_blank = sum(1 for l in lines if not l.strip())
            n_comment = sum(1 for l in lines if l.strip().startswith('#'))
            n_code = n_total - n_blank - n_comment
            total_lines += n_total
            code_lines += n_code
            blank_lines += n_blank
            comment_lines += n_comment
            rel_path = str(fpath.relative_to(PROJECT_ROOT))
            file_stats.append({
                "file": rel_path,
                "total": n_total,
                "code": n_code,
                "blank": n_blank,
                "comment": n_comment,
            })

    return {
        "file_count": file_count,
        "total_lines": total_lines,
        "code_lines": code_lines,
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "avg_lines_per_file": round(total_lines / file_count, 1) if file_count else 0,
        "files": file_stats,
    }


def count_tests():
    """Count test functions and files."""
    test_files = 0
    test_functions = 0
    test_names = []

    for fpath in TESTS_DIR.rglob("test_*.py"):
        test_files += 1
        lines = fpath.read_text().splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def test_"):
                test_functions += 1
                test_names.append(stripped.split("(")[0].replace("def ", ""))

    return {
        "test_files": test_files,
        "test_functions": test_functions,
        "test_names": test_names,
    }


def get_dependencies():
    """Parse pyproject.toml for dependency count."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return {"core": 0, "optional": 0, "names": []}

    text = pyproject.read_text()
    deps = []
    in_deps = False
    for line in text.splitlines():
        if line.strip() == 'dependencies = [':
            in_deps = True
            continue
        if in_deps:
            if line.strip() == ']':
                break
            name = line.strip().strip('"').strip("'").strip(",")
            if name:
                pkg = name.split(">=")[0].split("<")[0].split("==")[0].strip()
                deps.append(pkg)

    return {
        "core_count": len(deps),
        "core_names": deps,
    }


def try_radon(directory):
    """Try to compute cyclomatic complexity with radon."""
    try:
        result = subprocess.run(
            ["radon", "cc", str(directory), "-a", "-s", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            complexities = []
            for fpath, blocks in data.items():
                for block in blocks:
                    complexities.append(block.get("complexity", 0))
            if complexities:
                import numpy as np
                return {
                    "available": True,
                    "mean": round(float(np.mean(complexities)), 2),
                    "median": round(float(np.median(complexities)), 2),
                    "max": int(max(complexities)),
                    "n_functions": len(complexities),
                }
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return {"available": False}


def main():
    print("=" * 60)
    print("EXPERIMENT 6: SOFTWARE ENGINEERING METRICS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # --- Line counts ---
    thomas_stats = count_lines(THOMAS_DIR)
    our_stats = count_lines(OUR_DIR)
    test_stats = count_lines(TESTS_DIR)

    print("\n--- LINE COUNTS ---")
    print(f"{'Metric':<25} {'Thomas':>10} {'Ours':>10} {'Tests':>10}")
    print("-" * 58)
    print(f"{'Python files':<25} {thomas_stats['file_count']:>10} "
          f"{our_stats['file_count']:>10} {test_stats['file_count']:>10}")
    print(f"{'Total lines':<25} {thomas_stats['total_lines']:>10} "
          f"{our_stats['total_lines']:>10} {test_stats['total_lines']:>10}")
    print(f"{'Code lines':<25} {thomas_stats['code_lines']:>10} "
          f"{our_stats['code_lines']:>10} {test_stats['code_lines']:>10}")
    print(f"{'Avg lines/file':<25} {thomas_stats['avg_lines_per_file']:>10} "
          f"{our_stats['avg_lines_per_file']:>10} {test_stats['avg_lines_per_file']:>10}")

    # --- Per-file breakdown ---
    print("\n--- FILE BREAKDOWN (Thomas) ---")
    for f in thomas_stats["files"]:
        print(f"  {f['file']:<50} {f['code']:>5} code lines")

    print("\n--- FILE BREAKDOWN (Ours) ---")
    for f in our_stats["files"]:
        print(f"  {f['file']:<50} {f['code']:>5} code lines")

    # --- Tests ---
    tests = count_tests()
    print(f"\n--- TESTS ---")
    print(f"  Thomas: 0 test files, 0 test functions")
    print(f"  Ours:   {tests['test_files']} test files, "
          f"{tests['test_functions']} test functions")

    # --- Dependencies ---
    deps = get_dependencies()
    print(f"\n--- DEPENDENCIES ---")
    print(f"  Thomas: manual imports (numpy, pandas, matplotlib, scipy, openpyxl)")
    print(f"  Ours:   {deps['core_count']} declared in pyproject.toml")
    print(f"          {', '.join(deps['core_names'])}")

    # --- Cyclomatic complexity ---
    print(f"\n--- CYCLOMATIC COMPLEXITY ---")
    thomas_cc = try_radon(THOMAS_DIR)
    our_cc = try_radon(OUR_DIR)
    if thomas_cc["available"]:
        print(f"  Thomas: mean={thomas_cc['mean']}, median={thomas_cc['median']}, "
              f"max={thomas_cc['max']} ({thomas_cc['n_functions']} functions)")
    else:
        print(f"  Thomas: radon not available (install: pip install radon)")
    if our_cc["available"]:
        print(f"  Ours:   mean={our_cc['mean']}, median={our_cc['median']}, "
              f"max={our_cc['max']} ({our_cc['n_functions']} functions)")
    else:
        print(f"  Ours:   radon not available (install: pip install radon)")

    # --- Qualitative ---
    print(f"\n--- QUALITATIVE COMPARISON ---")
    qualitative = {
        "modularity": {
            "thomas": "Flat directory, all .py files in one folder, no package structure",
            "ours": "pip-installable package with core/, objective/, optimizers/, gui/, cli/ subpackages",
        },
        "testability": {
            "thomas": "No tests, no test infrastructure",
            "ours": f"{tests['test_functions']} tests across {tests['test_files']} files (pytest)",
        },
        "installability": {
            "thomas": "Manual: clone repo, install deps by hand, run from directory",
            "ours": "pip install -e . (pyproject.toml with declared dependencies)",
        },
        "reproducibility": {
            "thomas": "No deterministic seeding, results vary between runs",
            "ours": "eval_seed parameter ensures bit-identical results for same seed",
        },
        "parallelization": {
            "thomas": "Hardcoded Pool(6), no BLAS thread suppression",
            "ours": "Configurable n_workers, BLAS suppression, fork-based ProcessPool",
        },
        "gui": {
            "thomas": "Tkinter, coupled to optimization logic",
            "ours": "PySide6, 3-tab interface, threaded workers, decoupled from logic",
        },
    }
    for aspect, comparison in qualitative.items():
        print(f"\n  {aspect.upper()}:")
        print(f"    Thomas: {comparison['thomas']}")
        print(f"    Ours:   {comparison['ours']}")

    # --- Save report ---
    report = {
        "generated": datetime.now().isoformat(),
        "thomas_code": thomas_stats,
        "our_code": our_stats,
        "tests": {**test_stats, **tests},
        "dependencies": deps,
        "complexity": {"thomas": thomas_cc, "ours": our_cc},
        "qualitative": qualitative,
    }
    out_file = OUTPUT_DIR / "exp6_swe_report.json"
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_file.name}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()