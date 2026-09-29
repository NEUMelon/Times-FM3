"""Master autonomous pipeline orchestrator: sequentially coordinates E1 -> E5 -> E2 -> E3 -> Paper Analysis."""

import os
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/DASFAA")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe.config import RESULTS_DIR


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [MASTER PIPELINE] {msg}"
    print(formatted, flush=True)


def run_command(cmd: str, desc: str):
    log(f"Starting {desc}...")
    log(f"Command: {cmd}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, executable="/bin/bash")
    elapsed = time.time() - t0
    if res.returncode != 0:
        log(f"ERROR: {desc} exited with code {res.returncode} after {elapsed:.1f}s")
        raise RuntimeError(f"{desc} failed!")
    log(f"SUCCESS: {desc} completed in {elapsed:.1f}s")


def wait_for_e1():
    """Wait until E1 background job finishes and produces e1_ltsf_sampled.csv."""
    log("Monitoring E1 background execution...")
    csv_path = RESULTS_DIR / "e1_ltsf_sampled.csv"
    while True:
        # Check if python process for run_e1_ltsf is still running
        ps = subprocess.run("ps aux | grep run_e1_ltsf.py | grep -v grep | wc -l", shell=True, capture_output=True, text=True)
        count = int(ps.stdout.strip() or 0)
        
        if count == 0:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                log(f"E1 finished! Produced {len(df)} summary records in {csv_path}")
            else:
                log("E1 process exited. Checking results directory...")
            break
        else:
            time.sleep(15)


def main():
    log("=" * 80)
    log("MASTER ORCHESTRATOR INITIALIZED")
    log("=" * 80)

    # 1. Check E1
    e1_csv = RESULTS_DIR / "e1_ltsf_sampled.csv"
    if e1_csv.exists():
        log(f"E1 summary already exists at {e1_csv}. Skipping E1.")
    else:
        wait_for_e1()

    # 2. Check E5
    e5_csv = RESULTS_DIR / "e5_grouping_summary.csv"
    if e5_csv.exists() and len(pd.read_csv(e5_csv)) >= 42:
        log(f"E5 summary already exists with {len(pd.read_csv(e5_csv))} rows at {e5_csv}. Skipping E5.")
    else:
        run_command(
            "python3 -u /root/autodl-tmp/DASFAA/experiments/run_e5_grouping.py --datasets electricity solar --horizon 96 --samples 50",
            "E5: Channel Budget K & Efficiency Front",
        )

    # 3. Run E2: Interventions (N1-N4, I1-I6, P1) on ETTh1, weather, electricity, solar
    run_command(
        "python3 -u /root/autodl-tmp/DASFAA/experiments/run_e2_interventions.py --datasets ETTh1 weather electricity solar --horizon 96 --samples 100",
        "E2: Target-Centric Interventions",
    )

    # 4. Run E3: Synthetic data & Kalman Oracles (G1-G6)
    run_command(
        "python3 -u /root/autodl-tmp/DASFAA/experiments/run_e3_synthetic.py --windows 30 --horizon 96",
        "E3: Controlled Synthetic Structures & Oracles",
    )

    # 5. Run Paper Analysis & Plot Generation
    run_command(
        "python3 -u /root/autodl-tmp/DASFAA/analysis/generate_paper_tables_and_plots.py",
        "Paper Tables & Figure Generation",
    )

    log("=" * 80)
    log("ALL EXPERIMENTS & ARTIFACTS COMPLETE!")
    log("=" * 80)


if __name__ == "__main__":
    main()
