#!/usr/bin/env python3
"""
Monitor GA run progress and report when complete.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("workspace/ga_10gen_mac_cpu_run.log")
TARGET_GENS = 10
CHECK_INTERVAL = 30  # seconds


def count_generations():
    """Count [GA] lines in log file."""
    if not LOG_FILE.exists():
        return 0
    with open(LOG_FILE) as f:
        return sum(1 for line in f if "[GA]" in line)


def get_latest_generation():
    """Get the last [GA] line from log."""
    if not LOG_FILE.exists():
        return None
    with open(LOG_FILE) as f:
        lines = [line.strip() for line in f if "[GA]" in line]
    return lines[-1] if lines else None


def check_for_errors():
    """Check if there are any errors in the log."""
    if not LOG_FILE.exists():
        return False
    with open(LOG_FILE) as f:
        content = f.read()
        return "AssertionError" in content or "Traceback" in content


def main():
    print("🔍 Monitoring GA run progress...")
    print(f"📁 Log file: {LOG_FILE}")
    print(f"🎯 Target: {TARGET_GENS} generations")
    print(f"⏱️  Checking every {CHECK_INTERVAL} seconds")
    print("─" * 60)

    last_count = 0
    start_time = time.time()

    while True:
        current_count = count_generations()
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = int((time.time() - start_time) / 60)

        # Status update
        print(
            f"[{timestamp}] Generations: {current_count}/{TARGET_GENS} | Elapsed: {elapsed}m",
            end="",
        )

        # Show latest if new generation appeared
        if current_count > last_count:
            print(f" | ✨ NEW!")
            latest = get_latest_generation()
            if latest:
                # Extract key info
                parts = latest.split()
                gen_info = [
                    p for p in parts if "gen=" in p or "best=" in p or "mean=" in p
                ]
                print(f"    {' '.join(gen_info[:3])}")
            last_count = current_count
        else:
            print()

        # Check completion
        if current_count >= TARGET_GENS:
            print("\n" + "=" * 60)
            print("✅ RUN COMPLETE! All 10 generations finished.")
            print("=" * 60)
            print("\n📊 All generation results:")
            with open(LOG_FILE) as f:
                for line in f:
                    if "[GA]" in line:
                        print(line.strip())

            print("\n🎉 Next steps:")
            print("1. Extract metrics:")
            print(
                "   python scripts/research/plot_ga_results.py workspace/ga_10gen_mac_cpu_run.log --table"
            )
            print("\n2. Generate plot:")
            print(
                "   python scripts/research/plot_ga_results.py workspace/ga_10gen_mac_cpu_run.log \\"
            )
            print("     --output convergence.png --csv metrics.csv")
            print("\n3. Compare results:")
            print("   grep '[GA]' workspace/ga_10gen_run.log | wc -l  # Old: 1")
            print(
                "   grep '[GA]' workspace/ga_10gen_mac_cpu_run.log | wc -l  # New: 10"
            )
            break

        # Check for errors
        if check_for_errors():
            print("⚠️  WARNING: Potential error detected in log!")
            print("    Check: tail -50 workspace/ga_10gen_mac_cpu_run.log")

        # Wait before next check
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped. Run is still continuing in background.")
        print("   Resume monitoring: python scripts/research/monitor_ga.py")
        sys.exit(0)
