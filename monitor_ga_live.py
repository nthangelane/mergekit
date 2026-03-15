#!/usr/bin/env python3
"""
Live GA Run Monitor - Shows real-time progress
"""
import subprocess
import sys
import time
from datetime import datetime

PID = 15428
LOG_FILE = "workspace/ga_10gen_mac_cpu_run.log"
TARGET_GENS = 10


def clear_screen():
    print("\033[2J\033[H", end="")


def get_gen_count():
    try:
        result = subprocess.run(
            ["grep", "[GA]", LOG_FILE], capture_output=True, text=True
        )
        return len([l for l in result.stdout.strip().split("\n") if l])
    except:
        return 0


def get_last_ga_line():
    try:
        result = subprocess.run(
            ["grep", "[GA]", LOG_FILE], capture_output=True, text=True
        )
        lines = [l for l in result.stdout.strip().split("\n") if l]
        return lines[-1] if lines else None
    except:
        return None


def check_process():
    result = subprocess.run(["ps", "-p", str(PID)], capture_output=True)
    return result.returncode == 0


def get_ray_workers():
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    workers = [
        line
        for line in result.stdout.split("\n")
        if "ray::evaluate" in line or "ray::IDLE" in line
    ]
    return len(workers)


def get_latest_logs():
    try:
        result = subprocess.run(
            ["tail", "-30", LOG_FILE], capture_output=True, text=True
        )
        lines = [
            l.strip()
            for l in result.stdout.split("\n")
            if any(x in l for x in ["[GA]", "[SERIAL]", "[EVAL]"])
        ]
        return lines[-5:] if lines else []
    except:
        return []


def main():
    start_time = time.time()
    check_count = 0

    while True:
        check_count += 1
        clear_screen()

        # Header
        print("=" * 70)
        print("🔬 GA 10-Generation Experiment - Live Monitor")
        print("=" * 70)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')} | Check #{check_count}")
        print()

        # Process status
        is_running = check_process()
        elapsed = int(time.time() - start_time)
        mins = elapsed // 60
        secs = elapsed % 60

        print(f"⏱️  Elapsed: {mins}m {secs}s")
        print(f"🔧 Process (PID {PID}): {'🟢 Running' if is_running else '🔴 Stopped'}")

        if not is_running:
            print("\n❌ Process has stopped! Check the log for errors.")
            break

        # Ray workers
        workers = get_ray_workers()
        print(f"👷 Ray Workers: {workers} active")

        # Generation progress
        gens = get_gen_count()
        progress = (gens / TARGET_GENS) * 100
        bar_length = 40
        filled = int(bar_length * gens / TARGET_GENS)
        bar = "█" * filled + "░" * (bar_length - filled)

        print()
        print(f"📊 Generations: {gens}/{TARGET_GENS}")
        print(f"[{bar}] {progress:.0f}%")
        print()

        # Latest generation
        if gens > 0:
            last_ga = get_last_ga_line()
            if last_ga:
                print("📈 Latest Generation:")
                # Extract key metrics
                if "best=" in last_ga:
                    parts = last_ga.split()
                    metrics = [
                        p
                        for p in parts
                        if any(x in p for x in ["gen=", "best=", "mean="])
                    ]
                    print(f"   {' '.join(metrics[:4])}")
                else:
                    print(f"   {last_ga[:70]}...")
        else:
            if elapsed < 180:  # First 3 minutes
                print("⏳ First generation evaluating...")
                print("   (This takes 3-5 minutes on CPU)")
            else:
                print("⚠️  No generations completed yet after 3+ minutes")
                print("   Check the log file for details")

        print()
        print("─" * 70)
        print("📝 Recent Activity:")
        logs = get_latest_logs()
        if logs:
            for log in logs:
                # Truncate long lines
                log_short = log[:65] + "..." if len(log) > 65 else log
                print(f"   {log_short}")
        else:
            print("   (Waiting for activity...)")

        print("─" * 70)

        # Completion check
        if gens >= TARGET_GENS:
            print()
            print("🎉" * 35)
            print("✅ EXPERIMENT COMPLETE! All 10 generations finished!")
            print("🎉" * 35)
            print()
            print("Next steps:")
            print(
                "  python plot_ga_results.py workspace/ga_10gen_mac_cpu_run.log --table"
            )
            break

        # ETA
        if gens > 0:
            avg_time_per_gen = elapsed / gens
            remaining_gens = TARGET_GENS - gens
            eta_seconds = int(avg_time_per_gen * remaining_gens)
            eta_mins = eta_seconds // 60
            print(f"\n⏰ Estimated time remaining: ~{eta_mins} minutes")

        print("\n💡 Press Ctrl+C to stop monitoring (GA will continue running)")

        # Wait before next update
        time.sleep(10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped. GA is still running in background.")
        print(f"   Resume monitoring: python monitor_ga_live.py")
        print(f"   Check progress: grep '[GA]' {LOG_FILE} | wc -l")
        sys.exit(0)
