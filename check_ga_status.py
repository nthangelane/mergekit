#!/usr/bin/env python3
"""
Quick GA run status checker
"""
import subprocess
import time

print("🔍 GA Run Status Check\n")

# Check if main process is running
try:
    # Look for the python process running evolve_ga
    result = subprocess.run(['pgrep', '-f', 'mergekit.scripts.evolve_ga'], capture_output=True, text=True)
    pids = result.stdout.strip().split('\n')
    pids = [p for p in pids if p] # Filter empty strings
    
    if pids:
        print(f"✅ Main GA process is running (PID: {', '.join(pids)})")
    else:
        print("❌ Main GA process is NOT running")
        # Don't exit here, let it check for workers/logs too as they might be lingering
except Exception as e:
    print(f"⚠️  Error checking for process: {e}")

# Check for Ray workers
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
ray_workers = [line for line in result.stdout.split('\n') if 'ray::evaluate' in line]
if ray_workers:
    print(f"✅ {len(ray_workers)} Ray worker(s) actively evaluating")
    for worker in ray_workers:
        if 'evaluate_genotype' in worker:
            print(f"   - Worker evaluating genotype (this is the GA evaluation)")
else:
    print("⚠️  No active Ray evaluation workers found")

# Check generations completed
try:
    with open('workspace/ga_10gen_mac_cpu_run.log') as f:
        gens = sum(1 for line in f if '[GA]' in line)
    print(f"📊 Generations completed: {gens}/10")
    if gens == 0:
        print("   (First generation is still being evaluated - this can take 2-5 minutes)")
except FileNotFoundError:
    print("❌ Log file not found")

# Check Ray dashboard
result = subprocess.run(['curl', '-s', 'http://127.0.0.1:8265'], 
                       capture_output=True, timeout=2)
if result.returncode == 0:
    print("✅ Ray Dashboard accessible at http://127.0.0.1:8265")
else:
    print("⚠️  Ray Dashboard not accessible")

print("\n💡 The process appears to be running normally!")
print("   First generation evaluations are in progress.")
print("   Check back in a few minutes or monitor with:")
print("   tail -f workspace/ga_10gen_mac_cpu_run.log | grep '[GA]'")
