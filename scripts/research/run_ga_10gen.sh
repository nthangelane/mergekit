#!/bin/bash
# scripts/research/run_ga_10gen.sh - Run 10-generation GA experiment on CPU
# This script runs the GA experiment in the background and won't be interrupted

echo "🚀 Starting 10-generation GA experiment..."
echo "📁 Working directory: $(pwd)"
echo "⏱️  Expected runtime: 15-20 minutes"
echo ""

# Clean up any old Ray processes
echo "🧹 Cleaning up old Ray processes..."
ray stop --force 2>/dev/null
sleep 2

# Run the GA experiment
echo "▶️  Starting GA run..."
echo "📊 Log file: workspace/ga_10gen_mac_cpu_run.log"
echo "🌐 Ray Dashboard: http://127.0.0.1:8265 (once started)"
echo ""

CONFIG_FILE="workspace/ga_10gen_mac_cpu.yml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file '$CONFIG_FILE' not found!"
    echo "   Please create it or check the path."
    exit 1
fi

nohup python -m mergekit.scripts.evolve_ga \
  workspace/ga_10gen_mac_cpu.yml \
  --storage-path workspace/ga_10gen_mac_storage \
  --save-final-model \
  --no-merge-cuda \
  --strategy serial \
  --num-workers 1 \
  --num-gpus 0 \
  > workspace/ga_10gen_mac_cpu_run.log 2>&1 &

GA_PID=$!
echo "✅ GA process started with PID: $GA_PID"
echo ""
echo "📋 Monitor commands:"
echo "  - Watch live:        tail -f workspace/ga_10gen_mac_cpu_run.log | grep '\[GA\]'"
echo "  - Check count:       grep '\[GA\]' workspace/ga_10gen_mac_cpu_run.log | wc -l"
echo "  - View all output:   tail -f workspace/ga_10gen_mac_cpu_run.log"
echo "  - Check if running:  ps -p $GA_PID"
echo "  - Stop process:      kill $GA_PID"
echo ""
echo "⏳ The process is running in the background. Check back in 15-20 minutes!"
echo "   Or monitor progress with: tail -f workspace/ga_10gen_mac_cpu_run.log"
