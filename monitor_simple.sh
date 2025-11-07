#!/bin/bash
# Simple monitoring script - run this in a terminal

echo "Monitoring GA run (updates every 15 seconds)"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "========================================"
    echo "GA Run Monitor - $(date '+%H:%M:%S')"
    echo "========================================"
    echo ""
    
    # Check if process is running
    if ps -p 15428 > /dev/null 2>&1; then
        echo "✅ Process: Running (PID 15428)"
    else
        echo "❌ Process: Stopped"
        echo ""
        echo "Check the log: tail -50 workspace/ga_10gen_mac_cpu_run.log"
        exit 1
    fi
    
    # Count generations
    GENS=$(grep '\[GA\]' workspace/ga_10gen_mac_cpu_run.log 2>/dev/null | wc -l | tr -d ' ')
    echo "📊 Generations: $GENS/10"
    
    # Progress bar
    PERCENT=$((GENS * 10))
    printf "Progress: ["
    for i in {1..10}; do
        if [ $i -le $GENS ]; then
            printf "█"
        else
            printf "░"
        fi
    done
    printf "] $PERCENT%%\n"
    
    echo ""
    echo "Latest activity:"
    tail -8 workspace/ga_10gen_mac_cpu_run.log | tail -5
    
    echo ""
    echo "─────────────────────────────────────────"
    
    if [ "$GENS" -ge 10 ]; then
        echo ""
        echo "🎉 COMPLETE! All 10 generations finished!"
        echo ""
        echo "View results:"
        echo "  grep '[GA]' workspace/ga_10gen_mac_cpu_run.log"
        break
    fi
    
    sleep 15
done
