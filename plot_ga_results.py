#!/usr/bin/env python3
"""
Extract and plot GA evolution results from logs.
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("⚠️  matplotlib not installed. Install with: pip install matplotlib numpy")
    print("Falling back to CSV export only.")
    plt = None
    np = None


def extract_ga_metrics(log_file: str) -> List[Dict]:
    """Extract [GA] metrics from log file."""
    metrics = []
    
    pattern = r'\[GA\]\s+gen=(\d+)\s+best=([\d.e+-]+)\s+mean=([\d.e+-]+)\s+std=([\d.e+-]+)\s+evaluated=(\d+)\s+cache_hits=(\d+)\s+failed=(\d+)\s+crossover_children=(\d+)\s+type=(\w+)\s+immigrants=(\d+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                metrics.append({
                    'generation': int(match.group(1)),
                    'best_score': float(match.group(2)),
                    'mean_score': float(match.group(3)),
                    'std_score': float(match.group(4)),
                    'evaluated': int(match.group(5)),
                    'cache_hits': int(match.group(6)),
                    'failed': int(match.group(7)),
                    'crossover_children': int(match.group(8)),
                    'crossover_type': match.group(9),
                    'immigrants': int(match.group(10)),
                })
    
    return metrics


def save_to_csv(metrics: List[Dict], output_file: str):
    """Save metrics to CSV file."""
    import csv
    
    if not metrics:
        print("No metrics found to save")
        return
    
    keys = metrics[0].keys()
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)
    
    print(f"✅ CSV saved to: {output_file}")


def plot_results(metrics: List[Dict], output_file: str = "ga_results.png"):
    """Create visualizations of GA results."""
    
    if not plt or not np:
        print("⚠️  Plotting requires matplotlib. Install with: pip install matplotlib numpy")
        return
    
    if not metrics:
        print("No metrics to plot")
        return
    
    gens = [m['generation'] for m in metrics]
    best_scores = [m['best_score'] for m in metrics]
    mean_scores = [m['mean_score'] for m in metrics]
    std_scores = [m['std_score'] for m in metrics]
    cache_hits = [m['cache_hits'] for m in metrics]
    failed = [m['failed'] for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GA Evolution Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Best vs Mean Score
    ax = axes[0, 0]
    ax.plot(gens, best_scores, 'g-o', label='Best Score', linewidth=2, markersize=8)
    ax.plot(gens, mean_scores, 'b-s', label='Mean Score', linewidth=2, markersize=6)
    ax.fill_between(gens, 
                     np.array(mean_scores) - np.array(std_scores),
                     np.array(mean_scores) + np.array(std_scores),
                     alpha=0.2, color='blue', label='±1 Std Dev')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Score (Perplexity)', fontsize=11)
    ax.set_title('Fitness Over Generations', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Best Score Improvement
    ax = axes[0, 1]
    improvements = [best_scores[0] - s for s in best_scores]
    ax.bar(gens, improvements, color='green', alpha=0.7)
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Improvement from Gen 0 (abs)', fontsize=11)
    ax.set_title('Best Score Improvement', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cache Hits vs Failed Evals
    ax = axes[1, 0]
    x = np.arange(len(gens))
    width = 0.35
    ax.bar(x - width/2, cache_hits, width, label='Cache Hits', alpha=0.8, color='blue')
    ax.bar(x + width/2, failed, width, label='Failed Evals', alpha=0.8, color='red')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Cache Hits vs Failed Evaluations', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gens)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Statistics Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    GA EVOLUTION SUMMARY
    
    Total Generations: {len(metrics)}
    Total Evaluations: {sum(1 for m in metrics)}
    
    Best Score: {min(best_scores):.2e}
    Worst Score: {max(best_scores):.2e}
    
    Total Cache Hits: {sum(cache_hits)}
    Total Failed Evals: {sum(failed)}
    
    Best Gen: {gens[best_scores.index(min(best_scores))]}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    plt.close()


def print_table(metrics: List[Dict]):
    """Print metrics as a formatted table."""
    
    if not metrics:
        print("No metrics to display")
        return
    
    print("\n" + "="*120)
    print(f"{'Gen':<5} {'Best Score':<16} {'Mean Score':<16} {'Std Dev':<14} {'Cache':<8} {'Failed':<8} {'Crossover':<10}")
    print("="*120)
    
    for m in metrics:
        print(f"{m['generation']:<5} {m['best_score']:<16.4e} {m['mean_score']:<16.4e} {m['std_score']:<14.4e} "
              f"{m['cache_hits']:<8} {m['failed']:<8} {m['crossover_children']:<10}")
    
    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Extract and plot GA evolution results')
    parser.add_argument('log_file', help='Path to GA run log file')
    parser.add_argument('--output', '-o', default='ga_results.png', 
                       help='Output plot filename (default: ga_results.png)')
    parser.add_argument('--csv', help='Also save metrics to CSV file')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--table', action='store_true', help='Print metrics table')
    
    args = parser.parse_args()
    
    # Extract metrics
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"❌ Error: Log file not found: {args.log_file}")
        return
    
    print(f"📖 Reading log file: {args.log_file}")
    metrics = extract_ga_metrics(str(log_path))
    
    if not metrics:
        print("❌ No GA metrics found in log file")
        print("   Make sure the log contains [GA] generation summary lines")
        return
    
    print(f"✅ Found {len(metrics)} generation(s)")
    
    # Print table
    if args.table or not args.no_plot:
        print_table(metrics)
    
    # Save CSV
    if args.csv:
        save_to_csv(metrics, args.csv)
    
    # Create plots
    if not args.no_plot:
        plot_results(metrics, args.output)
    
    print("\n💡 Tip: Run more GA evaluations to see more generations and better trends!")


if __name__ == "__main__":
    main()
