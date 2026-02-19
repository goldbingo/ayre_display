#!/usr/bin/env python3
"""Analyze frame skip performance from detection logs - hourly summary."""

import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_log(log_path=os.path.join(_SCRIPT_DIR, '..', 'logs', 'detection.csv')):
    """Analyze skip performance grouped by hour."""

    hourly_stats = defaultdict(lambda: {
        'total': 0,
        'skipped': 0,
        'edge_cases': [],  # diff values in 150K-300K range
        'edge_skipped': 0,  # edge cases that were skipped
        'edge_processed': 0,  # edge cases that were processed
        'readings': set(),
    })

    try:
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get('timestamp', '')
                if not ts:
                    continue

                # Extract hour key (YYYY-MM-DD HH)
                try:
                    hour_key = ts[:13]  # "2026-01-25 16"
                except:
                    continue

                stats = hourly_stats[hour_key]
                stats['total'] += 1

                # Track readings
                reading = row.get('reading', '')
                if reading:
                    stats['readings'].add(reading)

                # Check skip status
                skipped = row.get('frame_skip', '') == '1'
                if skipped:
                    stats['skipped'] += 1

                # Check edge cases
                diff_edge = row.get('diff_edge', '')
                if diff_edge:
                    try:
                        diff_val = int(diff_edge)
                        stats['edge_cases'].append(diff_val)
                        if skipped:
                            stats['edge_skipped'] += 1
                        else:
                            stats['edge_processed'] += 1
                    except ValueError:
                        pass

    except FileNotFoundError:
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    return hourly_stats


def print_summary(hourly_stats):
    """Print hourly summary."""

    print("=" * 80)
    print("FRAME SKIP ANALYSIS - HOURLY SUMMARY")
    print("=" * 80)
    print()

    total_all = 0
    skipped_all = 0
    edge_all = []

    for hour_key in sorted(hourly_stats.keys()):
        stats = hourly_stats[hour_key]
        total = stats['total']
        skipped = stats['skipped']
        skip_rate = 100 * skipped / total if total > 0 else 0

        total_all += total
        skipped_all += skipped
        edge_all.extend(stats['edge_cases'])

        print(f"Hour: {hour_key}")
        print(f"  Frames: {total:,}  |  Skipped: {skipped:,} ({skip_rate:.1f}%)  |  Processed: {total - skipped:,}")
        print(f"  Readings seen: {', '.join(sorted(stats['readings'])) or 'none'}")

        # Edge case analysis
        edge_cases = stats['edge_cases']
        if edge_cases:
            print(f"  Edge cases (150K-300K): {len(edge_cases)}")
            print(f"    - Skipped: {stats['edge_skipped']}  |  Processed: {stats['edge_processed']}")
            print(f"    - Min: {min(edge_cases):,}  |  Max: {max(edge_cases):,}  |  Avg: {sum(edge_cases)//len(edge_cases):,}")

            # Flag potential issues
            false_process = [d for d in edge_cases if d < 200000 and d in stats['edge_cases']]
            near_threshold = [d for d in edge_cases if 195000 <= d <= 205000]
            if near_threshold:
                print(f"    ! Near threshold (195K-205K): {len(near_threshold)} cases")
        else:
            print(f"  Edge cases: none")

        print()

    # Overall summary
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    skip_rate_all = 100 * skipped_all / total_all if total_all > 0 else 0
    print(f"Total frames: {total_all:,}")
    print(f"Skipped: {skipped_all:,} ({skip_rate_all:.1f}%)")
    print(f"Processed: {total_all - skipped_all:,}")

    if edge_all:
        print(f"\nEdge cases total: {len(edge_all)}")
        print(f"  Range: {min(edge_all):,} - {max(edge_all):,}")

        # Distribution
        below_thresh = len([d for d in edge_all if d < 200000])
        above_thresh = len([d for d in edge_all if d >= 200000])
        print(f"  Below 200K (skipped): {below_thresh}")
        print(f"  Above 200K (processed): {above_thresh}")

        # Threshold recommendation
        if edge_all:
            p95 = sorted(edge_all)[int(len(edge_all) * 0.95)]
            p99 = sorted(edge_all)[int(len(edge_all) * 0.99)]
            print(f"\n  95th percentile: {p95:,}")
            print(f"  99th percentile: {p99:,}")

            if p95 > 200000:
                print(f"\n  ! WARNING: 95th percentile above threshold")
                print(f"    Consider raising threshold to {int(p95 * 1.1):,}")

    print()


if __name__ == '__main__':
    import os
    os.chdir('/Volumes/ExtData/proj/ayre_display')

    log_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/detection.csv'
    stats = analyze_log(log_path)
    print_summary(stats)
