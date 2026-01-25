#!/usr/bin/env python3
"""Send hourly skip analysis summary via iMessage."""

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, ".claude", "notify_config.json")
LOG_PATH = os.path.join(SCRIPT_DIR, "logs", "detection.csv")


def load_config():
    """Load notification config."""
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except:
        return {}


def send_imessage(recipient, message):
    """Send iMessage via AppleScript."""
    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{recipient}" of targetService
        send "{message}" to targetBuddy
    end tell
    '''
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
        return True
    except:
        return False


def analyze_last_hour():
    """Analyze the last hour of logs."""
    now = datetime.now()
    last_hour = (now - timedelta(hours=1)).strftime('%Y-%m-%d %H')

    stats = {
        'total': 0,
        'skipped': 0,
        'edge_cases': [],
        'readings': set(),
        'issues': [],
    }

    try:
        with open(LOG_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get('timestamp', '')
                if not ts.startswith(last_hour):
                    continue

                stats['total'] += 1

                # Reading
                reading = row.get('reading', '')
                if reading:
                    stats['readings'].add(reading)

                # Skip status
                if row.get('frame_skip', '') == '1':
                    stats['skipped'] += 1

                # Edge cases
                diff_edge = row.get('diff_edge', '')
                if diff_edge:
                    try:
                        stats['edge_cases'].append(int(diff_edge))
                    except:
                        pass

                # Issues
                issue = row.get('issue', '')
                if issue:
                    stats['issues'].append(issue)
    except:
        pass

    stats['hour'] = last_hour
    return stats


def format_summary(stats):
    """Format summary for iMessage."""
    if stats['total'] == 0:
        return f"[{stats['hour']}] No data"

    skip_rate = 100 * stats['skipped'] / stats['total']
    readings = ', '.join(sorted(stats['readings'])) or 'none'

    lines = [
        f"[{stats['hour']}]",
        f"Frames: {stats['total']:,} | Skip: {skip_rate:.0f}%",
        f"Readings: {readings}",
    ]

    # Edge case summary
    edge = stats['edge_cases']
    if edge:
        above_thresh = len([d for d in edge if d >= 200000])
        near_thresh = len([d for d in edge if 195000 <= d <= 205000])
        lines.append(f"Edge: {len(edge)} ({above_thresh} processed, {near_thresh} near)")

    # Issues
    if stats['issues']:
        issue_counts = {}
        for iss in stats['issues']:
            issue_counts[iss] = issue_counts.get(iss, 0) + 1
        issue_str = ', '.join(f"{k}:{v}" for k, v in issue_counts.items())
        lines.append(f"Issues: {issue_str}")
    else:
        lines.append("Issues: none")

    return '\n'.join(lines)


def analyze_current_hour():
    """Analyze the current hour of logs (for testing)."""
    current_hour = datetime.now().strftime('%Y-%m-%d %H')

    stats = {
        'total': 0,
        'skipped': 0,
        'edge_cases': [],
        'readings': set(),
        'issues': [],
    }

    try:
        with open(LOG_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get('timestamp', '')
                if not ts.startswith(current_hour):
                    continue

                stats['total'] += 1

                # Reading
                reading = row.get('reading', '')
                if reading:
                    stats['readings'].add(reading)

                # Skip status
                if row.get('frame_skip', '') == '1':
                    stats['skipped'] += 1

                # Edge cases
                diff_edge = row.get('diff_edge', '')
                if diff_edge:
                    try:
                        stats['edge_cases'].append(int(diff_edge))
                    except:
                        pass

                # Issues
                issue = row.get('issue', '')
                if issue:
                    stats['issues'].append(issue)
    except:
        pass

    stats['hour'] = current_hour
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Send hourly skip analysis summary')
    parser.add_argument('--test', action='store_true', help='Test mode: analyze current hour, print only')
    parser.add_argument('--now', action='store_true', help='Analyze current hour instead of last hour')
    args = parser.parse_args()

    config = load_config()
    recipient = config.get('imessage_recipient')

    if not recipient and not args.test:
        print("Error: No iMessage recipient configured")
        sys.exit(1)

    if args.now or args.test:
        stats = analyze_current_hour()
    else:
        stats = analyze_last_hour()

    summary = format_summary(stats)
    print(summary)
    print()

    if args.test:
        print("(Test mode - not sending)")
    elif send_imessage(recipient, summary):
        print(f"Sent to {recipient}")
    else:
        print("Failed to send iMessage")
        sys.exit(1)


if __name__ == '__main__':
    main()
