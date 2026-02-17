import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot LED diff experiment data')
parser.add_argument('-m', '--minutes', type=float, default=None, help='Show last N minutes (default: all)')
args = parser.parse_args()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# --- LED diff: max ---
led = pd.read_csv('/Volumes/ExtData/proj/claude/logs/led_diff_experiment.csv', on_bad_lines='skip')
for c in ['max_diff','changed','resnap','B1_diff','B2_diff','S1_diff','S2_diff','threshold']:
    led[c] = pd.to_numeric(led[c], errors='coerce')
led['timestamp'] = pd.to_datetime(led['timestamp'], errors='coerce')
led = led[led['max_diff'].notna() & led['timestamp'].notna()]
if args.minutes is not None:
    led = led[led['timestamp'] >= led['timestamp'].max() - pd.Timedelta(minutes=args.minutes)]

t = led['timestamp']
changed = led['changed'] == 1
resnap = led['resnap'] == 1
resnap_thresh = resnap & (led['max_diff'] >= led['threshold'])   # resnap triggered by threshold
resnap_change = resnap & (led['max_diff'] < led['threshold'])   # resnap triggered by drift/other
normal = ~changed & ~resnap

ax = axes[0]
# Layer 1: dots — change status
ax.scatter(t[normal], led['max_diff'][normal], s=3, c='steelblue', alpha=0.4, label=f'unchanged ({normal.sum()})')
if changed.any():
    ax.scatter(t[changed], led['max_diff'][changed], s=40, c='red', zorder=5, label=f'LED changed ({changed.sum()})')
# Layer 2: triangles — resnap type (on top of dots)
if resnap_thresh.any():
    ax.scatter(t[resnap_thresh], led['max_diff'][resnap_thresh], s=60, c='orange', alpha=0.8,
               marker='^', edgecolors='black', linewidths=0.3, zorder=6, label=f'resnap:thresh ({resnap_thresh.sum()})')
if resnap_change.any():
    ax.scatter(t[resnap_change], led['max_diff'][resnap_change], s=60, c='lime', alpha=0.8,
               marker='v', edgecolors='black', linewidths=0.3, zorder=7, label=f'resnap:change ({resnap_change.sum()})')
# Plot threshold line (follows changes over time)
if led['threshold'].notna().any():
    ax.plot(t, led['threshold'], color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='threshold', drawstyle='steps-post')
else:
    ax.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='threshold=5')
ax.set_ylabel('max_diff')
thresh_max = led['threshold'].max() if led['threshold'].notna().any() else 5.0
y_clip = thresh_max * 2
clipped_mask = led['max_diff'] > y_clip
if clipped_mask.any():
    ax.set_ylim(bottom=0, top=y_clip * 1.05)
else:
    ax.set_ylim(bottom=0, top=max(thresh_max, led['max_diff'].max()) * 1.1)
if clipped_mask.any():
    clipped_t = led.loc[clipped_mask, 'timestamp']
    clipped_v = led.loc[clipped_mask, 'max_diff']
    # Show markers at clip line
    clipped_changed = clipped_mask & changed
    clipped_resnap = clipped_mask & resnap & ~changed
    clipped_normal = clipped_mask & ~changed & ~resnap
    if clipped_normal.any():
        ax.scatter(t[clipped_normal], [y_clip]*clipped_normal.sum(), s=3, c='steelblue', alpha=0.4)
    if clipped_changed.any():
        ax.scatter(t[clipped_changed], [y_clip]*clipped_changed.sum(), s=80, c='red', zorder=5)
    if clipped_resnap.any():
        ax.scatter(t[clipped_resnap], [y_clip]*clipped_resnap.sum(), s=50, c='orange', alpha=0.8, marker='^', zorder=6)
    for _, row in led[clipped_mask].iterrows():
        ax.annotate(f'{row["max_diff"]:.0f}', xy=(row['timestamp'], y_clip), fontsize=7,
                    color='red', ha='center', va='bottom', fontweight='bold')
missed_mask = changed & (led['max_diff'] < led['threshold'])
missed = missed_mask.sum()
total_changes = changed.sum()
title = f'LED diff: max ({len(led)} frames, {total_changes} changes'
if missed > 0:
    title += f', {missed} missed by thresh!'
    for _, row in led[missed_mask].iterrows():
        ax.annotate(f'MISSED\n{row["max_diff"]:.1f}',
                    xy=(row['timestamp'], row['max_diff']),
                    xytext=(30, 30), textcoords='offset points',
                    fontsize=8, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
else:
    title += ', 0 missed'
title += ')'
ax.set_title(title)
ax.legend(fontsize=8, loc='upper left')

# --- LED diff: individual zones ---
ax1b = axes[1]
colors = {'B1': 'tab:blue', 'B2': 'tab:orange', 'S1': 'tab:green', 'S2': 'tab:red'}
for zone, color in colors.items():
    col = f'{zone}_diff'
    valid = led[col].notna()
    ax1b.scatter(t[valid & ~changed], led[col][valid & ~changed], s=2, c=color, alpha=0.3, label=zone)
if changed.any():
    for zone, color in colors.items():
        col = f'{zone}_diff'
        valid = led[col].notna() & changed
        if valid.any():
            ax1b.scatter(t[valid], led[col][valid], s=60, c=color, edgecolors='black', linewidths=0.5, zorder=5)
if led['threshold'].notna().any():
    ax1b.plot(t, led['threshold'], color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='threshold', drawstyle='steps-post')
else:
    ax1b.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='threshold=5')
zone_max = max(led[f'{z}_diff'].max() for z in colors if led[f'{z}_diff'].notna().any())
if zone_max > y_clip:
    ax1b.set_ylim(bottom=0, top=y_clip * 1.05)
else:
    ax1b.set_ylim(bottom=0, top=max(thresh_max, zone_max) * 1.1)
# Show clipped zone values
for zone, color in colors.items():
    col = f'{zone}_diff'
    zone_clipped = led[col].notna() & (led[col] > y_clip)
    if zone_clipped.any():
        ax1b.scatter(t[zone_clipped], [y_clip]*zone_clipped.sum(), s=10, c=color, zorder=5)
        for _, row in led[zone_clipped].iterrows():
            ax1b.annotate(f'{row[col]:.0f}', xy=(row['timestamp'], y_clip), fontsize=7,
                          color=color, ha='center', va='bottom', fontweight='bold')
ax1b.set_ylabel('zone diff')
ax1b.set_title('LED diff: per zone (B1/B2/S1/S2)')
ax1b.legend(fontsize=8, loc='upper left')
ax1b.set_xlabel('time')
ax1b.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.show()
