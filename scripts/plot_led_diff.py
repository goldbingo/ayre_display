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
for c in ['max_diff','changed','B1_diff','B2_diff','S1_diff','S2_diff','threshold']:
    led[c] = pd.to_numeric(led[c], errors='coerce')
led['timestamp'] = pd.to_datetime(led['timestamp'], errors='coerce')
led = led[led['max_diff'].notna() & led['timestamp'].notna()]
if args.minutes is not None:
    led = led[led['timestamp'] >= led['timestamp'].max() - pd.Timedelta(minutes=args.minutes)]

t = led['timestamp']
changed = led['changed'] == 1
# Resnap reason from CSV: 'threshold', 'cooldown', 'change', 'drift', or ''
resnap = led['resnap'].fillna('').astype(str).str.strip()
resnap_any = (resnap != '') & (resnap != 'nan') & (resnap != '0')
resnap_thresh = resnap == 'threshold'
resnap_cooldown = resnap == 'cooldown'
resnap_change = resnap == 'change'
resnap_drift = resnap == 'drift'

ax = axes[0]
# Layer 1: dots — change status
normal = ~changed & ~resnap_any
ax.scatter(t[normal], led['max_diff'][normal], s=3, c='steelblue', alpha=0.4, label=f'unchanged ({normal.sum()})')
if changed.any():
    ax.scatter(t[changed], led['max_diff'][changed], s=40, c='red', alpha=0.5, zorder=5, label=f'LED changed ({changed.sum()})')
# Layer 2: triangles — resnap type (on top of dots)
if resnap_thresh.any():
    ax.scatter(t[resnap_thresh], led['max_diff'][resnap_thresh], s=60, c='orange', alpha=0.5,
               marker='^', edgecolors='black', linewidths=0.3, zorder=6, label=f'resnap:thresh ({resnap_thresh.sum()})')
if resnap_cooldown.any():
    ax.scatter(t[resnap_cooldown], led['max_diff'][resnap_cooldown], s=60, c='yellow', alpha=0.5,
               marker='s', edgecolors='black', linewidths=0.3, zorder=6, label=f'resnap:cooldown ({resnap_cooldown.sum()})')
if resnap_change.any():
    ax.scatter(t[resnap_change], led['max_diff'][resnap_change], s=60, c='lime', alpha=0.5,
               marker='v', edgecolors='black', linewidths=0.3, zorder=7, label=f'resnap:change ({resnap_change.sum()})')
if resnap_drift.any():
    ax.scatter(t[resnap_drift], led['max_diff'][resnap_drift], s=40, c='cyan', alpha=0.5,
               marker='d', edgecolors='black', linewidths=0.3, zorder=6, label=f'resnap:drift ({resnap_drift.sum()})')
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
    clipped_changed = clipped_mask & changed
    clipped_resnap = clipped_mask & resnap_any & ~changed
    clipped_normal = clipped_mask & ~changed & ~resnap_any
    if clipped_normal.any():
        ax.scatter(t[clipped_normal], [y_clip]*clipped_normal.sum(), s=3, c='steelblue', alpha=0.4)
    if clipped_changed.any():
        ax.scatter(t[clipped_changed], [y_clip]*clipped_changed.sum(), s=80, c='red', zorder=5)
    if clipped_resnap.any():
        ax.scatter(t[clipped_resnap], [y_clip]*clipped_resnap.sum(), s=50, c='orange', alpha=0.8, marker='^', zorder=6)
    for _, row in led[clipped_mask].iterrows():
        ax.annotate(f'{row["max_diff"]:.0f}', xy=(row['timestamp'], y_clip), xytext=(8, 0),
                    textcoords='offset points', fontsize=7, color='red', ha='left', va='bottom', fontweight='bold')
# Skip ratio: exclude change-triggered overhead (threshold+change and their cooldown)
# A "miss" is not real if the next frame has resnap (leading-edge transition)
missed_mask = changed & ~resnap_any
# Remove leading-edge: missed at idx but idx+1 has resnap (LED already detected on miss frame,
# +1 resnap confirms the change was caught — changed may be 0 since LED already settled)
for idx in led[missed_mask].index:
    if idx + 1 in led.index:
        next_r = str(led.loc[idx + 1, 'resnap']).strip()
        if next_r not in ('', 'nan', '0'):
            missed_mask.iloc[missed_mask.index.get_loc(idx)] = False
real_missed = missed_mask.sum()
leading = (changed & ~resnap_any).sum() - real_missed
total_changes = changed.sum()
thresh_with_change = ((resnap_thresh | resnap_drift) & changed).sum()
cooldown_after_change = resnap_cooldown.sum()
skip_pct = 100 * (len(led) - thresh_with_change - cooldown_after_change) / len(led) if len(led) > 0 else 0
# Count cooldown-saved changes by offset from threshold resnap
caught_cooldown = changed & resnap_cooldown
cooldown_offsets = {}
for idx in led[caught_cooldown].index:
    for back in range(1, 20):
        if idx - back >= 0 and str(led.loc[idx - back, 'resnap']).strip() in ('threshold', 'drift'):
            cooldown_offsets[back] = cooldown_offsets.get(back, 0) + 1
            break
cd_detail = ''
if cooldown_offsets:
    cd_detail = ' cd:' + '/'.join(f'+{k}:{v}' for k, v in sorted(cooldown_offsets.items()))
miss_str = f'{real_missed} miss'
if leading > 0:
    miss_str += f' +{leading} lead'
if real_missed > 0:
    miss_str += '!'
title = f'LED diff: {len(led)} frames, skip {skip_pct:.1f}%, {total_changes} chg{cd_detail}, {miss_str})'
if real_missed > 0:
    for _, row in led[missed_mask].iterrows():
        ax.annotate(f'MISSED\n{row["max_diff"]:.1f}',
                    xy=(row['timestamp'], row['max_diff']),
                    xytext=(30, 30), textcoords='offset points',
                    fontsize=8, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
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
            ax1b.annotate(f'{row[col]:.0f}', xy=(row['timestamp'], y_clip), xytext=(8, 0),
                          textcoords='offset points', fontsize=7, color=color, ha='left', va='bottom', fontweight='bold')
ax1b.set_ylabel('zone diff')
ax1b.set_title('LED diff: per zone (B1/B2/S1/S2)')
ax1b.legend(fontsize=8, loc='upper left')
ax1b.set_xlabel('time')
ax1b.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.show()
