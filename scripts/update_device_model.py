#!/usr/bin/env python3
"""
Compute device_model.json offsets from camera_mount.json measurements.

Reads the 7 measured pixel positions (corner, panel top-left, B1, B2, S1, S2,
mute center) from camera_mount.json, computes the relative offsets, and updates
device_model.json.  All other fields (thresholds, ratios, etc.) are preserved.

Usage:
    python scripts/update_device_model.py
    python scripts/update_device_model.py --dry-run   # show changes without writing
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MOUNT_PATH = os.path.join(PROJECT_ROOT, 'calibration', 'camera_mount.json')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'calibration', 'device_model.json')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    """Write JSON with compact short arrays (matching original style)."""
    import re
    text = json.dumps(data, indent=4)
    # Collapse short numeric arrays onto one line
    V = r'[-\d.e]+'  # number token
    text = re.sub(rf'\[\s*\n\s+({V}),\s*\n\s+({V})\s*\n\s*\]',
                  r'[\1, \2]', text)
    text = re.sub(rf'\[\s*\n\s+({V}),\s*\n\s+({V}),\s*\n\s+({V})\s*\n\s*\]',
                  r'[\1, \2, \3]', text)
    text = re.sub(rf'\[\s*\n\s+({V}),\s*\n\s+({V}),\s*\n\s+({V}),\s*\n\s+({V})\s*\n\s*\]',
                  r'[\1, \2, \3, \4]', text)
    with open(path, 'w') as f:
        f.write(text)
        f.write('\n')


def main():
    dry_run = '--dry-run' in sys.argv

    # Load inputs
    mount = load_json(MOUNT_PATH)
    model = load_json(MODEL_PATH)

    corner = mount['corner_xy']
    cx, cy = corner

    # --- Validate required fields ---
    required_buttons = ['B1', 'B2', 'S1', 'S2']
    buttons = mount['button_centers']
    missing = [b for b in required_buttons if b not in buttons]
    if missing:
        print(f"ERROR: camera_mount.json missing button_centers: {missing}")
        sys.exit(1)

    if 'mute_center' not in mount:
        print("ERROR: camera_mount.json missing 'mute_center'")
        sys.exit(1)

    panel_rect = mount['panel_rect']  # [x, y, w, h]
    mute_center = mount['mute_center']

    # --- Compute offsets ---
    new_panel_offset = [panel_rect[0] - cx, panel_rect[1] - cy]
    new_mute_offset = [mute_center[0] - cx, mute_center[1] - cy]
    new_landmarks = {'corner': [0.0, 0.0]}
    for name in required_buttons:
        bx, by = buttons[name]
        new_landmarks[name] = [float(bx - cx), float(by - cy)]

    # --- Print summary ---
    print(f"Source: {MOUNT_PATH}")
    print(f"Target: {MODEL_PATH}")
    print(f"Corner: ({cx}, {cy})")
    print()

    changes = []

    def show(field, old_val, new_val):
        changed = old_val != new_val
        marker = ' *' if changed else ''
        print(f"  {field:25s}  {str(old_val):>20s}  ->  {str(new_val):<20s}{marker}")
        if changed:
            changes.append(field)

    print(f"{'Field':>27s}  {'Old':>20s}      {'New':<20s}")
    print(f"  {'-'*25}  {'-'*20}      {'-'*20}")

    show('panel_offset', model['panel_offset'], new_panel_offset)
    show('mute_button_offset', model['mute_button_offset'], new_mute_offset)

    old_landmarks = model.get('landmarks', {})
    for name in ['corner'] + required_buttons:
        old_val = old_landmarks.get(name, 'missing')
        new_val = new_landmarks[name]
        show(f'landmarks.{name}', old_val, new_val)

    print()
    if not changes:
        print("No changes needed — device_model.json already matches.")
        return

    if dry_run:
        print(f"Dry run: {len(changes)} field(s) would change.")
        return

    # --- Update model ---
    model['panel_offset'] = new_panel_offset
    model['mute_button_offset'] = new_mute_offset
    model['landmarks'] = new_landmarks

    save_json(MODEL_PATH, model)
    print(f"Updated {len(changes)} field(s) in device_model.json.")


if __name__ == '__main__':
    main()
