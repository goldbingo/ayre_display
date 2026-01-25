# Project Notes

## Temporary Files
Use `.claude/tmp/` for all temporary files.

## Project: 7-Segment Display Reader
- Main script: `live_demo.py`
- Core library: `segment_reader.py`
- Logs directory: `logs/`
- Example images: `example/`
- Templates: `templates/`

## Running Live Demo
```bash
python3 live_demo.py --skip 1    # Log every frame
python3 live_demo.py --skip 3    # Log every 3rd frame
```

## Monitoring

### Active Monitors
| Monitor | Command | Alerts |
|---------|---------|--------|
| live_demo.py | `python3 live_demo.py --skip 1` | Real-time display |
| Hourly summary | Cron @ :00 | iMessage |

### Instant iMessage Alerts
- **LED FAIL** - LED detection failed
- **MUTE_NA** - Abnormal MUTE pixel count
- **LED GLITCH** - B1/B2 flicker detected
- **DIGIT 1 LOW** - Digit "1" low confidence with "7" close (penalty issue)

### Logging
- `logs/detection.csv` - Detection data (every frame)
- `logs/*_issue_*.png` - Issue frame captures
- `logs/hourly_summary.log` - Cron output

## Analysis Tools
```bash
python3 analyze_skip.py              # Skip analysis by hour
python3 hourly_summary.py --test     # Test summary (no send)
python3 hourly_summary.py --now      # Send current hour summary
```

## Frame Skip Optimization
- Threshold: 200,000 (video noise ~82K-199K, content change 2-5M)
- Compares ROI to reference frame to detect changes
- ~97% skip rate when content stable
- Edge cases (150K-300K) logged in `diff_edge` column

## Digit "1" Penalty
- Templates matching on left side (<30% of width) get 30% penalty
- Prevents false matches from 0/6/8/P left bars
- Template `digit_1g.png` matches on right side (no penalty)

## Manual Template Learning
In live_demo.py, use keyboard shortcuts to save new templates:
- `l#` - Save left digit as # (e.g., `l6` saves left digit as "6")
- `r#` - Save right digit as # (e.g., `r8` saves right digit as "8")
- Templates saved to `templates/digit_#X.png` with next available letter

## Key Files
| File | Purpose |
|------|---------|
| `live_demo.py` | Main monitoring script |
| `segment_reader.py` | Core detection library |
| `analyze_skip.py` | Skip analysis tool |
| `hourly_summary.py` | Hourly iMessage reports |
| `.claude/notify_config.json` | iMessage config |
