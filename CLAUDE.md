# Project Notes

## Temporary Files
Use `.claude/tmp/` for all temporary files.

## Project: 7-Segment Display Reader
- Main script: `live_demo.py`
- Core library: `segment_reader.py`
- Design docs: `DESIGN.md`
- Logs directory: `logs/`
- Example images: `example/`
- Templates: `templates/`

## User Preferences

### "run live" command
When user says "run live", execute in background:
```bash
pkill -9 -f "live_demo.py"; sleep 1; .venv/bin/python live_demo.py --display --log &
```
- Always run in background (don't block terminal)
- Kill old instance first
- Default: headless, no logging, drain 2
- Use `--display` to show window
- Use `--log` to enable logging

### Before Git Commit (MANDATORY)
1. Run `python segment_reader.py` - **ALL** example/ images must pass
   - Do NOT use `head` or partial output - verify ALL images
   - Expected: 2 XX results (edge case images), rest must match filename
2. Update DESIGN.md if any features changed

### When Changing CSV Log Format
When adding/removing fields in `log_detection()` or changing the CSV header:
1. Archive old CSV: `mv logs/detection.csv logs/detection_archived_YYYYMMDD.csv`
2. New `detection.csv` will be created with correct header on next run
3. This prevents header/data mismatch in the CSV file

### Manual Template Learning
- `l#` - Save left digit as # (e.g., `l6`)
- `r#` - Save right digit as # (e.g., `r8`)

## Key Files
| File | Purpose |
|------|---------|
| `live_demo.py` | Main monitoring script |
| `segment_reader.py` | Core detection library |
| `DESIGN.md` | Full design documentation |
| `.claude/notify_config.json` | iMessage config |

