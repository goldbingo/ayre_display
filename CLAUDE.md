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
When user says "run live", execute:
```bash
pkill -9 -f "live_demo.py"; sleep 1; .venv/bin/python live_demo.py --display --log
```
- Kill old instance first
- Default: headless, no logging, drain 2
- Use `--display` to show window
- Use `--log` to enable logging

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
