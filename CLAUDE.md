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

## Quick Reference

### Running Live Demo
```bash
python3 live_demo.py                          # Default mode
python3 live_demo.py --hwdec --gop-decode     # Low CPU with display
python3 live_demo.py --headless               # No display window
python3 live_demo.py --skip 15 --headless     # Lowest CPU
```

### Virtual Environment
```bash
.venv/bin/python live_demo.py                 # Use venv for cv2
```

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
