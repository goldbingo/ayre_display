#!/bin/sh
# watchdog.sh - FreeBSD compatible
# Monitors live_demo.py and restarts if it hangs (no heartbeat for TIMEOUT seconds)

HEARTBEAT="/tmp/live_demo_heartbeat"
TIMEOUT=120
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

while true; do
    # Remove stale heartbeat
    rm -f "$HEARTBEAT"

    # Start live_demo in background
    python live_demo.py --target-fps 1.15 --log --mqtt-config mqtt_config.json&
    PID=$!
    echo "$(date): Started live_demo.py (PID $PID)"

    while kill -0 $PID 2>/dev/null; do
        sleep 30

        # Check heartbeat file age
        if [ -f "$HEARTBEAT" ]; then
            NOW=$(date +%s)
            MTIME=$(stat -f %m "$HEARTBEAT" 2>/dev/null || stat -c %Y "$HEARTBEAT" 2>/dev/null)
            AGE=$((NOW - MTIME))

            if [ $AGE -gt $TIMEOUT ]; then
                echo "$(date): No heartbeat for ${AGE}s, killing PID $PID..."
                kill -9 $PID 2>/dev/null
                break
            fi
        else
            # No heartbeat file yet - give it time to start
            :
        fi
    done

    echo "$(date): Process exited, restarting in 2s..."
    sleep 2
done
