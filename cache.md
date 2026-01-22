# Caching Strategy

## Background

The camera angle, position, and environment lighting are not always fixed. When the major and even fallback detection methods both fail, the last known good detected window locations are valuable. This data may provide a chance to keep the major digit recognition, LED, and MUTE detection functioning well.

## Cache Types

### Runtime Cache

Good indicators are crucial to decide whether the cache needs to be updated:
- Pattern match confidence levels
- More landmarks identified with high confidence

In addition to indicators, **variance** is important. The variance of the indicators and the detected window locations shows how good the detection is going. Only when indicators are good and variances are small should we consider updating the cache.

### File Cache

- File I/O activity should be controlled
- Compare difference between runtime cache and file cache
- Only update file cache when the difference is large enough
- On application startup, use file cache to restore the last known good state

## When to Use Cached Data

### Use Case 1: Backup

When frame-by-frame detection and its major/fallback methods all provide bad results:
- Confidence low
- Buttons not detected

Use cached values to provide window, gap, and zone locations.

### Use Case 2: Performance

Since camera angle, position, and lighting don't change quickly:
- Skip location detection, only do digit/LED recognition
- Reduces computational load
- Benefits low power scenarios

A user option to choose which use case is preferred.

## Analysis Before Implementation

Since lighting and panel digits change throughout the day:
- Log indicators and locations over 24 hours
- Helps determine thresholds for cache updates
- Helps decide when to use cached values vs frame-by-frame detection
- Capture frames with issues for debugging
