background:
    The camera angle, position, environment lighting are not always fixed. when the major and even fall back detection methold both failed, the last known good detected window locations are valuable. The data may provide a chance to keep the major digit recongnition, LED and MUTE detection to function well.

cache types and update considerations:
    run time cache:
        Good indicators is crutial to decide wether the cache needs to be update, such as attern match confidenance levels, more landmarks identitied with high confidence. In addition to indicators, variance is important. The varience of the indicators and the detected window locations shows how good the detection is going. Only when indicators are good and variences are small, we may consider to update the cache.

    file cache:
        File I/O activity should be controlled. By comparing difference betweenn the run time cache and the tile cache, only update file cache when the difference is large enough.
        When application start up, use the file cache to restore the last known good state.

when to use cached data instead of from each frame:
    Two different use cases:
        Back up: When using frame by frame detection and its major and fallback methods all provide bad result(confedence low, buttons not detected), the use the cached ones to provide the window, gap, zone locations.
        Performance: Since the camera angle, position and lighting are not change so fast, skipping the location detections and only do necessary digit or LED recongnition can help the performace. Reduce the location detection rate will also benefit low power.

analysis before implementation:
    Since the lighting and the digits of panel are changing, logging of the indicators, locations covers 24 hours will help you to decide the thresholds of updating cache and when to use cached value without detection frane by frame. 
