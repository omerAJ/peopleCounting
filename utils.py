import datetime
import time
import db_utils

def point_position(x1, y1, x2, y2, px, py):
    """
    Determines whether a point (px, py) is above, below, or on the line
    defined by two points (x1, y1) and (x2, y2).

    Returns:
    "above" if the point is above the line,
    "below" if the point is below the line,
    "on" if the point is on the line.
    """

    # Check for vertical line
    if x2 - x1 == 0:
        if px < x1:
            return "left"
        elif px > x1:
            return "right"
        else:
            return "on"

    # Calculate slope (m) and y-intercept (c) for the line equation: y = mx + c
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    # Calculate y coordinate on the line corresponding to px
    line_y = m * px + c

    if px < x1 or px > x2:
        return "out"
    # Check the position of the point relative to the line
    if py > line_y:
        return "below"
    elif py < line_y:
        return "above"
    else:
        return "on"

def smooth_points(points, window_size=5):
    if len(points) < window_size:
        return points

    smoothed = []
    for i in range(len(points) - window_size + 1):
        chunk = points[i:i+window_size]
        avg_x = sum(x for x, y in chunk) / window_size
        avg_y = sum(y for x, y in chunk) / window_size
        smoothed.append((avg_x, avg_y))
    return smoothed

def add_reading_to_db(dbName, count_dict):
    
    reading = {
        "date": datetime.date.today().strftime("%Y-%m-%d"),
        "timeStamp": datetime.datetime.now().time().strftime("%H:%M:%S"),
    }
    reading.update(count_dict)  # merge the dictionaries
    db_utils.add_reading(dbName, reading)
    # last_db_update_time = time.time()  # update the last update time
    print(f'{dbName} dbUpdated')