from datetime import timedelta
from timestamps import parse_timestamp

def detect_gaps(data, threshold_minutes=5):
    """Find timestamps where the gap is larger than threshold."""
    anomalies = []
    for i in range(1, len(data)):
        prev = parse_timestamp(data[i-1]["timestamp"])
        curr = parse_timestamp(data[i]["timestamp"])
        if curr - prev > timedelta(minutes=threshold_minutes):
            anomalies.append((prev, curr))
    return anomalies