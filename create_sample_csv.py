import csv
from timestamps import get_current_timestamp

def create_sample_csv(path="sample_heart_rate.csv"):
    rows = [
        {"timestamp": get_current_timestamp(), "heart_rate": 72},
        {"timestamp": get_current_timestamp(), "heart_rate": 80},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "heart_rate"])
        writer.writeheader()
        writer.writerows(rows)