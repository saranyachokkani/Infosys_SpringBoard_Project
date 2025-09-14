import csv

def load_heart_rate(path="sample_heart_rate.csv"):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))