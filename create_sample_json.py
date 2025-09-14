import json
from timestamps import get_current_timestamp

def create_sample_json(path="sample_fitness_data.json"):
    data = [
        {"timestamp": get_current_timestamp(), "steps": 10, "sleep": 0},
        {"timestamp": get_current_timestamp(), "steps": 25, "sleep": 0},
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)