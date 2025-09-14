import json

def load_fitness_data(path="sample_fitness_data.json"):
    with open(path) as f:
        return json.load(f)