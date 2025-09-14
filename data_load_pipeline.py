from load_csv import load_heart_rate
from load_json import load_fitness_data

def load_all():
    hr = load_heart_rate()
    fitness = load_fitness_data()
    return {"heart_rate": hr, "fitness": fitness}

if _name_ == "_main_":
    data = load_all()
    print("Loaded data:", data)