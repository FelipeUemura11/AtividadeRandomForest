import pickle
from pathlib import Path

# Carrega o scaler
scaler_path = Path("models/scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

print("Feature names in scaler:")
print(scaler.feature_names_in_)

print("\nScaler type:", type(scaler))
print("Scaler parameters:", scaler.get_params()) 