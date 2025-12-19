import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "water_cluster_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

def predict_cluster(user_input):
    annual = float(user_input["annual"])
    groundwater = float(user_input["groundwater"])
    rice = float(user_input["rice"])
    wheat = float(user_input["wheat"])
    windspeed = float(user_input["windspeed"])
    sugarcane = float(user_input["sugarcane"])
    dugwell_count = float(user_input["dugwell_count"])

    features_21 = [
        annual,
        annual * 0.15,
        annual * 0.25,
        annual * 0.45,
        annual * 0.15,
        groundwater,
        groundwater * 1.05,
        dugwell_count * 0.8,
        groundwater * 1.2,
        annual * 0.10,
        annual * 0.15,
        annual * 0.12,
        annual * 0.08,
        windspeed * 0.9,
        windspeed * 1.0,
        windspeed * 1.1,
        rice,
        wheat,
        (rice + wheat) * 0.3,
        sugarcane,
        (rice + sugarcane) * 0.2
    ]

    X = np.array(features_21).reshape(1, -1)
    X_scaled = scaler.transform(X)
    cluster = model.predict(X_scaled)[0]

    return int(cluster)
