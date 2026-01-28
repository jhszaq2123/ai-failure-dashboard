import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "temperature": np.random.normal(70, 10, n),
    "vibration": np.random.normal(0.5, 0.2, n),
    "pressure": np.random.normal(30, 5, n),
    "humidity": np.random.normal(40, 10, n),
    "load": np.random.normal(70, 15, n),
    "rpm": np.random.normal(1500, 200, n),
    "voltage": np.random.normal(220, 5, n)
})

# Prosta reguła awarii
data["failure"] = (
    (data["temperature"] > 85) |
    (data["vibration"] > 0.8) |
    (data["rpm"] > 1800)
).astype(int)

data.to_csv("data/sensor_data.csv", index=False)
print("Data generated: data/sensor_data.csv")
