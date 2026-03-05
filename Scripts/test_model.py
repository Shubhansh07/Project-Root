
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
model = joblib.load("sleep_apnea_model.pkl")
encoder = joblib.load("label_encoder.pkl")
print("Model loaded successfully")
flow = pd.read_csv(
    r"C:\Users\shubh\Downloads\internship\internship\Data\AP05\Flow Nasal - 28.05.2024.txt",
    sep=";",
    skiprows=7,
    names=["time","flow"],
    on_bad_lines="skip"
)

flow["time"] = pd.to_datetime(
    flow["time"],
    format="%d.%m.%Y %H:%M:%S,%f",
    errors="coerce"
)
flow = flow.dropna(subset=["time"])
print(flow.head(10))

spo2 = pd.read_csv(
    "SPO2 - 28.05.2024.txt",
    sep=";",
    skiprows=7,
    names=["time","spo2"],
    on_bad_lines="skip"
)

spo2["time"] = pd.to_datetime(
    spo2["time"],
    format="%d.%m.%Y %H:%M:%S,%f",
    errors="coerce"
)

spo2 = spo2.dropna(subset=["time"])

thorac = pd.read_csv(
    "Thorac Movement - 28.05.2024.txt",
    sep=";",
    skiprows=7,
    names=["time","thorac"],
    on_bad_lines="skip"
)

thorac["time"] = pd.to_datetime(
    thorac["time"],
    format="%d.%m.%Y %H:%M:%S,%f",
    errors="coerce"
)

thorac = thorac.dropna(subset=["time"])

events = pd.read_csv(
    "Flow Events - 28.05.2024.txt",
    sep=";",
    skiprows=3,
    names=["time_range","duration","event","stage"]
)

events = events.dropna(subset=["duration"])

events[["start","end"]] = events["time_range"].str.split("-",expand=True)

events["start"] = pd.to_datetime(events["start"], format="%d.%m.%Y %H:%M:%S,%f")

events["end"] = pd.to_datetime(
    events["start"].dt.strftime("%d.%m.%Y")+" "+events["end"],
    format="%d.%m.%Y %H:%M:%S,%f"
)
spo2_upsampled = np.repeat(spo2["spo2"].values,8)

min_len = min(len(flow),len(spo2_upsampled),len(thorac))

flow = flow.iloc[:min_len]
spo2_upsampled = spo2_upsampled[:min_len]
thorac = thorac.iloc[:min_len]

data = pd.DataFrame({
"time":flow["time"],
"flow":flow["flow"],
"spo2":spo2_upsampled,
"thorac":thorac["thorac"]
})
window_size = 960
step = 160

windows = []

for i in range(0, len(data) - window_size, step):
    windows.append(data.iloc[i:i+window_size])

    
    
labels = []

for w in windows:

    start = w["time"].iloc[0]
    end = w["time"].iloc[-1]

    label = "Normal"

    for _, e in events.iterrows():
        if start <= e["end"] and end >= e["start"]:
            label = e["event"]
            break

    labels.append(label)

labels = [
    "Obstructive Apnea" if l in ["Mixed Apnea", "Central Apnea"] else l
    for l in labels
]

# remove labels that the encoder doesn't know
labels = [l if l in encoder.classes_ else "Hypopnea" for l in labels]
def extract_features(window):

    flow = window["flow"]
    spo2 = window["spo2"]
    thorac = window["thorac"]

    return {
        "flow_mean":flow.mean(),
        "flow_std":flow.std(),
        "flow_max":flow.max(),
        "flow_min":flow.min(),
        "flow_range":flow.max()-flow.min(),

        "spo2_mean":spo2.mean(),
        "spo2_min":spo2.min(),

        "thorac_mean":thorac.mean(),
        "thorac_std":thorac.std(),
        "thorac_max":thorac.max(),
        "thorac_min":thorac.min(),
        "thorac_range":thorac.max()-thorac.min()
    }
features = []

for w in windows:
    features.append(extract_features(w))

X_test = pd.DataFrame(features)
y_true = encoder.transform(labels)
predictions = model.predict(X_test)
print(classification_report(y_true,predictions))
print(
    classification_report(
        y_true,
        predictions,
        target_names=encoder.classes_
    )
)
results = X_test.copy()
results["Actual"] = encoder.inverse_transform(y_true)
results["Predicted"] = encoder.inverse_transform(predictions)
print(results.head())
