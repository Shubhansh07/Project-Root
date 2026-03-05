import os
print(os.listdir())

import pandas as pd
import numpy as np


flow = pd.read_csv(
    r"C:\Users\shubh\Downloads\internship\internship\Data\AP02\Flow  - 30.05.2024.txt",
    sep=";",
    skiprows=7,
    names=["time","flow"]
)

flow["time"] = pd.to_datetime(
    flow["time"],
    format="%d.%m.%Y %H:%M:%S,%f"
)

print("Flow sample:")
print(flow.head())


spo2 = pd.read_csv(
    r"C:\Users\shubh\Desktop\Healthcare_Irrerugalities\SPO2  - 30.05.2024.txt",
    sep=";",
    skiprows=7,
    names=["time","spo2"]
)

spo2["time"] = pd.to_datetime(
    spo2["time"],
    format="%d.%m.%Y %H:%M:%S,%f"
)

print("SpO2 sample:")
print(spo2.head())


thorac = pd.read_csv(
    r"C:\Users\shubh\Desktop\Healthcare_Irrerugalities\Thorac  - 30.05.2024.txt",
    sep=";",
    skiprows=7,
    names=["time","thorac"]
)

thorac["time"] = pd.to_datetime(
    thorac["time"],
    format="%d.%m.%Y %H:%M:%S,%f"
)

print("Thoracic sample:")
print(thorac.head())


events = pd.read_csv(
    "Flow Events - 29_05_2024.txt",
    sep=";",
    skiprows=3,
    names=["time_range","duration","event","stage"]
)

events = events.dropna(subset=["duration"])

print("Event sample:")
print(events.head())


events[["start","end"]] = events["time_range"].str.split("-",expand=True)

events["start"] = pd.to_datetime(
    events["start"],
    format="%d.%m.%Y %H:%M:%S,%f"
)


events["end"] = pd.to_datetime(
    events["start"].dt.strftime("%d.%m.%Y") + " " + events["end"],
    format="%d.%m.%Y %H:%M:%S,%f"
)

print(events.head())

spo2_upsampled = np.repeat(spo2["spo2"].values, 8)

min_len = min(len(flow), len(spo2_upsampled), len(thorac))

flow = flow.iloc[:min_len]
spo2_upsampled = spo2_upsampled[:min_len]
thorac = thorac.iloc[:min_len]

data = pd.DataFrame({
    "time": flow["time"],
    "flow": flow["flow"],
    "spo2": spo2_upsampled,
    "thorac": thorac["thorac"]
})

print("Combined signal:")
print(data.head())
window_size = 960
windows = []

step = 160

for i in range(0, len(data) - window_size, step):
    windows.append(data.iloc[i:i+window_size])

print("Total windows:", len(windows))
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

print("Label examples:", labels[:20])
print(events["event"].value_counts())
def extract_features(window):

    flow_signal = window["flow"]
    spo2_signal = window["spo2"]
    thorac_signal = window["thorac"]

    return {
        "flow_mean": flow_signal.mean(),
        "flow_std": flow_signal.std(),
        "flow_max": flow_signal.max(),
        "flow_min": flow_signal.min(),
        "flow_range": flow_signal.max() - flow_signal.min(),

        "spo2_mean": spo2_signal.mean(),
        "spo2_min": spo2_signal.min(),

        "thorac_mean": thorac_signal.mean(),
        "thorac_std": thorac_signal.std(),
        "thorac_max": thorac_signal.max(),
        "thorac_min": thorac_signal.min(),
        "thorac_range": thorac_signal.max() - thorac_signal.min()
    }


feature_rows = []
for w in windows:
    feature_rows.append(extract_features(w))


dataset = pd.DataFrame(feature_rows)

dataset["label"] = labels

print(dataset.head())
print(dataset["label"].value_counts())
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

dataset["label"] = encoder.fit_transform(dataset["label"])

import joblib
joblib.dump(encoder, "label_encoder.pkl")
print(encoder.classes_)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


X = dataset.drop("label", axis=1)
y = dataset["label"]

from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
import joblib

joblib.dump(model, "sleep_apnea_model.pkl")

print("Model saved successfully")
dataset.to_csv("patient3_dataset.csv", index=False)
