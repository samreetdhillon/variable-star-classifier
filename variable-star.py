"""
Problem 5: Classifying a Variable Star (Optional)
Task: Take a light curve dataset.
Using basic statistics or ML (scikit-learn), classify it as pulsating, eclipsing, or irregular.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astroquery.vizier import Vizier
import lightkurve as lk
import requests, os

from lightkurve import search_lightcurve, read
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("asassn_variables_x.csv")

#define mapping directory
group_mapping = {
    "EA": "Eclipsing",
    "EB": "Eclipsing",
    "EW": "Eclipsing",
    "DSCT": "Pulsating",
    "HADS": "Pulsating",
    "RRAB": "Pulsating",
    "RRC": "Pulsating",
    "RRD": "Pulsating",
    "DCEP": "Pulsating",
    "DCEPS": "Pulsating",
    "CWA": "Pulsating",
    "CWB": "Pulsating",
    "M": "Pulsating",   # Mira
    "ROT": "Rot_Irr",
    "YSO": "Rot_Irr",
    "SR": "Rot_Irr",
    "L": "Rot_Irr",
    "VAR": "Rot_Irr",
    "RVA": "Rot_Irr"    # If present
}

# Apply Mapping
df["super_label"] = df["ML_classification"].map(group_mapping)
print(df["super_label"].value_counts())

#Features X and Labels y
X = df[["Period", "Amplitude", "Mean_gmag"]].values
y = df["super_label"].values

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Standardize/Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators = 200, class_weight = "balanced", n_jobs=-1, random_state=42)
clf.fit(X_train_scaled, y_train)

#Prediction and Accuracy Check
y_pred = clf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))