import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class VariableStarClassifier:
    def __init__(self):
        self.group_mapping = {
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
            "M": "Pulsating",
            "ROT": "Rot_Irr",
            "YSO": "Rot_Irr",
            "SR": "Rot_Irr",
            "L": "Rot_Irr",
            "VAR": "Rot_Irr",
            "RVA": "Rot_Irr"
        }

        self.feature_names = [
            "Period", "Amplitude", "Mean_gmag", 
            "LKSL_statistic", "bp_rp", "parallax_over_error"
        ]

        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    def prepare_data(self, df):
        # Apply mapping
        df["super_label"] = df["ML_classification"].map(self.group_mapping)
        print(df["super_label"].value_counts())

        # Features X and Labels y
        X = df[self.feature_names].values
        y = df["super_label"].values

        return df, X, y

    def train(self, X, y, test_size=0.3, random_state=42):
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Standardize/Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest Classifier
        self.clf.fit(X_train_scaled, y_train)

        # Prediction
        y_pred = self.clf.predict(X_test_scaled)

        return X_test_scaled, y_test, y_pred

    def evaluate(self, y_test, y_pred):
        print("-------------------------Classification Analysis-------------------------------------")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("-------------------------------------------------------------------------------------")

    def get_feature_importances(self):
        importances = self.clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        return importances, self.feature_names, indices
