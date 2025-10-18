"""
Task: Take a light curve dataset and classify stars  as pulsating, eclipsing, or irregular
using basic statistics or ML (scikit-learn).
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

df = pd.read_csv("asassn_variables.csv")

#--------------------------Classifier--------------------------#

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
    "M": "Pulsating",   
    "ROT": "Rot_Irr",
    "YSO": "Rot_Irr",
    "SR": "Rot_Irr",
    "L": "Rot_Irr",
    "VAR": "Rot_Irr",
    "RVA": "Rot_Irr"  
}

# Apply Mapping
df["super_label"] = df["ML_classification"].map(group_mapping)
print(df["super_label"].value_counts())

#Features X and Labels y
X = df[["Period", "Amplitude", "Mean_gmag", "LKSL_statistic", "bp_rp", "parallax_over_error"]].values
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

# Get feature importances from trained model
importances = clf.feature_importances_
feature_names = ["Period", "Amplitude", "Mean_gmag", "LKSL_statistic", "bp_rp", "parallax_over_error"]

# Sort by importance
indices = np.argsort(importances)[::-1]

#Prediction and Accuracy Check
y_pred = clf.predict(X_test_scaled)
print("-------------------------Classification Analysis-------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("-------------------------------------------------------------------------------------")

#------------------Compute Stellar Parameters for Pulsating Stars------------------#

# extract the pulsating stars
super_label = "super_label"
df_pulsators = df[df[super_label] == "Pulsating"].copy()

# Only use stars with good parallax data
df_pulsators = df_pulsators[df_pulsators["parallax_over_error"] > 5].copy()

# Compute distance in parsecs
df_pulsators["distance_pc"] = 1000/df_pulsators["parallax"]

# compute absolute magnitude
df_pulsators["abs_gmag"] = df_pulsators["Mean_gmag"] - 5 * np.log10(df_pulsators["distance_pc"]/10)

# Compute T-eff from bp_rp color using empirical relation

# Simple cut: giants vs dwarfs (This is approximate; adjust thresholds as needed)
df_pulsators["star_type"] = np.where(df_pulsators["abs_gmag"] < 3, "Giant", "dwarf")
df_pulsators["star_type"].value_counts()

# Example: giants slightly metal-poor, dwarfs solar
df_pulsators["Fe_H"] = np.where(df_pulsators["star_type"] == "Giant", -0.5, 0.0)

# Function to compute T-eff using Mucciarelli et al. (2021) relation
def teff_mucciarelli(bp_rp, Fe_H, star_type):
    # Assign coefficients
    if star_type == "dwarf":
        b0, b1, b2, b3, b4, b5 = 0.4929, 0.5092, -0.0353, 0.0192, -0.0020, -0.0395
    else:  # giant
        b0, b1, b2, b3, b4, b5 = 0.5323, 0.4775, -0.0344, -0.0110, -0.0020, -0.0009

    C = bp_rp
    theta = b0 + b1*C + b2*C**2 + b3*Fe_H + b4*Fe_H**2 + b5*Fe_H*C

    Teff = 5040 / theta
    return Teff

# Compute T-eff for pulsating stars
df_pulsators["Teff"] = df_pulsators.apply(
    lambda row: teff_mucciarelli(row["bp_rp"], row["Fe_H"], row["star_type"]),
    axis=1
)

# Sun's absolute magnitude in Gaia G band
M_sun_g = 4.67

# Compute luminosity in solar units
df_pulsators["Luminosity"] = 10 ** (0.4 * (M_sun_g - df_pulsators["abs_gmag"]))

T_sun = 5772  # K

# Compute radius in solar units using Stefan-Boltzmann law
df_pulsators["Radius"] = np.sqrt(df_pulsators["Luminosity"] / (df_pulsators["Teff"]/T_sun)**4)

def estimate_mass_from_luminosity(L):
    """
    Estimate stellar mass (in solar masses) from luminosity (in solar units)
    using piecewise mass-luminosity relation.
    """
    if L < 0.023:   # corresponds roughly to 0.43 M_sun
        alpha = 2.3
    elif L < 16:    # 0.43–2 M_sun
        alpha = 4.0
    elif L < 1e4:   # 2–20 M_sun
        alpha = 3.5
    else:           # >20 M_sun
        alpha = 1.0

    M = L ** (1/alpha)
    return M

# Apply to your pulsators
df_pulsators["Mass"] = df_pulsators["Luminosity"].apply(estimate_mass_from_luminosity)

# ================== Summary of Physical Properties for Pulsating Stars ==================

physical_cols = ["Teff", "Luminosity", "Radius", "Mass", "abs_gmag", "distance_pc"]

print("\n")
print("--------------------- Overall Descriptive Statistics ------------------------------")
print(df_pulsators[physical_cols].describe())
print("-----------------------------------------------------------------------------------\n")

print("\n")
print("--------------------- Correlation Matrix ------------------------------")
corr_cols = ["Period", "Amplitude", "Mass", "Luminosity", "Radius", "Teff"]
print(df_pulsators[corr_cols].corr())
print("------------------------------------------------------------------------\n")

# Group by pulsating subtype (DSCT, DCEP, RRAB, etc.)
summary_by_subtype = df_pulsators.groupby("ML_classification")[physical_cols].agg(
    ['count', 'mean', 'std', 'min', 'max']
)
print("\n")
print("--------------------- Summary by Pulsating Subtype ------------------------------")
print(summary_by_subtype)
print("-------------------------------------------------------------------------------\n")

# Group by star type (dwarf vs giant)
summary_by_star_type = df_pulsators.groupby("star_type")[physical_cols].agg(
    ['count', 'mean', 'std', 'min', 'max']
)
print("\n")
print("--------------------- Summary by Star Type (Dwarf vs Giant) ---------------------")
print(summary_by_star_type)
print("-------------------------------------------------------------------------------")
print("\n")
print("For reference: Sun → Teff = 5772 K, L = 1 L☉, R = 1 R☉")


# Save to CSV
df_pulsators.to_csv("pulsators_with_observables.csv", index=False)
print("\n")
print("File saved: pulsators_with_observables.csv")

#----------------------Plotting---------------------#

# Plot feature importances
plt.figure(figsize=(8, 5))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.ylabel("Importance")
plt.show()

# Plot HR diagram

plt.figure(figsize=(8, 10))
plt.scatter(df_pulsators["bp_rp"], df_pulsators["abs_gmag"],
            s=5, c=df_pulsators["bp_rp"], cmap="plasma", alpha=0.6)

plt.gca().invert_yaxis()  # Bright stars at the top
plt.colorbar(label="bp - rp color")
plt.xlabel("bp - rp (Color Index)")
plt.ylabel("Absolute Magnitude (G band)")
plt.title("Hertzsprung–Russell Diagram for Pulsating Stars (ASAS-SN)")
plt.show()

# Histograms of derived properties
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].hist(df_pulsators["Teff"], bins=50, color="orange", alpha=0.7)
axs[0].set_xlabel("Effective Temperature (K)")
axs[0].set_ylabel("Number of Stars")

axs[1].hist(df_pulsators["Luminosity"], bins=50, color="gold", alpha=0.7)
axs[1].set_xlabel("Luminosity (L/L☉)")

axs[2].hist(df_pulsators["Radius"], bins=50, color="skyblue", alpha=0.7)
axs[2].set_xlabel("Radius (R/R☉)")

plt.suptitle("Distribution of Derived Stellar Properties for Pulsating Stars")
plt.show()

# Teff vs Luminosity colored by Radius
plt.figure(figsize=(6, 5))
plt.scatter(df_pulsators["Teff"], df_pulsators["Luminosity"],
            s=5, c=df_pulsators["Radius"], cmap="viridis", alpha=0.7)
plt.colorbar(label="Radius (R/R☉)")
plt.xlabel("Effective Temperature (K)")
plt.ylabel("Luminosity (L/L☉)")
plt.title("Teff–Luminosity Relation for Pulsating Stars")
plt.show()

# Mass vs Luminosity
plt.figure(figsize=(6, 5))
plt.scatter(df_pulsators["Mass"], df_pulsators["Luminosity"], s=5, alpha=0.6)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Mass [M☉]")
plt.ylabel("Luminosity [L☉]")
plt.title("Mass–Luminosity Relation for Pulsating Stars")
plt.show()