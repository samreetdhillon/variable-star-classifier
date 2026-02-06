import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred, classes):
    # Plots a heatmap of the confusion matrix to identify misclassifications.
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix: Predicted vs. Actual Stellar Classes")
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.show()

def plot_feature_importances(importances, feature_names, indices):
    plt.figure(figsize=(8, 5))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

def plot_hr_diagram_by_type(df_pulsators):
    plt.figure(figsize=(10, 8))
    # Group by subtype to give each its own color and label
    for star_type, group in df_pulsators.groupby("ML_classification"):
        plt.scatter(group["bp_rp"], group["abs_gmag"], s=10, label=star_type, alpha=0.6)

    plt.gca().invert_yaxis()
    plt.xlabel("BP - RP (Color)")
    plt.ylabel("Absolute Magnitude (M_G)")
    plt.title("ASAS-SN Pulsators on the HR Diagram")
    plt.legend(markerscale=3, title="Classification")
    plt.grid(alpha=0.3)
    plt.show()

def plot_derived_properties_histograms(df_pulsators):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Teff Histogram
    axs[0].hist(df_pulsators["Teff"], bins=50, color="orange", edgecolor='black')
    axs[0].set_xlabel("Teff (K)")
    
    # Luminosity Histogram (Using Log Scale)
    log_l = np.log10(df_pulsators["Luminosity"])
    axs[1].hist(log_l, bins=50, color="gold", edgecolor='black')
    axs[1].set_xlabel("log10(Luminosity/L☉)")

    # Radius Histogram (Using Log Scale)
    log_r = np.log10(df_pulsators["Radius"])
    axs[2].hist(log_r, bins=50, color="skyblue", edgecolor='black')
    axs[2].set_xlabel("log10(Radius/R☉)")

    for ax in axs: ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_teff_luminosity(df_pulsators):
    plt.figure(figsize=(6, 5))
    plt.scatter(df_pulsators["Teff"], df_pulsators["Luminosity"],
                s=5, c=df_pulsators["Radius"], cmap="viridis", alpha=0.7)
    plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.colorbar(label="Radius (R/R☉)")
    plt.xlabel("Effective Temperature (K)")
    plt.ylabel("Luminosity (L/L☉)")
    plt.title("Teff–Luminosity Relation for Pulsating Stars")
    plt.tight_layout()
    plt.show()

def plot_mass_luminosity(df_pulsators):
    plt.figure(figsize=(6, 5))
    plt.scatter(df_pulsators["Mass"], df_pulsators["Luminosity"], s=5, alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Mass [M☉]")
    plt.ylabel("Luminosity [L☉]")
    plt.title("Mass–Luminosity Relation for Pulsating Stars")
    plt.tight_layout()
    plt.show()

def create_all_plots(df_pulsators, importances, feature_names, indices,
                     y_test=None, y_pred=None, classes=None):
    plot_feature_importances(importances, feature_names, indices)
    if y_test is not None and y_pred is not None and classes is not None:
        plot_confusion_matrix(y_test, y_pred, classes)
    #plot_hr_diagram(df_pulsators)
    plot_hr_diagram_by_type(df_pulsators)
    plot_derived_properties_histograms(df_pulsators)
    plot_teff_luminosity(df_pulsators)
    plot_mass_luminosity(df_pulsators)