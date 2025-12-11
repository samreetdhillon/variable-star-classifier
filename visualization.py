import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importances(importances, feature_names, indices):
    plt.figure(figsize=(8, 5))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_hr_diagram(df_pulsators):
    plt.figure(figsize=(8, 10))
    plt.scatter(df_pulsators["bp_rp"], df_pulsators["abs_gmag"],
                s=5, c=df_pulsators["bp_rp"], cmap="plasma", alpha=0.6)

    plt.gca().invert_yaxis()  # Bright stars at the top
    plt.colorbar(label="bp - rp color")
    plt.xlabel("bp - rp (Color Index)")
    plt.ylabel("Absolute Magnitude (G band)")
    plt.title("Hertzsprung–Russell Diagram for Pulsating Stars (ASAS-SN)")
    plt.tight_layout()
    plt.show()


def plot_derived_properties_histograms(df_pulsators):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].hist(df_pulsators["Teff"], bins=50, color="orange", alpha=0.7)
    axs[0].set_xlabel("Effective Temperature (K)")
    axs[0].set_ylabel("Number of Stars")

    axs[1].hist(df_pulsators["Luminosity"], bins=50, color="gold", alpha=0.7)
    axs[1].set_xlabel("Luminosity (L/L☉)")

    axs[2].hist(df_pulsators["Radius"], bins=50, color="skyblue", alpha=0.7)
    axs[2].set_xlabel("Radius (R/R☉)")

    plt.suptitle("Distribution of Derived Stellar Properties for Pulsating Stars")
    plt.tight_layout()
    plt.show()


def plot_teff_luminosity(df_pulsators):
    plt.figure(figsize=(6, 5))
    plt.scatter(df_pulsators["Teff"], df_pulsators["Luminosity"],
                s=5, c=df_pulsators["Radius"], cmap="viridis", alpha=0.7)
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


def create_all_plots(df_pulsators, importances, feature_names, indices):
    plot_feature_importances(importances, feature_names, indices)
    plot_hr_diagram(df_pulsators)
    plot_derived_properties_histograms(df_pulsators)
    plot_teff_luminosity(df_pulsators)
    plot_mass_luminosity(df_pulsators)
