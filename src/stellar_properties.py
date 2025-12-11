import numpy as np
import pandas as pd


def compute_stellar_properties(df):
    # Extract the pulsating stars
    df_pulsators = df[df["super_label"] == "Pulsating"].copy()

    # Only use stars with good parallax data
    df_pulsators = df_pulsators[df_pulsators["parallax_over_error"] > 5].copy()

    # Compute distance in parsecs
    df_pulsators["distance_pc"] = 1000 / df_pulsators["parallax"]

    # Compute absolute magnitude
    df_pulsators["abs_gmag"] = df_pulsators["Mean_gmag"] - 5 * np.log10(df_pulsators["distance_pc"] / 10)

    # Simple cut: giants vs dwarfs (This is approximate; adjust thresholds as needed)
    df_pulsators["star_type"] = np.where(df_pulsators["abs_gmag"] < 3, "Giant", "dwarf")

    # Example: giants slightly metal-poor, dwarfs solar
    df_pulsators["Fe_H"] = np.where(df_pulsators["star_type"] == "Giant", -0.5, 0.0)

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
    df_pulsators["Radius"] = np.sqrt(df_pulsators["Luminosity"] / (df_pulsators["Teff"] / T_sun) ** 4)

    # Apply mass estimation
    df_pulsators["Mass"] = df_pulsators["Luminosity"].apply(estimate_mass_from_luminosity)

    return df_pulsators


def teff_mucciarelli(bp_rp, Fe_H, star_type):
    """
    Compute effective temperature using Mucciarelli et al. (2021) relation.
    """
    # Assign coefficients
    if star_type == "dwarf":
        b0, b1, b2, b3, b4, b5 = 0.4929, 0.5092, -0.0353, 0.0192, -0.0020, -0.0395
    else:  # giant
        b0, b1, b2, b3, b4, b5 = 0.5323, 0.4775, -0.0344, -0.0110, -0.0020, -0.0009

    C = bp_rp
    theta = b0 + b1 * C + b2 * C ** 2 + b3 * Fe_H + b4 * Fe_H ** 2 + b5 * Fe_H * C

    Teff = 5040 / theta
    return Teff


def estimate_mass_from_luminosity(L):
    if L < 0.023:   # corresponds roughly to 0.43 M_sun
        alpha = 2.3
    elif L < 16:    # 0.43–2 M_sun
        alpha = 4.0
    elif L < 1e4:   # 2–20 M_sun
        alpha = 3.5
    else:           # >20 M_sun
        alpha = 1.0

    M = L ** (1 / alpha)
    return M


def print_summary_statistics(df_pulsators):
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


def save_pulsators_data(df_pulsators, filename="pulsators_with_observables.csv"):
    df_pulsators.to_csv(filename, index=False)
    print("\n")
    print(f"File saved: {filename}")
