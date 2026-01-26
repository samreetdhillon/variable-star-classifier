import numpy as np
import pandas as pd

def compute_stellar_properties(df):
    # 1. Extract the pulsating stars
    df_pulsators = df[df["super_label"] == "Pulsating"].copy()

    # 2. Clean data: Remove non-positive parallaxes and bad colors
    # Parallax must be > 0 to compute distance. 
    # Parallax_over_error > 5 ensures high-quality distance estimates.
    df_pulsators = df_pulsators[
        (df_pulsators["parallax"] > 0) & 
        (df_pulsators["parallax_over_error"] > 5) &
        (df_pulsators["bp_rp"].notna())
    ].copy()

    # 3. Compute distance and absolute magnitude
    df_pulsators["distance_pc"] = 1000 / df_pulsators["parallax"]
    # Distance modulus formula
    df_pulsators["abs_gmag"] = df_pulsators["Mean_gmag"] - 5 * np.log10(df_pulsators["distance_pc"]) + 5

    # 4. Refined Giant vs Dwarf cut (Color-Dependent)
    # A simple vertical cut is often wrong. 
    # Usually, if bp_rp > 0.8 and abs_gmag < 3.5, it's a Giant.
    # If bp_rp < 0.8, the cut should be more aggressive (around abs_gmag < 2).
    is_giant = (
        ((df_pulsators["bp_rp"] >= 0.8) & (df_pulsators["abs_gmag"] < 3.5)) |
        ((df_pulsators["bp_rp"] < 0.8) & (df_pulsators["abs_gmag"] < 2.0))
    )
    df_pulsators["star_type"] = np.where(is_giant, "Giant", "dwarf")

    # 5. Assign Metallicity
    df_pulsators["Fe_H"] = np.where(df_pulsators["star_type"] == "Giant", -0.5, 0.0)

    # 6. Compute Teff (Vectorized if possible, otherwise apply)
    # Note: Teff is used in the Stefan-Boltzmann law below
    df_pulsators["Teff"] = df_pulsators.apply(
        lambda row: teff_mucciarelli(row["bp_rp"], row["Fe_H"], row["star_type"]),
        axis=1
    )

    # 7. Compute Luminosity (L/Lsun)
    M_sun_g = 4.67
    df_pulsators["Luminosity"] = 10 ** (0.4 * (M_sun_g - df_pulsators["abs_gmag"]))

    # 8. Compute Radius (R/Rsun) using Stefan-Boltzmann: L = 4πR²σT⁴
    T_sun = 5772  # K
    df_pulsators["Radius"] = np.sqrt(df_pulsators["Luminosity"]) * (T_sun / df_pulsators["Teff"])**2

    # 9. Apply mass estimation
    df_pulsators["Mass"] = df_pulsators["Luminosity"].apply(estimate_mass_from_luminosity)

    return df_pulsators

def teff_mucciarelli(bp_rp, Fe_H, star_type):
    """
    Compute effective temperature using Mucciarelli et al. (2021) relation.
    Calibrated for 0.2 < (BP-RP) < 1.5.
    """
    # Assign coefficients based on star type
    if star_type == "dwarf":
        b0, b1, b2, b3, b4, b5 = 0.4929, 0.5092, -0.0353, 0.0192, -0.0020, -0.0395
    else:  # giant
        b0, b1, b2, b3, b4, b5 = 0.5323, 0.4775, -0.0344, -0.0110, -0.0020, -0.0009

    # C is the color index BP - RP
    C = bp_rp
    
    # Calculate theta (5040/Teff)
    theta = b0 + b1 * C + b2 * C**2 + b3 * Fe_H + b4 * Fe_H**2 + b5 * Fe_H * C
    
    # Safety Check: Ensure theta is positive and within physical bounds
    # theta ~ 0.5 is 10,000K, theta ~ 2.0 is 2,500K
    theta = np.clip(theta, 0.4, 2.5) 

    Teff = 5040 / theta
    return Teff

def estimate_mass_from_luminosity(L):
    """
    Estimates stellar mass using a continuous piecewise Mass-Luminosity relation.
    The constants (k) ensure the function is continuous at the boundaries.
    """
    # Handle L <= 0 to avoid math errors
    if L <= 0:
        return np.nan

    if L < 0.023:     
        # M < 0.43 M_sun: L = 0.0025 * M^2.3
        alpha = 2.3
        k = 0.0025
        M = (L / k) ** (1 / alpha)
        
    elif L < 16:      
        # 0.43 < M < 2 M_sun: L = M^4.0 (The standard solar-like relation)
        alpha = 4.0
        M = L ** (1 / alpha)
        
    elif L < 1e4:     
        # 2 < M < 20 M_sun: L = 1.5 * M^3.5
        alpha = 3.5
        k = 1.5
        M = (L / k) ** (1 / alpha)
        
    else:             
        # M > 20 M_sun: L ~ M (Eddington limit scaling)
        M = L  # alpha = 1.0

    # Clip to physical limits for variable stars (0.1 to 50 M_sun)
    return np.clip(M, 0.1, 50.0)

def print_summary_statistics(df_pulsators):
    # Ensure columns exist before printing to avoid KeyError
    physical_cols = [c for c in ["Teff", "Luminosity", "Radius", "Mass", "abs_gmag", "distance_pc"] if c in df_pulsators.columns]
    corr_cols = [c for c in ["Period", "Amplitude", "Mass", "Luminosity", "Radius", "Teff"] if c in df_pulsators.columns]

    print("\n" + "="*85)
    print("--------------------- Overall Descriptive Statistics ------------------------------")
    # Handling potential NaNs in the describe output
    print(df_pulsators[physical_cols].describe())
    print("-" * 85 + "\n")

    print("--------------------- Correlation Matrix ------------------------------")
    # numeric_only=True is vital for newer Pandas versions
    print(df_pulsators[corr_cols].corr(numeric_only=True))
    print("-" * 72 + "\n")

    # Group by subtype
    if "ML_classification" in df_pulsators.columns:
        print("--------------------- Summary by Pulsating Subtype ------------------------------")
        summary_by_subtype = df_pulsators.groupby("ML_classification")[physical_cols].agg(
            ['count', 'mean', 'std'] # Reduced to mean/std for terminal readability
        )
        print(summary_by_subtype)
        print("-" * 85 + "\n")

    # Group by star type
    if "star_type" in df_pulsators.columns:
        print("--------------------- Summary by Star Type (Dwarf vs Giant) ---------------------")
        summary_by_star_type = df_pulsators.groupby("star_type")[physical_cols].agg(['count', 'mean', 'std'])
        print(summary_by_star_type)
        print("-" * 85)

    print(f"\nFor reference: Sun → Teff = 5772 K, L = 1 L☉, R = 1 R☉")
    print(f"Total Pulsators Analyzed: {len(df_pulsators)}")

def save_pulsators_data(df_pulsators, filename="pulsators_with_observables.csv"):
    df_pulsators.to_csv(filename, index=False)
    print(f"\nFile saved: {filename}")