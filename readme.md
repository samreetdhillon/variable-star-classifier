# Variable Star Classification

## Project Overview

This repository contains a modular pipeline to classify variable stars into three broad categories:

- **Eclipsing** (EA, EB, EW)
- **Pulsating** (RR Lyrae, Cepheids, Delta Scuti, Mira/HADS)
- **Rotational/Irregular** (ROT, YSO, SR, L, VAR)

It uses a publicly available **ASAS-SN Variable Star catalog** and applies a **Random Forest Classifier** from `scikit-learn` to predict the class of a star based on simple numeric features.

Next, it computes basic physical properties for the subset of pulsating stars.

## Read the full project report [here](https://drive.google.com/file/d/1cfCMLl_K5BFnsvc-xjDT24zDdwxNciRh/view?usp=sharing).

## Project Structure

```
variable-star-classifier/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── classifier.py            # Classification module
│   ├── stellar_properties.py   # Stellar properties computation
│   └── visualization.py         # Plotting and visualization
├── data/
│   └── asassn_variables.csv    # Input catalog
├── outputs/ (not committed in the repository)
│   └── pulsators_with_observables.csv  # Generated output
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
└── readme.md                    # This file
```

---

## Overview of what the pipeline does

1. **Classification** (`src/classifier.py`):
   - Loads the ASAS-SN catalog and maps detailed classes to coarse groups
   - Extracts features (Period, Amplitude, Mean_gmag, LKSL_statistic, bp_rp, parallax_over_error)
   - Splits into train/test sets, standardizes features, and trains a Random Forest classifier
   - Evaluates accuracy, confusion matrix, and classification report

2. **Stellar Properties** (`src/stellar_properties.py`):
   - Filters pulsating stars with reliable parallaxes
   - Computes distances, absolute magnitudes
   - Estimates effective temperature using Mucciarelli et al. (2021) relation
   - Calculates luminosity, radius (Stefan–Boltzmann law), and mass
   - Generates summary statistics and saves results to CSV

3. **Visualization** (`src/visualization.py`):
   - Feature importance plots
   - HR diagrams
   - Distribution histograms (Teff, Luminosity, Radius)
   - Teff–Luminosity and Mass–Luminosity relations

4. **Main Orchestrator** (`main.py`):
   - Coordinates the entire workflow from data loading to visualization

---

## Key Modules and Functions

### `src/classifier.py`

- **`VariableStarClassifier`** — Main classifier class with methods for:
  - `prepare_data()` — Maps ASAS-SN classes to coarse groups
  - `train()` — Trains Random Forest classifier
  - `evaluate()` — Prints accuracy metrics
  - `get_feature_importances()` — Returns feature importance values

### `src/stellar_properties.py`

- **`compute_stellar_properties()`** — Computes all physical properties for pulsating stars
- **`teff_mucciarelli()`** — Empirical Teff estimator from bp−rp and [Fe/H]
- **`estimate_mass_from_luminosity()`** — Piecewise mass–luminosity relation
- **`print_summary_statistics()`** — Generates statistical summaries
- **`save_pulsators_data()`** — Saves results to CSV

### `src/visualization.py`

- **`plot_feature_importances()`** — Feature importance bar chart
- **`plot_hr_diagram()`** — Hertzsprung-Russell diagram
- **`plot_derived_properties_histograms()`** — Teff, Luminosity, Radius distributions
- **`plot_teff_luminosity()`** — Teff vs Luminosity scatter plot
- **`plot_mass_luminosity()`** — Mass-Luminosity relation
- **`create_all_plots()`** — Generates all visualizations

---

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `scipy`
  - `lightkurve`
  - `astroquery`
  - `requests`

Install dependencies:

```sh
pip install -r requirements.txt
```

Or manually:

```sh
pip install numpy pandas matplotlib scipy scikit-learn astroquery lightkurve requests
```

## Usage

1. Ensure your data file is in the `data/` folder: `data/asassn_variables.csv`

2. Run the main script from the repository root:

```sh
python main.py
```

3. Output:
   - Console metrics and statistics
   - Interactive plots
   - CSV file: `outputs/pulsators_with_observables.csv`

Example output for classifier:

```
super_label
Rot_Irr      257429
Eclipsing     82868
Pulsating     38564
Name: count, dtype: int64

Accuracy: 0.9927854371409215
Confusion Matrix:
 [[24614   128   119]
 [  216 11247   106]
 [   81   170 76978]]
Classification Report:
               precision    recall  f1-score   support

   Eclipsing       0.99      0.99      0.99     24861
   Pulsating       0.97      0.97      0.97     11569
     Rot_Irr       1.00      1.00      1.00     77229

    accuracy                           0.99    113659
   macro avg       0.99      0.99      0.99    113659
weighted avg       0.99      0.99      0.99    113659

-------------------------------------------------------------------------------------


--------------------- Overall Descriptive Statistics ------------------------------
               Teff    Luminosity        Radius          Mass      abs_gmag   distance_pc
count  14405.000000  14405.000000  14405.000000  14405.000000  14405.000000  14405.000000
mean    5440.939790     28.615266      7.008503      2.298666      1.597105   3736.252448
std     1492.231761     45.365199      7.504709      0.731060      1.190395   2201.559385
min     2163.839404      0.037908      0.759752      0.441248     -3.402342    174.146248
25%     5250.806166      8.879325      3.248551      1.726215      0.774659   1855.287570
50%     5964.573562     19.934282      4.786462      2.351335      1.420998   3295.978906
75%     6398.290246     36.152327      6.267439      2.787288      2.299050   5327.650506
max     7994.514868   1694.090593     99.125562      8.366683      8.223170  15267.175573
-----------------------------------------------------------------------------------



--------------------- Correlation Matrix ------------------------------
              Period  Amplitude      Mass  Luminosity    Radius      Teff
Period      1.000000   0.856096 -0.346647   -0.122111  0.507255 -0.858384
Amplitude   0.856096   1.000000 -0.206617   -0.061841  0.599950 -0.852395
Mass       -0.346647  -0.206617  1.000000    0.748040  0.388875  0.265986
Luminosity -0.122111  -0.061841  0.748040    1.000000  0.473380  0.030123
Radius      0.507255   0.599950  0.388875    0.473380  1.000000 -0.651938
Teff       -0.858384  -0.852395  0.265986    0.030123 -0.651938  1.000000
------------------------------------------------------------------------



--------------------- Summary by Pulsating Subtype ------------------------------
                   Teff                                                     ... distance_pc
                  count         mean         std          min          max  ...       count         mean          std          min           max
ML_classification                                                           ...
CWA                  47  4629.738848  481.332578  3377.975757  5394.642850  ...          47  6537.096041  2130.353337  2081.598668  11037.527594
CWB                  30  4997.326492  583.274713  3388.123049  5948.363908  ...          30  4867.670946  2117.484125  2103.934357   9615.384615
DCEP                358  3963.804724  487.033119  3026.390178  5819.388329  ...         358  5202.100508  2527.063162  1275.672917  15267.175573
DCEPS                99  4225.178341  464.832114  3189.995516  5305.915264  ...          99  5632.167141  2589.163217  1825.483753  13315.579228
DSCT               2371  6144.398850  525.493442  4648.568969  7803.306237  ...        2371  1675.967282   712.587242   557.662280   5243.838490
HADS               1991  6401.174903  590.363064  4281.688887  7823.474765  ...        1991  2621.933148  1218.509030   379.405850   7558.578987
M                  2379  2413.058015  249.678551  2163.839404  4273.339789  ...        2379  2306.773502  1331.366006   174.146248  11848.341232
RRAB               4745  5821.458843  421.274570  3316.846867  6708.599422  ...        4745  5034.598149  1921.977456   472.701489  12165.450122
RRC                2377  6498.424577  447.538737  4633.816599  7994.514868  ...        2377  5186.812431  1905.661471   532.907008  11890.606421
RRD                   8  6066.756259  407.541385  5486.142240  6481.906121  ...           8  5935.437046  2496.716035   883.080184   9950.248756

[10 rows x 30 columns]
-------------------------------------------------------------------------------



--------------------- Summary by Star Type (Dwarf vs Giant) ---------------------
            Teff                                                     Luminosity  ...  abs_gmag distance_pc
           count         mean          std          min          max      count  ...       max       count         mean          std         min           max
star_type                                                                        ...
Giant      12925  5689.712366  1250.434491  2293.627961  7994.514868      12925  ...  2.999818       12925  3952.945077  2203.474509  372.856078  15267.175573
dwarf       1480  3268.381987  1661.090033  2163.839404  7823.474765       1480  ...  8.223170        1480  1843.852292   884.577005  174.146248   5527.915976

[2 rows x 30 columns]
-------------------------------------------------------------------------------


For reference: Sun → Teff = 5772 K, L = 1 L☉, R = 1 R☉


File saved: pulsators_with_observables.csv
```

## Notes / Caveats

- The Teff relation and the giant/dwarf cut are approximate; treat derived masses/radii as order-of-magnitude estimates.
- The classifier uses a class-balanced Random Forest. So consider cross-validation or hyperparameter tuning for production use.
- Required columns in input CSV: `Period`, `Amplitude`, `Mean_gmag`, `LKSL_statistic`, `bp_rp`, `parallax`, `parallax_over_error`, `ML_classification`.
- Teff is computed using an empirical relation from [_A. Mucciarelli 2021_](https://arxiv.org/abs/2106.03882).
- All generated outputs are saved to the `outputs/` folder.

## Extensions & Future Work

- Implement deep learning models for higher-dimensional feature analysis
- Add cross-validation and hyperparameter optimization
- Extend to additional variable star classes
- Implement automated light curve analysis

## Author

Samreet S. Dhillon  
M.Sc Physics,  
Panjab University, Chandigarh.
