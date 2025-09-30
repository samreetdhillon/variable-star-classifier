# Variable Star Classification

## Project Overview

This project classifies variable stars into three broad categories:

- **Eclipsing** (EA, EB, EW)
- **Pulsating** (RR Lyrae, Cepheids, Delta Scuti, Mira/HADS)
- **Rotational/Irregular** (ROT, YSO, SR, L, VAR)

It uses a publicly available **ASAS-SN Variable Star catalog** and applies a **Random Forest Classifier** from `scikit-learn` to predict the class of a star based on simple numeric features.

---

## Features

- Classifies stars into 3 main categories using machine learning.
- Handles large datasets (~380,000 stars).
- Balances classes to improve accuracy for underrepresented types.
- Uses basic numeric features:
  - `Period` (days)
  - `Amplitude` (mag)
  - `Mean_gmag` (average g-band magnitude)

---

## Requirements

- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `lightkurve`, `astroquery`

Install dependencies via pip:

```bash
pip install numpy pandas scikit-learn matplotlib lightkurve astroquery
```

## Usage

1. Download the ASAS-SN variable star catalog (asassn_variables_x.csv).

2. Run the Python script variable_star_classifier.py.

3. The program outputs:
   - Class counts for each category
   - Accuracy
   - Confusion matrix
   - Classification report

Example output (grouped into 3 classes):

```
super_label
Rot_Irr      257429
Eclipsing     82868
Pulsating     38564
Name: count, dtype: int64
Accuracy: 0.9494716652442834

Confusion Matrix:
 [[22961   861  1039]
 [ 2441  8982   146]
 [ 1033   223 75973]]

Classification Report:
               precision    recall  f1-score   support

   Eclipsing       0.87      0.92      0.90     24861
   Pulsating       0.89      0.78      0.83     11569
     Rot_Irr       0.98      0.98      0.98     77229

    accuracy                           0.95    113659
   macro avg       0.92      0.89      0.90    113659
weighted avg       0.95      0.95      0.95    113659

```

## Extensions & Future Work

- Incorporate Fourier features from light curves to improve pulsating star classification.

- Use additional photometric bands for better accuracy.

- Implement deep learning models for higher-dimensional feature analysis.

- Add visualization for feature distributions and confusion matrices.

## Author

Samreet S. Dhillon  
M.Sc Physics,  
Panjab University, Chandigarh.
