import pandas as pd
from classifier import VariableStarClassifier
from stellar_properties import (
    compute_stellar_properties,
    print_summary_statistics,
    save_pulsators_data
)
from visualization import create_all_plots


def main():
    # Load data
    print("Loading data from ASAS-SN catalog...")
    df = pd.read_csv("data/asassn_variables.csv")
    print(f"Loaded {len(df)} variable stars\n")

    # ==================== Classification ====================
    print("="*80)
    print("STEP 1: Classifying Variable Stars")
    print("="*80)
    
    classifier = VariableStarClassifier()
    
    # Prepare data
    df, X, y = classifier.prepare_data(df)
    
    # Train classifier
    print("\nTraining Random Forest classifier...")
    X_test_scaled, y_test, y_pred = classifier.train(X, y)
    
    # Evaluate
    classifier.evaluate(y_test, y_pred)
    
    # Get feature importances
    importances, feature_names, indices = classifier.get_feature_importances()

    # ==================== Stellar Properties ====================
    print("\n" + "="*80)
    print("STEP 2: Computing Stellar Properties for Pulsating Stars")
    print("="*80)
    
    df_pulsators = compute_stellar_properties(df)
    print(f"\nFound {len(df_pulsators)} pulsating stars with good parallax data")
    
    # Print statistics
    print_summary_statistics(df_pulsators)
    
    # Save results
    save_pulsators_data(df_pulsators, "outputs/pulsators_with_observables.csv")

    # ==================== Visualization ====================
    print("\n" + "="*80)
    print("STEP 3: Creating Visualizations")
    print("="*80)
    print("\nGenerating plots...")
    
    create_all_plots(df_pulsators, importances, feature_names, indices)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
