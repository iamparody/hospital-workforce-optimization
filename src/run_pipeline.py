"""
Main pipeline script - run this daily.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

from feature_engineering import create_features, calculate_threshold
from model_predict import predict_next_7_days, save_predictions, generate_executive_summary
from data_validation import validate_input_data, generate_data_report

def main(input_file, output_dir='output'):
    """
    Run the complete prediction pipeline.
    
    Parameters:
    -----------
    input_file : str
        Path to daily data CSV
    
    output_dir : str
        Output directory for results
    """
    print("=" * 60)
    print("HOSPITAL STAFFING PREDICTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and validate data
    print("\n📥 Step 1: Loading data...")
    df = pd.read_csv(input_file)
    
    validation = validate_input_data(df)
    print(f"✓ Data validation: {validation['is_valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    # Step 2: Engineer features
    print("\n⚙️ Step 2: Engineering features...")
    features_df = create_features(df)
    print(f"✓ Created {len(features_df.columns)} features")
    
    # Step 3: Calculate threshold (if needed)
    print("\n📊 Step 3: Calculating thresholds...")
    threshold_info = calculate_threshold(features_df)
    print(f"✓ Threshold: {threshold_info['pressure_threshold']:.2f}")
    print(f"✓ Equivalent ratio: {threshold_info['equivalent_ratio_threshold']:.3f}")
    
    # Step 4: Make predictions
    print("\n🔮 Step 4: Making predictions...")
    predictions = predict_next_7_days(features_df, feature_engineered=True)
    
    if predictions.empty:
        print("✗ No predictions generated")
        return
    
    print(f"✓ Generated predictions for {predictions['facility_id'].nunique()} facilities")
    
    # Step 5: Save results
    print("\n💾 Step 5: Saving results...")
    detailed_path, summary_path = save_predictions(predictions, output_dir)
    
    # Step 6: Generate report
    print("\n📋 Step 6: Generating report...")
    report = generate_data_report(df, features_df)
    report_path = f'{output_dir}/data_quality_report.json'
    
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Step 7: Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    summary = generate_executive_summary(predictions)
    print(f"\n📈 Executive Summary:")
    print(f"  Total facilities: {summary.get('total_facilities', 0)}")
    print(f"  High risk facilities: {summary.get('high_risk_facilities', 0)}")
    print(f"  Facilities to watch: {len(summary.get('watchlist', []))}")
    
    if summary.get('watchlist'):
        print(f"  Watchlist: {', '.join(summary['watchlist'])}")
    
    print(f"\n📁 Output files:")
    print(f"  Detailed predictions: {detailed_path}")
    print(f"  Executive summary: {summary_path}")
    print(f"  Data quality report: {report_path}")
    
    return predictions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict staffing shortages')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='predictions', help='Output directory')
    
    args = parser.parse_args()
    predictions = main(args.input, args.output)