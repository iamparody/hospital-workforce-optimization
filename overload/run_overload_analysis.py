#!/usr/bin/env python3
"""
Run Overload Analysis - Simple runner script
Process facility data and generate overload analysis outputs.
"""

from process_facility_overload import FacilityOverloadProcessor
import pandas as pd
import os
from datetime import datetime

def main():
    """
    Main function to run overload analysis.
    
    HOW TO USE:
    1. Update the INPUT_FILE variable below with your CSV file path
    2. Run this script: python run_overload_analysis.py
    3. Check the output folder for results
    """
    
    # ============ CHANGE THIS EVERY TIME ============
    # Replace this with the path to your new CSV file
    INPUT_FILE = "department_overload_clean.csv"  # <-- CHANGE THIS
    # ===============================================
    
    # Output folder (creates automatically)
    OUTPUT_FOLDER = "./overload_output"
    
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("=" * 60)
    print("FACILITY OVERLOAD ANALYSIS RUNNER")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Please update INPUT_FILE variable in the script with your CSV path.")
        return
    
    print(f"Input file: {INPUT_FILE}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # Initialize the processor
    processor = FacilityOverloadProcessor()
    
    try:
        # Run the analysis
        outputs = processor.process(INPUT_FILE, OUTPUT_FOLDER)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE - OUTPUT FILES:")
        print("=" * 60)
        
        # List all generated files
        for key, filepath in outputs.items():
            if filepath and os.path.exists(filepath):
                filename = os.path.basename(filepath)
                filesize = os.path.getsize(filepath) / 1024  # KB
                print(f"✓ {key:20} : {filename} ({filesize:.1f} KB)")
        
        print("\n" + "=" * 60)
        print("HOW TO USE RESULTS IN VISUALIZATIONS:")
        print("=" * 60)
        print("1. For time-series charts: Use daily_overload_metrics file")
        print("2. For facility rankings: Use facility_summary file")
        print("3. For critical periods: Use sustained_overload_periods file")
        print("4. For quick overview: Read the report text file")
        
        # Optional: Load the data for immediate use
        print("\n" + "=" * 60)
        print("QUICK DATA LOADING (optional):")
        print("=" * 60)
        
        try:
            # Load the main output files into DataFrames
            daily_df = pd.read_csv(outputs['daily_metrics'])
            summary_df = pd.read_csv(outputs['facility_summary'])
            
            print(f"✓ Loaded daily data: {len(daily_df)} rows")
            print(f"✓ Loaded summary data: {len(summary_df)} facilities")
            
            # Show top 3 most overloaded facilities
            print("\nTop 3 most overloaded facilities:")
            top_3 = summary_df.head(3)
            for idx, row in top_3.iterrows():
                print(f"  {row['facility_id']}: {row['overload_percentage']:.1f}% overload - {row['burnout_risk']} risk")
                
        except Exception as e:
            print(f"Note: Could not auto-load data - {e}")
            print("You can load the CSV files manually with pd.read_csv()")
        
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure your CSV has the required columns:")
        print("   facility_id, day, beds_occupied, beds_available, total_staff, patient_load")
        print("2. Check CSV format (comma-separated, proper encoding)")
        print("3. Verify file path is correct")

if __name__ == "__main__":
    main()