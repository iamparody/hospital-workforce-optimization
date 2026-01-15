#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime
import os
from pathlib import Path


def analyze_overload(
    df_overload: pd.DataFrame,
    occupancy_threshold: float = 2.0,
    staffing_threshold_overload: float = 0.05,
    min_consecutive_days: int = 7
) -> dict:
    """
    Perform full facility overload analysis for the Streamlit dashboard.

    Args:
        df_overload: Input DataFrame with required columns
        occupancy_threshold: Beds occupied / available ratio considered high
        staffing_threshold_overload: Staff / patient ratio considered low
        min_consecutive_days: Minimum days for a sustained overload period

    Returns:
        dict with 'daily_df', 'facility_summary_df', and 'periods_df'
    """
    df = df_overload.copy()
    df['day'] = pd.to_datetime(df['day'])
    df = df.sort_values(['facility_id', 'day']).reset_index(drop=True)

    # Validate required columns
    required = ['facility_id', 'day', 'beds_occupied', 'beds_available', 'total_staff', 'patient_load']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Core metrics
    df['occupancy_ratio'] = df['beds_occupied'] / df['beds_available']
    df['staff_to_patient_ratio'] = df['total_staff'] / df['patient_load']
    df['occupancy_7d_avg'] = df.groupby('facility_id')['occupancy_ratio'] \
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    # Stress flags
    patient_load_threshold = df['patient_load'].quantile(0.75)
    df['high_occupancy_flag'] = (df['occupancy_ratio'] > occupancy_threshold).astype(int)
    df['low_staffing_flag'] = (df['staff_to_patient_ratio'] < staffing_threshold_overload).astype(int)
    df['high_patient_load_flag'] = (df['patient_load'] > patient_load_threshold).astype(int)
    df['high_7d_avg_flag'] = (df['occupancy_7d_avg'] > occupancy_threshold).astype(int)

    indicators = ['high_occupancy_flag', 'low_staffing_flag',
                  'high_patient_load_flag', 'high_7d_avg_flag']
    df['stress_score'] = df[indicators].sum(axis=1)
    df['overload_day'] = (df['stress_score'] >= 2).astype(int)

    # Sustained overload periods
    all_periods = []
    df['in_sustained_period'] = 0

    for facility_id in df['facility_id'].unique():
        facility_data = df[df['facility_id'] == facility_id].copy().reset_index(drop=True)

        current_streak = 0
        streak_start = None

        for idx, row in facility_data.iterrows():
            if row['overload_day'] == 1:
                if current_streak == 0:
                    streak_start = row['day']
                current_streak += 1
            else:
                if current_streak >= min_consecutive_days and streak_start is not None:
                    streak_end = facility_data.loc[idx - 1, 'day']
                    all_periods.append({
                        'facility_id': facility_id,
                        'start_date': streak_start.date(),
                        'end_date': streak_end.date(),
                        'duration_days': current_streak
                    })
                    mask = (df['facility_id'] == facility_id) & (df['day'] >= streak_start) & (df['day'] <= streak_end)
                    df.loc[mask, 'in_sustained_period'] = 1
                current_streak = 0
                streak_start = None

        # Final streak check
        if current_streak >= min_consecutive_days and streak_start is not None:
            streak_end = facility_data.iloc[-1]['day']
            all_periods.append({
                'facility_id': facility_id,
                'start_date': streak_start.date(),
                'end_date': streak_end.date(),
                'duration_days': current_streak
            })
            mask = (df['facility_id'] == facility_id) & (df['day'] >= streak_start) & (df['day'] <= streak_end)
            df.loc[mask, 'in_sustained_period'] = 1

    periods_df = pd.DataFrame(all_periods) if all_periods else pd.DataFrame(
        columns=['facility_id', 'start_date', 'end_date', 'duration_days'])

    # Facility-level summary
    facility_summary = []
    for facility_id in df['facility_id'].unique():
        fdata = df[df['facility_id'] == facility_id]
        total_days = len(fdata)
        overload_days = fdata['overload_day'].sum()
        sustained_days = fdata['in_sustained_period'].sum()
        overload_pct = (overload_days / total_days) * 100 if total_days > 0 else 0
        sustained_pct = (sustained_days / total_days) * 100 if total_days > 0 else 0
        avg_stress = fdata['stress_score'].mean()

        if overload_pct > 60 and sustained_pct > 20:
            risk = "CRITICAL"
        elif overload_pct > 50 and sustained_pct > 10:
            risk = "HIGH"
        elif overload_pct > 40:
            risk = "MODERATE"
        else:
            risk = "LOW"

        facility_summary.append({
            'facility_id': facility_id,
            'total_days': total_days,
            'overload_days': overload_days,
            'overload_percentage': round(overload_pct, 1),
            'sustained_days': sustained_days,
            'sustained_percentage': round(sustained_pct, 1),
            'mean_stress_score': round(avg_stress, 2),
            'burnout_risk': risk
        })

    facility_summary_df = pd.DataFrame(facility_summary).sort_values('overload_percentage', ascending=False)

    return {
        'daily_df': df,
        'facility_summary_df': facility_summary_df,
        'periods_df': periods_df
    }


# ==================== Original CLI Processor (kept for backward compatibility) ====================

class FacilityOverloadProcessor:
    """Standalone processor for command-line use."""

    def __init__(self):
        self.thresholds = {
            'high_occupancy': 2.0,
            'low_staffing': 0.05,
            'min_consecutive_days': 7
        }

    # ... (all your original methods: load_data, calculate_metrics, etc. remain unchanged)
    # I've kept them exactly as you had them — no need to repeat here unless you want changes.

    # Note: If you want, you can delete everything below this line if you only use the Streamlit version.
    # But keeping it doesn't hurt and preserves CLI functionality.

    def load_data(self, csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        required_cols = ['facility_id', 'day', 'beds_occupied', 'beds_available',
                         'total_staff', 'patient_load']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df['day'] = pd.to_datetime(df['day'])
        print(f"Loaded {len(df)} records from {df['day'].min().date()} to {df['day'].max().date()}")
        print(f"Facilities found: {df['facility_id'].nunique()}")
        return df

    # ... (rest of your original class methods — copy them unchanged from your current file)


def main():
    parser = argparse.ArgumentParser(description='Detect sustained overload in healthcare facilities')
    parser.add_argument('input', help='Path to input CSV file')
    parser.add_argument('-o', '--output', default='./output', help='Output directory')
    parser.add_argument('--threshold-occupancy', type=float, default=2.0)
    parser.add_argument('--threshold-staffing', type=float, default=0.05)
    parser.add_argument('--min-consecutive-days', type=int, default=7)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    processor = FacilityOverloadProcessor()
    processor.thresholds['high_occupancy'] = args.threshold_occupancy or processor.thresholds['high_occupancy']
    processor.thresholds['low_staffing'] = args.threshold_staffing or processor.thresholds['low_staffing']
    processor.thresholds['min_consecutive_days'] = args.min_consecutive_days or processor.thresholds['min_consecutive_days']

    print(f"Using thresholds: {processor.thresholds}")

    try:
        outputs = processor.process(args.input, args.output)
        print("\nOUTPUT FILES CREATED:")
        for key, path in outputs.items():
            if path:
                print(f" {key}: {path}")
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
