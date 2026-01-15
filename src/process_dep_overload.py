#!/usr/bin/env python3
"""
Facility Overload Detection - CSV Processor
Processes facility data CSV and produces overload analysis outputs.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime
import os
from pathlib import Path


class FacilityOverloadProcessor:
    """Process facility data to detect sustained overload periods."""
    
    def __init__(self):
        # Configuration thresholds
        self.thresholds = {
            'high_occupancy': 2.0,
            'low_staffing': 0.05,
            'min_consecutive_days': 7
        }
        
    def load_data(self, csv_path):
        """Load and validate CSV data."""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['facility_id', 'day', 'beds_occupied', 'beds_available',
                        'total_staff', 'patient_load']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date column
        df['day'] = pd.to_datetime(df['day'])
        
        print(f"Loaded {len(df)} records from {df['day'].min().date()} to {df['day'].max().date()}")
        print(f"Facilities found: {df['facility_id'].nunique()}")
        
        return df
    
    def calculate_metrics(self, df):
        """Calculate derived metrics."""
        data = df.copy()
        
        # Basic ratios
        data['occupancy_ratio'] = data['beds_occupied'] / data['beds_available']
        data['staff_to_patient_ratio'] = data['total_staff'] / data['patient_load']
        
        # Rolling averages for each facility
        data = data.sort_values(['facility_id', 'day'])
        
        # Calculate rolling averages
        data['occupancy_7d_avg'] = data.groupby('facility_id')['occupancy_ratio'] \
                                      .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        data['occupancy_14d_avg'] = data.groupby('facility_id')['occupancy_ratio'] \
                                       .transform(lambda x: x.rolling(window=14, min_periods=1).mean())
        
        return data
    
    def detect_overload(self, df):
        """Detect overload days using multi-indicator approach."""
        data = df.copy()
        
        # 1. Individual stress indicators
        data['high_occupancy_flag'] = (data['occupancy_ratio'] > self.thresholds['high_occupancy']).astype(int)
        data['low_staffing_flag'] = (data['staff_to_patient_ratio'] < self.thresholds['low_staffing']).astype(int)
        
        # Patient load flag (top 25%)
        patient_load_threshold = data['patient_load'].quantile(0.75)
        data['high_patient_load_flag'] = (data['patient_load'] > patient_load_threshold).astype(int)
        
        # High 7-day average flag
        data['high_7d_avg_flag'] = (data['occupancy_7d_avg'] > self.thresholds['high_occupancy']).astype(int)
        
        # 2. Composite overload score (sum of indicators)
        indicators = ['high_occupancy_flag', 'low_staffing_flag', 
                     'high_patient_load_flag', 'high_7d_avg_flag']
        data['stress_score'] = data[indicators].sum(axis=1)
        
        # 3. Overload day flag (2 out of 4 indicators)
        data['overload_day'] = (data['stress_score'] >= 2).astype(int)
        
        print(f"Overload detection complete: {data['overload_day'].sum()} overload days "
              f"({data['overload_day'].mean()*100:.1f}% of total)")
        
        return data
    
    def find_sustained_periods(self, df):
        """Find sustained overload periods for each facility."""
        print("\nFinding sustained overload periods...")
        
        all_periods = []
        
        for facility_id in df['facility_id'].unique():
            facility_data = df[df['facility_id'] == facility_id].copy()
            facility_data = facility_data.sort_values('day')
            
            periods = []
            current_streak = 0
            streak_start = None
            
            for idx, row in facility_data.iterrows():
                if row['overload_day'] == 1:
                    if current_streak == 0:
                        streak_start = row['day']
                    current_streak += 1
                else:
                    if current_streak >= self.thresholds['min_consecutive_days']:
                        streak_end = facility_data.loc[idx-1, 'day']
                        periods.append({
                            'facility_id': facility_id,
                            'start_date': streak_start,
                            'end_date': streak_end,
                            'duration_days': current_streak,
                            'start_date_str': streak_start.strftime('%Y-%m-%d'),
                            'end_date_str': streak_end.strftime('%Y-%m-%d')
                        })
                    current_streak = 0
                    streak_start = None
            
            # Check final streak
            if current_streak >= self.thresholds['min_consecutive_days']:
                streak_end = facility_data.iloc[-1]['day']
                periods.append({
                    'facility_id': facility_id,
                    'start_date': streak_start,
                    'end_date': streak_end,
                    'duration_days': current_streak,
                    'start_date_str': streak_start.strftime('%Y-%m-%d'),
                    'end_date_str': streak_end.strftime('%Y-%m-%d')
                })
            
            all_periods.extend(periods)
        
        # Create DataFrame
        if all_periods:
            periods_df = pd.DataFrame(all_periods)
            periods_df = periods_df.sort_values('duration_days', ascending=False)
            print(f"Found {len(periods_df)} sustained overload periods")
        else:
            periods_df = pd.DataFrame(columns=['facility_id', 'start_date', 'end_date', 
                                              'duration_days', 'start_date_str', 'end_date_str'])
            print("No sustained overload periods found")
        
        return periods_df
    
    def summarize_facility_metrics(self, df):
        """Create summary metrics for each facility."""
        print("\nCalculating facility summaries...")
        
        summaries = []
        
        for facility_id in df['facility_id'].unique():
            facility_data = df[df['facility_id'] == facility_id].copy()
            
            # Basic metrics
            total_days = len(facility_data)
            overload_days = facility_data['overload_day'].sum()
            overload_pct = (overload_days / total_days) * 100
            
            # Average metrics
            avg_occupancy = facility_data['occupancy_ratio'].mean()
            avg_staff_ratio = facility_data['staff_to_patient_ratio'].mean()
            avg_stress_score = facility_data['stress_score'].mean()
            
            # Max stress indicators
            max_occupancy = facility_data['occupancy_ratio'].max()
            min_staff_ratio = facility_data['staff_to_patient_ratio'].min()
            max_stress_score = facility_data['stress_score'].max()
            
            summaries.append({
                'facility_id': facility_id,
                'total_days': total_days,
                'overload_days': overload_days,
                'overload_percentage': overload_pct,
                'avg_occupancy_ratio': avg_occupancy,
                'avg_staff_ratio': avg_staff_ratio,
                'avg_stress_score': avg_stress_score,
                'max_occupancy_ratio': max_occupancy,
                'min_staff_ratio': min_staff_ratio,
                'max_stress_score': max_stress_score,
                'burnout_risk': self._assess_risk(overload_pct, avg_stress_score)
            })
        
        summary_df = pd.DataFrame(summaries)
        summary_df = summary_df.sort_values('overload_percentage', ascending=False)
        
        return summary_df
    
    def _assess_risk(self, overload_pct, avg_stress_score):
        """Assess burnout risk level."""
        if overload_pct > 60 and avg_stress_score > 2.5:
            return "CRITICAL"
        elif overload_pct > 50 and avg_stress_score > 2.0:
            return "HIGH"
        elif overload_pct > 40:
            return "MODERATE-HIGH"
        elif overload_pct > 30:
            return "MODERATE"
        else:
            return "LOW"
    
    def create_daily_output(self, df):
        """Create daily-level output with all metrics."""
        print("Creating daily output file...")
        
        # Select and order columns
        output_cols = [
            'facility_id', 'day',
            'beds_occupied', 'beds_available', 'total_staff', 'patient_load',
            'occupancy_ratio', 'staff_to_patient_ratio',
            'occupancy_7d_avg', 'occupancy_14d_avg',
            'high_occupancy_flag', 'low_staffing_flag',
            'high_patient_load_flag', 'high_7d_avg_flag',
            'stress_score', 'overload_day'
        ]
        
        # Ensure all columns exist
        existing_cols = [col for col in output_cols if col in df.columns]
        daily_df = df[existing_cols].copy()
        
        # Format date
        daily_df['day'] = daily_df['day'].dt.strftime('%Y-%m-%d')
        
        return daily_df
    
    def process(self, input_csv, output_dir='./output'):
        """
        Main processing pipeline.
        
        Args:
            input_csv: Path to input CSV file
            output_dir: Directory to save output files
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = self.load_data(input_csv)
        
        # Calculate metrics
        df = self.calculate_metrics(df)
        
        # Detect overload
        df = self.detect_overload(df)
        
        # Find sustained periods
        sustained_periods_df = self.find_sustained_periods(df)
        
        # Create facility summaries
        facility_summary_df = self.summarize_facility_metrics(df)
        
        # Create daily output
        daily_output_df = self.create_daily_output(df)
        
        # Save outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Daily metrics with overload flags
        daily_file = f"{output_dir}/daily_overload_metrics_{timestamp}.csv"
        daily_output_df.to_csv(daily_file, index=False)
        print(f"✓ Daily metrics saved to: {daily_file}")
        
        # 2. Facility summary
        summary_file = f"{output_dir}/facility_summary_{timestamp}.csv"
        facility_summary_df.to_csv(summary_file, index=False)
        print(f"✓ Facility summary saved to: {summary_file}")
        
        # 3. Sustained overload periods
        if not sustained_periods_df.empty:
            periods_file = f"{output_dir}/sustained_overload_periods_{timestamp}.csv"
            sustained_periods_df.to_csv(periods_file, index=False)
            print(f"✓ Sustained periods saved to: {periods_file}")
        else:
            print("✓ No sustained periods to save")
        
        # 4. Combined report
        report_file = f"{output_dir}/overload_analysis_report_{timestamp}.txt"
        self._generate_report(report_file, facility_summary_df, sustained_periods_df, df)
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"Output files saved to: {output_dir}/")
        print(f"Timestamp: {timestamp}")
        
        return {
            'daily_metrics': daily_file,
            'facility_summary': summary_file,
            'sustained_periods': periods_file if not sustained_periods_df.empty else None,
            'report': report_file
        }
    
    def _generate_report(self, report_file, facility_summary_df, sustained_periods_df, df):
        """Generate a text report summary."""
        with open(report_file, 'w') as f:
            f.write("FACILITY OVERLOAD ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Records Processed: {len(df):,}\n")
            f.write(f"Date Range: {df['day'].min().date()} to {df['day'].max().date()}\n")
            f.write(f"Facilities Analyzed: {df['facility_id'].nunique()}\n")
            f.write(f"Total Overload Days: {df['overload_day'].sum():,} "
                   f"({df['overload_day'].mean()*100:.1f}%)\n\n")
            
            f.write("FACILITY RANKING BY OVERLOAD PERCENTAGE:\n")
            f.write("-" * 50 + "\n")
            for idx, row in facility_summary_df.iterrows():
                f.write(f"{row['facility_id']}: {row['overload_percentage']:.1f}% "
                       f"({row['overload_days']} days) - {row['burnout_risk']} risk\n")
            
            f.write("\nSUSTAINED OVERLOAD PERIODS (7+ consecutive days):\n")
            f.write("-" * 50 + "\n")
            if not sustained_periods_df.empty:
                for idx, row in sustained_periods_df.head(10).iterrows():
                    f.write(f"{row['facility_id']}: {row['start_date_str']} to "
                           f"{row['end_date_str']} ({row['duration_days']} days)\n")
                if len(sustained_periods_df) > 10:
                    f.write(f"... and {len(sustained_periods_df) - 10} more periods\n")
            else:
                f.write("No sustained overload periods found\n")
            
            f.write("\nRISK DISTRIBUTION:\n")
            f.write("-" * 50 + "\n")
            risk_counts = facility_summary_df['burnout_risk'].value_counts()
            for risk, count in risk_counts.items():
                f.write(f"{risk}: {count} facilities\n")
            
            f.write("\nKEY METRICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Average occupancy ratio: {df['occupancy_ratio'].mean():.2f}\n")
            f.write(f"Average staff-to-patient ratio: {df['staff_to_patient_ratio'].mean():.4f}\n")
            f.write(f"Average stress score: {df['stress_score'].mean():.2f}\n")
        
        print(f"✓ Report saved to: {report_file}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Detect sustained overload in healthcare facilities'
    )
    parser.add_argument('input', help='Path to input CSV file')
    parser.add_argument('-o', '--output', default='./output', 
                       help='Output directory (default: ./output)')
    parser.add_argument('--threshold-occupancy', type=float, default=2.0,
                       help='High occupancy threshold (default: 2.0)')
    parser.add_argument('--threshold-staffing', type=float, default=0.05,
                       help='Low staffing threshold (default: 0.05)')
    parser.add_argument('--min-consecutive-days', type=int, default=7,
                       help='Minimum days for sustained overload (default: 7)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize processor
    processor = FacilityOverloadProcessor()
    
    # Update thresholds if provided
    if args.threshold_occupancy:
        processor.thresholds['high_occupancy'] = args.threshold_occupancy
    if args.threshold_staffing:
        processor.thresholds['low_staffing'] = args.threshold_staffing
    if args.min_consecutive_days:
        processor.thresholds['min_consecutive_days'] = args.min_consecutive_days
    
    print(f"Using thresholds: {processor.thresholds}")
    
    try:
        # Process the data
        outputs = processor.process(args.input, args.output)
        
        print("\nOUTPUT FILES CREATED:")
        for key, path in outputs.items():
            if path:
                print(f"  {key}: {path}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()