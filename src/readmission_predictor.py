"""
readmission_predictor.py
Complete pipeline for predicting patient readmission within 30 days.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

class ReadmissionPredictor:
    """End-to-end readmission prediction pipeline."""
    
    def __init__(self, model_path='readmission_model.pkl'):
        self.model = None
        self.model_path = model_path
        self.feature_columns = None
        self.optimal_threshold = 0.55
        
    def prepare_data(self, df):
        """
        Prepare raw data for modeling.
        
        Args:
            df: Raw dataframe with patient data
            
        Returns:
            Prepared dataframe with engineered features
        """
        data = df.copy()
        
        # 1. Convert datetime columns
        for col in ['arrival_datetime', 'discharge_datetime', 'next_visit']:
            data[f'{col}_dt'] = pd.to_datetime(data[col], errors='coerce')
        
        # 2. Create missing value indicator
        data['has_next_visit'] = data['next_visit'].notna().astype(int)
        
        # 3. Calculate length of stay
        data['length_of_stay_hours'] = (
            data['discharge_datetime_dt'] - data['arrival_datetime_dt']
        ).dt.total_seconds() / 3600
        
        # 4. Create overnight stay indicator
        data['overnight_stay'] = (data['length_of_stay_hours'] > 12).astype(int)
        
        # 5. Extract temporal features
        data['arrival_day_of_week'] = data['arrival_datetime_dt'].dt.dayofweek
        data['is_weekend_admission'] = (data['arrival_day_of_week'] >= 5).astype(int)
        data['arrival_hour'] = data['arrival_datetime_dt'].dt.hour
        data['late_night_admission'] = ((data['arrival_hour'] >= 22) | (data['arrival_hour'] <= 6)).astype(int)
        data['arrival_month'] = data['arrival_datetime_dt'].dt.month
        
        # 6. Patient history features
        patient_visit_counts = data.groupby('patient_id').size()
        data['previous_visit_count'] = data['patient_id'].map(patient_visit_counts) - 1
        data['frequent_visitor'] = (data['previous_visit_count'] > 2).astype(int)
        
        # 7. Encode categorical variables
        categorical_cols = ['triage_category', 'visit_type']
        for col in categorical_cols:
            data = pd.get_dummies(data, columns=[col], drop_first=True, prefix=col)
        
        return data
    
    def get_feature_columns(self, df):
        """Extract feature columns from prepared dataframe."""
        # Define base features
        base_features = [
            'age', 'sex', 
            'known_chronic_condition', 'num_chronic_diagnoses', 'num_procedures',
            'length_of_stay_hours', 'overnight_stay',
            'is_weekend_admission', 'late_night_admission', 'arrival_month',
            'has_next_visit', 'previous_visit_count', 'frequent_visitor'
        ]
        
        # Add encoded categorical features
        encoded_features = [col for col in df.columns 
                          if any(col.startswith(prefix) for prefix in ['triage_category_', 'visit_type_'])]
        
        return base_features + encoded_features
    
    def train_model(self, df, target_col='readmitted_30d'):
        """
        Train LightGBM model on prepared data.
        
        Args:
            df: Prepared dataframe
            target_col: Name of target column
            
        Returns:
            Trained model
        """
        # Prepare features
        self.feature_columns = self.get_feature_columns(df)
        X = df[self.feature_columns]
        y = df[target_col]
        
        # Time-based split (80/20)
        df_sorted = df.sort_values('arrival_datetime_dt')
        split_idx = int(len(df_sorted) * 0.8)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Positive class rate: {y_train.mean():.3f}")
        
        # Train LightGBM model
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nModel trained successfully!")
        print(f"ROC-AUC on test set: {roc_auc:.4f}")
        
        # Find optimal threshold
        self.optimal_threshold = self.find_optimal_threshold(y_test, y_pred_proba)
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")
        
        # Save model
        self.save_model()
        
        return self.model
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold to maximize F1-score."""
        thresholds = np.arange(0.1, 0.6, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            if '1' in report:
                f1 = report['1']['f1-score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh
        
        return best_threshold
    
    def predict(self, df):
        """
        Make predictions on new data.
        
        Args:
            df: Raw dataframe with patient data
            
        Returns:
            Dataframe with predictions added
        """
        if self.model is None:
            self.load_model()
        
        # Prepare data
        prepared_df = self.prepare_data(df)
        
        # Ensure all feature columns exist
        missing_features = set(self.feature_columns) - set(prepared_df.columns)
        for feature in missing_features:
            prepared_df[feature] = 0
        
        # Get predictions
        X = prepared_df[self.feature_columns]
        risk_scores = self.model.predict_proba(X)[:, 1]
        predictions = (risk_scores > self.optimal_threshold).astype(int)
        
        # Add to original dataframe
        result_df = df.copy()
        result_df['readmission_risk_score'] = risk_scores
        result_df['predicted_readmission'] = predictions
        
        return result_df
    
    def save_model(self):
        """Save model and metadata."""
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'optimal_threshold': self.optimal_threshold
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")
    
        def load_model(self):
        """Load saved model using absolute path based on this script's location."""
        # Get the directory where readmission_predictor.py is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(script_dir, 'readmission_model.pkl')
        
        # Debug prints (visible in Streamlit Cloud logs)
        print(f"[DEBUG] Script directory: {script_dir}")
        print(f"[DEBUG] Looking for model at: {full_model_path}")
        print(f"[DEBUG] File exists? {os.path.exists(full_model_path)}")
        
        if os.path.exists(full_model_path):
            try:
                with open(full_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.feature_columns = model_data['feature_columns']
                self.optimal_threshold = model_data['optimal_threshold']
                print(f"✓ Model successfully loaded from {full_model_path}")
            except Exception as e:
                print(f"✗ Error loading/pickling model: {str(e)}")
                raise RuntimeError(f"Failed to load model due to pickle error: {str(e)}")
        else:
            raise FileNotFoundError(f"Model file not found at {full_model_path}")
    
    def get_feature_importance(self):
        """Get feature importance from trained model."""
        if self.model is None:
            self.load_model()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

# Example usage
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_data.csv')
    
    # Initialize predictor
    predictor = ReadmissionPredictor()
    
    # If training new model
    # prepared_data = predictor.prepare_data(df)
    # predictor.train_model(prepared_data)
    
    # If making predictions with existing model
    # predictions = predictor.predict(df)
    # predictions.to_csv('predictions.csv', index=False)
    
    print("Readmission Predictor initialized. Use predict() method on your data.")

