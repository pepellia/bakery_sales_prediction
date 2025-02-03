import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Import configuration
from bakery_sales_prediction.config import (TRAIN_PATH, TEST_PATH, 
                                          SAMPLE_SUBMISSION_PATH,
                                          WARENGRUPPEN, get_warengruppe_name)

# Create output directories
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output', SCRIPT_NAME)
VIZ_DIR = os.path.join(SCRIPT_DIR, 'visualizations', SCRIPT_NAME)
for directory in [OUTPUT_DIR, VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

class BakeryModel:
    def __init__(self):
        self.model = LinearRegression()
        sns.set_style("whitegrid")
    
    def prepare_data(self, df):
        """Prepare features for the model"""
        df = df.copy()
        
        # Basic date components
        df['Wochentag'] = df['Datum'].dt.dayofweek
        df['is_weekend'] = df['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
        df['week_of_year'] = df['Datum'].dt.isocalendar().week
        df['month'] = df['Datum'].dt.month
        df['day_of_month'] = df['Datum'].dt.day
        
        # Create dummy variables for weekday and product group
        weekday_dummies = pd.get_dummies(df['Wochentag'], prefix='weekday', drop_first=True)
        product_dummies = pd.get_dummies(df['Warengruppe'], prefix='product', drop_first=True)
        
        # Combine features
        features = pd.concat([
            weekday_dummies,
            product_dummies,
            df[['is_weekend', 'week_of_year', 'month', 'day_of_month']]
        ], axis=1)
        
        return features
    
    def train_model(self, train_data, val_data=None, verbose=True):
        """Train the model and evaluate on validation data if provided"""
        # Prepare features
        X_train = self.prepare_data(train_data)
        y_train = train_data['Umsatz']
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Training predictions and metrics
        train_pred = self.model.predict(X_train)
        train_metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_train, train_pred)),
            'MAE': mean_absolute_error(y_train, train_pred),
            'R²': r2_score(y_train, train_pred)
        }
        
        # Validation metrics if validation data is provided
        val_metrics = None
        if val_data is not None:
            X_val = self.prepare_data(val_data)
            y_val = val_data['Umsatz']
            val_pred = self.model.predict(X_val)
            val_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_val, val_pred)),
                'MAE': mean_absolute_error(y_val, val_pred),
                'R²': r2_score(y_val, val_pred)
            }
        
        if verbose:
            print("\nTraining Metrics (2013-07-01 to 2017-07-31):")
            for metric, value in train_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            if val_metrics:
                print("\nValidation Metrics (2017-08-01 to 2018-07-31):")
                for metric, value in val_metrics.items():
                    print(f"{metric}: {value:.4f}")
        
        # Analyze and visualize feature importance
        self._analyze_feature_importance(X_train)
        
        return train_metrics, val_metrics
    
    def predict(self, data):
        """Make predictions for new data"""
        X = self.prepare_data(data)
        return self.model.predict(X)
    
    def _analyze_feature_importance(self, X):
        """Analyze and visualize feature importance"""
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(self.model.coef_)
        })
        importance = importance.sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(importance.head(10))
        
        # Feature Importance Visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance.head(10), x='Importance', y='Feature')
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Absolute Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'feature_importance.png'))
        plt.close()
        
        # Save feature importance to file
        importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    
    def analyze_predictions(self, actual_data, predictions, dataset_name=""):
        """Analyze and visualize predictions"""
        # Add predictions to data
        data = actual_data.copy()
        data['Predictions'] = predictions
        
        # 1. Scatter Plot: Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Umsatz'], data['Predictions'], alpha=0.5)
        max_val = max(data['Umsatz'].max(), data['Predictions'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Sales (€)')
        plt.ylabel('Predicted Sales (€)')
        plt.title(f'Actual vs Predicted Sales - {dataset_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, f'predictions_vs_actual_{dataset_name.lower()}.png'))
        plt.close()
        
        # 2. Time Series by Product Group
        for group, name in WARENGRUPPEN.items():
            group_data = data[data['Warengruppe'] == group].sort_values('Datum')
            
            plt.figure(figsize=(15, 6))
            plt.plot(group_data['Datum'], group_data['Umsatz'], 
                    label='Actual', alpha=0.7)
            plt.plot(group_data['Datum'], group_data['Predictions'], 
                    label='Predicted', alpha=0.7, linestyle='--')
            plt.title(f'Time Series - {name} ({dataset_name})')
            plt.xlabel('Date')
            plt.ylabel('Sales (€)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(VIZ_DIR, f'timeseries_group_{group}_{dataset_name.lower()}.png'))
            plt.close()
        
        # 3. Analysis by Product Group
        group_metrics = []
        for group, name in WARENGRUPPEN.items():
            group_data = data[data['Warengruppe'] == group]
            metrics = {
                'Group': group,
                'Name': name,
                'RMSE': np.sqrt(mean_squared_error(group_data['Umsatz'], group_data['Predictions'])),
                'MAE': mean_absolute_error(group_data['Umsatz'], group_data['Predictions']),
                'R²': r2_score(group_data['Umsatz'], group_data['Predictions'])
            }
            group_metrics.append(metrics)
        
        metrics_df = pd.DataFrame(group_metrics)
        
        # Save metrics
        metrics_df.to_csv(os.path.join(OUTPUT_DIR, f'metrics_by_group_{dataset_name.lower()}.csv'), 
                         index=False)
        
        # Plot metrics by group
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE plot
        sns.barplot(data=metrics_df, x='Name', y='RMSE', ax=ax1)
        ax1.set_title(f'RMSE by Product Group - {dataset_name}')
        ax1.set_xlabel('Product Group')
        ax1.tick_params(axis='x', rotation=45)
        
        # R² plot
        sns.barplot(data=metrics_df, x='Name', y='R²', ax=ax2)
        ax2.set_title(f'R² Score by Product Group - {dataset_name}')
        ax2.set_xlabel('Product Group')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, f'metrics_by_group_{dataset_name.lower()}.png'))
        plt.close()
        
        return metrics_df

def main():
    """Main function to run the model training and evaluation"""
    print("Starting model training and evaluation...")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    
    # Create train/validation split
    train_mask = (train_df['Datum'] >= '2013-07-01') & (train_df['Datum'] <= '2017-07-31')
    val_mask = (train_df['Datum'] >= '2017-08-01') & (train_df['Datum'] <= '2018-07-31')
    
    train_data = train_df[train_mask].copy()
    val_data = train_df[val_mask].copy()
    
    # Initialize and train model
    print("\nTraining model...")
    model = BakeryModel()
    train_metrics, val_metrics = model.train_model(train_data, val_data)
    
    # Analyze predictions
    print("\nAnalyzing predictions...")
    train_pred = model.predict(train_data)
    val_pred = model.predict(val_data)
    
    train_analysis = model.analyze_predictions(train_data, train_pred, "Training")
    val_analysis = model.analyze_predictions(val_data, val_pred, "Validation")
    
    # Generate submission predictions
    print("\nGenerating submission predictions...")
    test_df = pd.read_csv(TEST_PATH)
    test_df['Datum'] = pd.to_datetime(test_df['id'].astype(str).str[:6], format='%y%m%d')
    test_pred = model.predict(test_df)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Umsatz': test_pred
    })
    
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission file saved to: {submission_path}")
    
    # Save all metrics
    metrics = {
        'training': train_metrics,
        'validation': val_metrics
    }
    pd.DataFrame(metrics).to_json(os.path.join(OUTPUT_DIR, 'metrics.json'))

if __name__ == "__main__":
    main() 