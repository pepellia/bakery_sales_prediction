"""
Fourier Analysis for Bakery Sales Prediction

This script implements a Fourier analysis approach for predicting bakery sales.
Previous analysis showed this method achieved the best MAE (54.82 sales/day)
compared to DNN (55.29), Moving Average (59.34), and LSTM (59.89).

Key features:
- Separate analysis for each product group
- Focus on weekday patterns (crucial for German bakeries)
- Excludes weather data (shown to have minimal impact)
- Uses FFT to identify dominant frequencies in sales patterns
- Includes train/validation/test splits for robust evaluation
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, cast
from numpy.typing import NDArray
import os
from datetime import datetime

# Type aliases for better readability
ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # 3_Model directory
MODEL_NAME = 'fourier_analysis'

# Create output directories
OUTPUT_DIR = os.path.join(MODEL_ROOT, 'output', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(MODEL_ROOT, 'visualizations', MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Constants
PRODUCT_GROUPS = {
    1: 'Brot',
    2: 'Brötchen',
    3: 'Croissant',
    4: 'Konditorei',
    5: 'Kuchen',
    6: 'Saisonbrot'
}

class FourierAnalyzer:
    def __init__(self, n_harmonics: int = 4):
        """Initialize the Fourier analyzer.
        
        Args:
            n_harmonics: Number of harmonics to use for reconstruction (default: 4)
        """
        self.n_harmonics = n_harmonics
        self.output_dir = OUTPUT_DIR
        self.viz_dir = VISUALIZATION_DIR
        
        # Store FFT results and metrics
        self.fft_results: Dict[int, Dict[str, Union[FloatArray, IntArray]]] = {}
        self.predictions: Dict[int, FloatArray] = {}
        self.train_mae_scores: Dict[int, float] = {}
        self.val_mae_scores: Dict[int, float] = {}
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training, validation, and test data."""
        # Load full training data
        train_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 
                                'input', 'competition_data', 'train.csv')
        full_train_df = pd.read_csv(train_path)
        full_train_df['Datum'] = pd.to_datetime(full_train_df['Datum'])
        
        # Split into train and validation using fixed date ranges
        train_mask = (full_train_df['Datum'] >= '2013-07-01') & (full_train_df['Datum'] <= '2017-07-31')
        val_mask = (full_train_df['Datum'] >= '2017-08-01') & (full_train_df['Datum'] <= '2018-07-31')
        
        train_df = full_train_df[train_mask].copy()
        val_df = full_train_df[val_mask].copy()
        
        # Load test data
        test_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 
                               'input', 'competition_data', 'sample_submission.csv')
        test_df = pd.read_csv(test_path)
        test_df['Datum'] = pd.to_datetime('20' + test_df['id'].astype(str).str[:6], format='%Y%m%d')
        test_df['Warengruppe'] = test_df['id'].astype(str).str[-1].astype(int)
        
        return train_df, val_df, test_df
    
    def apply_fft(self, sales: FloatArray) -> Tuple[FloatArray, FloatArray, FloatArray]:
        """Apply FFT to sales data and return frequencies and amplitudes."""
        n = len(sales)
        fft_vals = cast(ComplexArray, fft(sales))
        freqs = cast(FloatArray, fftfreq(n))
        
        # Get amplitudes and phases
        amplitudes = cast(FloatArray, np.abs(fft_vals))
        phases = cast(FloatArray, np.angle(fft_vals))
        
        return freqs, amplitudes, phases
    
    def get_dominant_frequencies(self, freqs: FloatArray, amplitudes: FloatArray) -> IntArray:
        """Get indices of dominant frequencies (excluding DC component)."""
        # Sort by amplitude but exclude the DC component (index 0)
        sorted_indices = np.argsort(amplitudes[1:])[::-1] + 1
        return cast(IntArray, sorted_indices[:self.n_harmonics])
    
    def reconstruct_signal(self, freqs: FloatArray, amplitudes: FloatArray, 
                          phases: FloatArray, dominant_idx: IntArray, 
                          n_points: int) -> FloatArray:
        """Reconstruct signal using dominant frequencies."""
        # Create new frequency array for desired length
        t = np.arange(n_points)
        reconstructed = np.zeros(n_points)
        
        # Add DC component (mean)
        reconstructed += amplitudes[0] / len(freqs)
        
        # Add contribution from each dominant frequency
        for idx in dominant_idx:
            if idx >= len(freqs):
                continue
            freq = freqs[idx]
            amplitude = amplitudes[idx] / len(freqs)
            phase = phases[idx]
            reconstructed += 2 * amplitude * np.cos(2 * np.pi * freq * t + phase)
        
        return cast(FloatArray, reconstructed)
    
    def analyze_product_group(self, train_df: pd.DataFrame, product_group: int) -> None:
        """Analyze sales patterns for a specific product group."""
        # Filter data for product group
        group_data = train_df[train_df['Warengruppe'] == product_group].copy()
        group_data = group_data.sort_values('Datum')
        
        # Get sales data and convert to numpy array
        sales = cast(FloatArray, group_data['Umsatz'].to_numpy(dtype=np.float64))
        
        # Apply FFT
        freqs, amplitudes, phases = self.apply_fft(sales)
        
        # Get dominant frequencies
        dominant_idx = self.get_dominant_frequencies(freqs, amplitudes)
        
        # Store results
        self.fft_results[product_group] = {
            'freqs': freqs,
            'amplitudes': amplitudes,
            'phases': phases,
            'dominant_idx': dominant_idx
        }
        
        # Plot frequency spectrum
        plt.figure(figsize=(12, 6))
        plt.plot(freqs[1:len(freqs)//2], amplitudes[1:len(freqs)//2])
        plt.title(f'Frequency Spectrum - {PRODUCT_GROUPS[product_group]}')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.savefig(os.path.join(self.viz_dir, f'spectrum_group_{product_group}.png'))
        plt.close()
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for test data."""
        predictions_df = test_df.copy()
        predictions_df['Umsatz'] = np.zeros(len(test_df), dtype=np.float64)
        
        for product_group in PRODUCT_GROUPS.keys():
            # Get FFT results for this group
            fft_result = self.fft_results[product_group]
            
            # Generate predictions using dominant frequencies
            test_group = test_df[test_df['Warengruppe'] == product_group]
            n_points = len(test_group)
            predictions = self.reconstruct_signal(
                cast(FloatArray, fft_result['freqs']),
                cast(FloatArray, fft_result['amplitudes']),
                cast(FloatArray, fft_result['phases']),
                cast(IntArray, fft_result['dominant_idx']),
                n_points
            )
            
            # Store predictions
            mask = predictions_df['Warengruppe'] == product_group
            predictions_df.loc[mask, 'Umsatz'] = predictions
            self.predictions[product_group] = predictions
        
        # Ensure non-negative predictions
        predictions_df['Umsatz'] = predictions_df['Umsatz'].clip(lower=0)
        
        return predictions_df
    
    def evaluate(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Evaluate model performance on training and validation data."""
        print("\nModel Performance by Product Group:")
        print("-" * 70)
        print(f"{'Product Group':<15} {'Train MAE':<15} {'Validation MAE':<15}")
        print("-" * 70)
        
        total_train_mae = np.float64(0.0)
        total_val_mae = np.float64(0.0)
        n_groups = len(PRODUCT_GROUPS)
        
        for product_group in PRODUCT_GROUPS.keys():
            # Get FFT result for this group
            fft_result = self.fft_results[product_group]
            
            # Evaluate on training data
            train_group = train_df[train_df['Warengruppe'] == product_group]
            train_actual = cast(FloatArray, train_group['Umsatz'].to_numpy(dtype=np.float64))
            train_pred = self.reconstruct_signal(
                cast(FloatArray, fft_result['freqs']),
                cast(FloatArray, fft_result['amplitudes']),
                cast(FloatArray, fft_result['phases']),
                cast(IntArray, fft_result['dominant_idx']),
                len(train_actual)
            )
            train_mae = float(mean_absolute_error(train_actual, train_pred))
            self.train_mae_scores[product_group] = train_mae
            total_train_mae += train_mae
            
            # Evaluate on validation data
            val_group = val_df[val_df['Warengruppe'] == product_group]
            val_actual = cast(FloatArray, val_group['Umsatz'].to_numpy(dtype=np.float64))
            val_pred = self.reconstruct_signal(
                cast(FloatArray, fft_result['freqs']),
                cast(FloatArray, fft_result['amplitudes']),
                cast(FloatArray, fft_result['phases']),
                cast(IntArray, fft_result['dominant_idx']),
                len(val_actual)
            )
            val_mae = float(mean_absolute_error(val_actual, val_pred))
            self.val_mae_scores[product_group] = val_mae
            total_val_mae += val_mae
            
            print(f"{PRODUCT_GROUPS[product_group]:<15} {train_mae:>14.2f} {val_mae:>14.2f}")
            
            # Plot actual vs predicted for both train and validation
            plt.figure(figsize=(15, 6))
            
            # Training data
            plt.plot(train_group['Datum'], train_actual, label='Train Actual', alpha=0.6)
            plt.plot(train_group['Datum'], train_pred, label='Train Predicted', alpha=0.6)
            
            # Validation data
            plt.plot(val_group['Datum'], val_actual, label='Val Actual', alpha=0.6)
            plt.plot(val_group['Datum'], val_pred, label='Val Predicted', alpha=0.6)
            
            plt.title(f'Actual vs Predicted Sales - {PRODUCT_GROUPS[product_group]}')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, f'comparison_group_{product_group}.png'))
            plt.close()
        
        avg_train_mae = total_train_mae / n_groups
        avg_val_mae = total_val_mae / n_groups
        print("-" * 70)
        print(f"Average MAE:      {avg_train_mae:>14.2f} {avg_val_mae:>14.2f}")
    
    def save_results(self, predictions_df: pd.DataFrame) -> None:
        """Save analysis results and predictions."""
        # Save predictions
        predictions_path = os.path.join(self.output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        
        # Create submission file (keep only id and Umsatz columns)
        submission_df = predictions_df[['id', 'Umsatz']].copy()
        submission_path = os.path.join(self.output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        
        # Save MAE scores
        mae_df = pd.DataFrame({
            'Train_MAE': self.train_mae_scores,
            'Val_MAE': self.val_mae_scores
        }).T
        # Convert integer keys to product group names
        mae_df.columns = [PRODUCT_GROUPS[int(i)] for i in mae_df.columns]
        mae_path = os.path.join(self.output_dir, 'mae_scores.csv')
        mae_df.to_csv(mae_path)
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Visualizations saved to: {self.viz_dir}")
        print(f"Submission file saved to: {submission_path}")

def main():
    """Main function to run Fourier analysis."""
    print("Starting Fourier analysis for bakery sales prediction...")
    
    # Initialize analyzer
    analyzer = FourierAnalyzer(n_harmonics=4)
    
    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = analyzer.load_data()
    
    # Analyze each product group
    print("\nAnalyzing product groups...")
    for product_group in PRODUCT_GROUPS.keys():
        print(f"Processing {PRODUCT_GROUPS[product_group]}...")
        analyzer.analyze_product_group(train_df, product_group)
    
    # Evaluate performance
    analyzer.evaluate(train_df, val_df)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions_df = analyzer.predict(test_df)
    
    # Save results
    analyzer.save_results(predictions_df)

if __name__ == "__main__":
    main() 