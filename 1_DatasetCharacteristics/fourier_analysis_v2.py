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
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Union

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
        """Initialize the Fourier analyzer."""
        self.n_harmonics = n_harmonics
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output', 'fourier_analysis')
        self.viz_dir = os.path.join(os.path.dirname(__file__), 'visualizations', 'fourier_analysis')

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        # Store FFT results
        self.fft_results: Dict[int, Dict[str, Union[np.ndarray, np.ndarray]]] = {}
        self.predictions: Dict[int, np.ndarray] = {}
        self.mae_scores: Dict[int, float] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data."""
        train_path = os.path.join(os.path.dirname(__file__), '..', '0_DataPreparation', 'input', 'competition_data', 'train.csv')
        train_df = pd.read_csv(train_path)
        train_df['Datum'] = pd.to_datetime(train_df['Datum'])

        test_path = os.path.join(os.path.dirname(__file__), '..', '0_DataPreparation', 'input', 'competition_data', 'sample_submission.csv')
        test_df = pd.read_csv(test_path)
        test_df['Datum'] = pd.to_datetime('20' + test_df['id'].astype(str).str[:6], format='%Y%m%d')
        test_df['Warengruppe'] = test_df['id'].astype(str).str[-1].astype(int)

        return train_df, test_df

    def apply_fft(self, sales: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply FFT to sales data and return frequencies and amplitudes."""
        n = len(sales)
        fft_vals = fft(sales)
        freqs = fftfreq(n, d=1)  # Assumes daily data
        amplitudes = np.abs(fft_vals)
        phases = np.angle(fft_vals)

        return freqs, amplitudes, phases

    def get_dominant_frequencies(self, amplitudes: np.ndarray) -> np.ndarray:
        """Get indices of dominant frequencies (excluding DC component)."""
        sorted_indices = np.argsort(amplitudes[1:])[::-1] + 1
        return sorted_indices[:self.n_harmonics]

    def reconstruct_signal(self, freqs: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray, dominant_idx: np.ndarray, n_points: int) -> np.ndarray:
        """Reconstruct signal using dominant frequencies."""
        reconstructed = np.zeros(n_points, dtype=complex)

        for idx in dominant_idx:
            reconstructed += amplitudes[idx] * np.exp(1j * (2 * np.pi * freqs[idx] * np.arange(n_points) + phases[idx]))

        return np.real(reconstructed)

    def analyze_product_group(self, train_df: pd.DataFrame, product_group: int) -> None:
        """Analyze sales patterns for a specific product group."""
        group_data = train_df[train_df['Warengruppe'] == product_group].sort_values('Datum')
        sales = group_data['Umsatz'].values

        freqs, amplitudes, phases = self.apply_fft(sales)
        dominant_idx = self.get_dominant_frequencies(amplitudes)

        self.fft_results[product_group] = {
            'freqs': freqs,
            'amplitudes': amplitudes,
            'phases': phases,
            'dominant_idx': dominant_idx
        }

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
        predictions_df['Umsatz'] = 0.0

        for product_group in PRODUCT_GROUPS.keys():
            fft_result = self.fft_results[product_group]
            n_points = len(test_df[test_df['Warengruppe'] == product_group])
            predictions = self.reconstruct_signal(
                fft_result['freqs'], fft_result['amplitudes'], fft_result['phases'], fft_result['dominant_idx'], n_points
            )
            predictions_df.loc[predictions_df['Warengruppe'] == product_group, 'Umsatz'] = predictions

        predictions_df['Umsatz'] = predictions_df['Umsatz'].clip(lower=0)
        return predictions_df

    def evaluate(self, train_df: pd.DataFrame) -> None:
        """Evaluate model performance on training data."""
        for product_group in PRODUCT_GROUPS.keys():
            group_data = train_df[train_df['Warengruppe'] == product_group]
            actual_sales = group_data['Umsatz'].values
            fft_result = self.fft_results[product_group]
            reconstructed = self.reconstruct_signal(
                fft_result['freqs'], fft_result['amplitudes'], fft_result['phases'], fft_result['dominant_idx'], len(actual_sales)
            )
            mae = mean_absolute_error(actual_sales, reconstructed)
            self.mae_scores[product_group] = mae
            print(f"{PRODUCT_GROUPS[product_group]}: MAE = {mae:.2f}")

    def save_results(self, predictions_df: pd.DataFrame) -> None:
        predictions_df.to_csv(os.path.join(self.output_dir, 'predictions.csv'), index=False)
        mae_df = pd.DataFrame.from_dict(self.mae_scores, orient='index', columns=['MAE'])
        mae_df.to_csv(os.path.join(self.output_dir, 'mae_scores.csv'))

def main():
    analyzer = FourierAnalyzer(n_harmonics=4)
    train_df, test_df = analyzer.load_data()

    for product_group in PRODUCT_GROUPS.keys():
        analyzer.analyze_product_group(train_df, product_group)

    analyzer.evaluate(train_df)
    predictions_df = analyzer.predict(test_df)
    analyzer.save_results(predictions_df)

if __name__ == "__main__":
    main()