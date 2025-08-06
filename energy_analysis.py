import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend for Render/servers
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Global Matplotlib defaults
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (8, 6)


# ==============================
# Helper: Downsample large arrays
# ==============================
def downsample_data(x, y=None, max_points=5000):
    """Downsample data for plotting without losing overall shape."""
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        if y is not None:
            return x[idx], y[idx]
        return x[idx]
    return (x, y) if y is not None else x


class EnergyDataAnalyzer:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = None
        self.datetime_column = None
        self.analysis_results = {}

    # ==============================
    # Load & Prepare Data
    # ==============================
    def load_and_prepare_data(self):
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(self.df)} rows from {self.csv_file_path}")

            # Convert all numeric columns to float32 for memory efficiency
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].astype('float32')

            # Auto-detect datetime column
            datetime_candidates = [col for col in self.df.columns
                                   if any(k in col.lower() for k in ['date', 'time', 'datetime', 'timestamp'])]
            if datetime_candidates:
                self.datetime_column = datetime_candidates[0]
                self.df[self.datetime_column] = pd.to_datetime(self.df[self.datetime_column])

            # Auto-detect target column
            target_candidates = [col for col in self.df.columns
                                 if any(k in col.lower() for k in ['power', 'energy', 'output', 'active', 'generated'])]
            if target_candidates:
                self.target_column = target_candidates[0]
            else:
                # fallback: use last numeric column
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.target_column = numeric_cols[-1]

            # Feature columns = all numeric except target
            self.feature_columns = [col for col in self.df.select_dtypes(include=[np.number]).columns
                                    if col != self.target_column]

            return True

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    # ==============================
    # Rolling Averages (optimized)
    # ==============================
    def create_rolling_averages(self, window_sizes=[7, 30]):
        if self.target_column is None:
            return

        for window in window_sizes:
            col_name = f'rolling_{window}'
            self.df[col_name] = (
                self.df[self.target_column]
                .rolling(window=window, min_periods=1)
                .mean()
                .astype('float32')
            )
            self.feature_columns.append(col_name)

    # ==============================
    # Train Linear Regression
    # ==============================
    def train_linear_regression(self, test_size=0.2, random_state=42):
        if not self.feature_columns or self.target_column is None:
            return False

        try:
            X = self.df[self.feature_columns].fillna(0).to_numpy(dtype=np.float32)
            y = self.df[self.target_column].to_numpy(dtype=np.float32)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.analysis_results['linear_regression'] = {
                'mse': float(mse),
                'r2': float(r2),
                'feature_importance': dict(zip(self.feature_columns, self.model.coef_)),
                'predictions': {
                    'actual': y_test,
                    'predicted': y_pred
                }
            }

            # Explicit cleanup
            del X, y, X_train, X_test, y_train, y_test
            gc.collect()

            return True

        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False

    # ==============================
    # Plot Functions
    # ==============================
    def generate_plots(self, output_dir='plots', lightweight_mode=False):
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Generate only key plots in lightweight mode
            self._plot_regression_results(output_dir, lightweight_mode)
            self._plot_rolling_averages(output_dir)
            self._plot_anomaly_detection(output_dir)

        except Exception as e:
            print(f"Error generating plots: {str(e)}")
        finally:
            plt.close('all')
            gc.collect()

    def _plot_regression_results(self, output_dir, lightweight_mode=False):
        results = self.analysis_results.get('linear_regression')
        if not results:
            return

        actual = results['predictions']['actual']
        predicted = results['predictions']['predicted']

        # Downsample
        actual_ds, predicted_ds = downsample_data(actual, predicted, max_points=5000)

        plt.figure(figsize=(10, 6))
        plt.scatter(actual_ds, predicted_ds, alpha=0.5, s=15)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Prediction vs Actual')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/regression_results.png', dpi=120, bbox_inches='tight')
        plt.close()

        if not lightweight_mode:
            # Residuals Histogram (skip in lightweight mode)
            residuals = actual - predicted
            plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Residuals Distribution')
            plt.xlabel('Residual')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/residuals_histogram.png', dpi=120, bbox_inches='tight')
            plt.close()

    def _plot_rolling_averages(self, output_dir):
        rolling_cols = [col for col in self.df.columns if col.startswith('rolling_')]

        if rolling_cols and self.datetime_column:
            plt.figure(figsize=(12, 6))

            # Downsample time series
            time_ds, value_ds = downsample_data(
                self.df[self.datetime_column].to_numpy(),
                self.df[self.target_column].to_numpy()
            )

            plt.plot(time_ds, value_ds, alpha=0.4, label='Original', color='gray')

            for col in rolling_cols:
                _, roll_ds = downsample_data(
                    self.df[self.datetime_column].to_numpy(),
                    self.df[col].to_numpy()
                )
                plt.plot(time_ds, roll_ds, label=col.replace('_', ' ').title())

            plt.legend()
            plt.title(f'{self.target_column} with Rolling Averages')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/rolling_averages.png', dpi=120, bbox_inches='tight')
            plt.close()

    def _plot_anomaly_detection(self, output_dir):
        if self.target_column is None:
            return

        data = self.df[self.target_column].dropna()
        mean_val = data.mean()
        std_val = data.std()

        upper_threshold = mean_val + 2 * std_val
        lower_threshold = mean_val - 2 * std_val

        anomalies = self.df[
            (self.df[self.target_column] > upper_threshold) |
            (self.df[self.target_column] < lower_threshold)
        ]

        plt.figure(figsize=(12, 6))
        if self.datetime_column:
            x_data = self.df[self.datetime_column]
        else:
            x_data = np.arange(len(data))

        x_ds, y_ds = downsample_data(x_data.to_numpy() if hasattr(x_data, 'to_numpy') else x_data,
                                     data.to_numpy())

        plt.plot(x_ds, y_ds, alpha=0.6, label='Data')
        plt.axhline(y=upper_threshold, color='red', linestyle='--', label='Upper Threshold')
        plt.axhline(y=lower_threshold, color='red', linestyle='--', label='Lower Threshold')
        plt.axhline(y=mean_val, color='green', linestyle='-', label='Mean')

        if not anomalies.empty:
            plt.scatter(anomalies.index, anomalies[self.target_column],
                        color='red', s=20, label='Anomalies')

        plt.legend()
        plt.title('Anomaly Detection')
        plt.xticks(rotation=45 if self.datetime_column else 0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/anomaly_detection.png', dpi=120, bbox_inches='tight')
        plt.close()

    # ==============================
    # Summary Stats
    # ==============================
    def generate_summary_stats(self):
        if self.df is None:
            return {}

        stats = {
            'data_points': len(self.df),
            'features': len(self.feature_columns),
            'target_column': self.target_column,
            'datetime_column': self.datetime_column,
            'date_range': None,
            'target_stats': {}
        }

        if self.datetime_column:
            stats['date_range'] = {
                'start': str(self.df[self.datetime_column].min().date()),
                'end': str(self.df[self.datetime_column].max().date())
            }

        if self.target_column:
            target_data = self.df[self.target_column]
            stats['target_stats'] = {
                'mean': float(target_data.mean()),
                'std': float(target_data.std()),
                'min': float(target_data.min()),
                'max': float(target_data.max()),
                'median': float(target_data.median())
            }

        if 'linear_regression' in self.analysis_results:
            stats['model_performance'] = {
                'mse': self.analysis_results['linear_regression']['mse'],
                'r2': self.analysis_results['linear_regression']['r2']
            }

        return stats


# ==============================
# Utility: Clean NaN values for JSON serialization
# ==============================
def clean_nan_values(obj):
    """
    Recursively replace NaNs in dicts/lists/arrays with None for safe JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]

    elif isinstance(obj, np.ndarray):
        return [clean_nan_values(v) for v in obj.tolist()]

    else:
        if pd.isna(obj):
            return None
        return obj


# ==============================
# Wrapper function
# ==============================
def analyze_energy_csv(csv_file_path, output_dir='analysis_output', lightweight_mode=False):
    try:
        analyzer = EnergyDataAnalyzer(csv_file_path)

        if not analyzer.load_and_prepare_data():
            return {'error': 'Failed to load data'}

        analyzer.create_rolling_averages()
        analyzer.train_linear_regression()
        stats = analyzer.generate_summary_stats()

        try:
            analyzer.generate_plots(output_dir, lightweight_mode)
        except Exception as plot_error:
            print(f"Warning: Chart generation failed: {str(plot_error)}")

        # Clean before returning (fixes Render JSON issues)
        return clean_nan_values({
            'analysis_results': analyzer.analysis_results,
            'stats': stats,
            'output_dir': output_dir,
            'lightweight_mode': lightweight_mode
        })

    except MemoryError:
        gc.collect()
        return {'error': 'Memory limit exceeded. Try using lightweight mode.'}
    except Exception as e:
        return {'error': str(e)}


# CLI entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        results = analyze_energy_csv(csv_file)
        print("Analysis completed!")
        print(f"Results: {results}")
    else:
        print("Usage: python energy_analysis.py <csv_file_path>")
