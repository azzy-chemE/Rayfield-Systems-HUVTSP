import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import gc
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (8, 6)

def downsample_data(x, y=None, max_points=5000):
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

    def load_and_prepare_data(self):
        try:
            self.df = pd.read_csv(self.csv_file_path)

            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].astype('float32')

            datetime_candidates = [col for col in self.df.columns
                                   if any(k in col.lower() for k in ['date', 'time', 'datetime', 'timestamp'])]
            if datetime_candidates:
                self.datetime_column = datetime_candidates[0]
                self.df[self.datetime_column] = pd.to_datetime(self.df[self.datetime_column])

            target_candidates = [col for col in self.df.columns
                                  if any(k in col.lower() for k in ['power', 'energy', 'output', 'active', 'generated'])]
            if target_candidates:
                self.target_column = target_candidates[0]
            else:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.target_column = numeric_cols[-1]

            self.feature_columns = [col for col in self.df.select_dtypes(include=[np.number]).columns
                                    if col != self.target_column]

            return True

        except Exception as e:
            return False

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

            del X, y, X_train, X_test, y_train, y_test
            gc.collect()

            return True

        except Exception as e:
            return False

    def generate_plots(self, output_dir='plots'):
        try:
            os.makedirs(output_dir, exist_ok=True)

            self._plot_regression_results(output_dir)
            self._plot_rolling_averages(output_dir)
            self._plot_anomaly_detection(output_dir)

        except Exception as e:
            pass
        finally:
            plt.close('all')
            gc.collect()

    def _plot_regression_results(self, output_dir):
        results = self.analysis_results.get('linear_regression')
        if not results:
            return

        actual = results['predictions']['actual']
        predicted = results['predictions']['predicted']

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

        # Always generate residuals histogram
        residuals = actual - predicted
        plt.figure(figsize=(8, 6))
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

    def _generate_anomalies_data(self):
        """Generate anomalies data independently of chart generation"""
        if self.target_column is None:
            print("DEBUG: No target column found for anomaly detection")
            return False

        data = self.df[self.target_column].dropna()
        mean_val = data.mean()
        std_val = data.std()

        upper_threshold = mean_val + 2 * std_val
        lower_threshold = mean_val - 2 * std_val

        anomalies = self.df[
            (self.df[self.target_column] > upper_threshold) |
            (self.df[self.target_column] < lower_threshold)
        ]
        
        print(f"DEBUG: Anomaly detection - Mean: {mean_val:.2f}, Std: {std_val:.2f}")
        print(f"DEBUG: Anomaly detection - Upper threshold: {upper_threshold:.2f}, Lower threshold: {lower_threshold:.2f}")
        print(f"DEBUG: Anomaly detection - Found {len(anomalies)} anomalies")
        print(f"DEBUG: Anomaly detection - Data shape: {self.df.shape}")
        print(f"DEBUG: Anomaly detection - Target column: {self.target_column}")
        
        # Store anomalies data for later use
        print(f"DEBUG: Storing anomalies data - {len(anomalies)} anomalies found")
        self.anomalies_data = {
            'anomalies': anomalies,
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'mean_val': mean_val,
            'std_val': std_val
        }
        print(f"DEBUG: Anomalies data stored successfully")
        print(f"DEBUG: Anomalies data keys: {list(self.anomalies_data.keys())}")
        print(f"DEBUG: Anomalies DataFrame shape: {self.anomalies_data['anomalies'].shape}")
        
        return True

    def _plot_anomaly_detection(self, output_dir):
        """Generate anomaly detection plot (requires anomalies data to be generated first)"""
        if not hasattr(self, 'anomalies_data') or self.anomalies_data['anomalies'].empty:
            print("DEBUG: No anomalies data available for plotting")
            return

        data = self.df[self.target_column].dropna()
        anomalies = self.anomalies_data['anomalies']
        mean_val = self.anomalies_data['mean_val']
        upper_threshold = self.anomalies_data['upper_threshold']
        lower_threshold = self.anomalies_data['lower_threshold']

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

    def get_anomalies_table(self):
        """Generate a table of anomalies sorted by x values (increasing)"""
        print("DEBUG: get_anomalies_table called")
        
        if not hasattr(self, 'anomalies_data'):
            print("DEBUG: No anomalies_data attribute found")
            return None
        
        print(f"DEBUG: anomalies_data keys: {list(self.anomalies_data.keys())}")
        
        if self.anomalies_data['anomalies'].empty:
            print("DEBUG: Anomalies DataFrame is empty")
            return None
        
        print(f"DEBUG: Found {len(self.anomalies_data['anomalies'])} anomalies")
        
        anomalies = self.anomalies_data['anomalies']
        print(f"DEBUG: Anomalies DataFrame columns: {list(anomalies.columns)}")
        print(f"DEBUG: Anomalies DataFrame head:")
        print(anomalies.head())
        
        # Create table data
        table_data = []
        
        for idx, row in anomalies.iterrows():
            # Get x value (timestamp or index)
            if self.datetime_column:
                x_value = row[self.datetime_column]
                x_str = str(x_value)
            else:
                x_value = idx
                x_str = f"Index {idx}"
            
            # Get y value (target column value)
            y_value = row[self.target_column]
            
            # Determine if it's above or below threshold
            if y_value > self.anomalies_data['upper_threshold']:
                threshold_type = "Above Upper"
                threshold_value = self.anomalies_data['upper_threshold']
            else:
                threshold_type = "Below Lower"
                threshold_value = self.anomalies_data['lower_threshold']
            
            # Calculate deviation from mean
            deviation = y_value - self.anomalies_data['mean_val']
            deviation_percent = (deviation / self.anomalies_data['mean_val']) * 100
            
            table_data.append({
                'x_value': x_value,
                'x_str': x_str,
                'y_value': float(y_value),
                'threshold_type': threshold_type,
                'threshold_value': float(threshold_value),
                'deviation': float(deviation),
                'deviation_percent': float(deviation_percent)
            })
        
        # Sort by x_value (increasing)
        table_data.sort(key=lambda x: x['x_value'])
        
        print(f"DEBUG: Created {len(table_data)} table entries")
        
        result = {
            'table_data': table_data,
            'total_anomalies': len(table_data),
            'upper_threshold': float(self.anomalies_data['upper_threshold']),
            'lower_threshold': float(self.anomalies_data['lower_threshold']),
            'mean_value': float(self.anomalies_data['mean_val']),
            'std_value': float(self.anomalies_data['std_val'])
        }
        
        print(f"DEBUG: Returning anomalies table with {len(table_data)} anomalies")
        return result

def clean_nan_values(obj):
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return [clean_nan_values(v) for v in obj.tolist()]
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):  # numpy/pandas types
        if pd.isna(obj):
            return None
        else:
            return float(obj) if hasattr(obj, 'item') else obj
    elif hasattr(obj, 'strftime'):  # Handle datetime/timestamp objects
        return str(obj)
    else:
        if pd.isna(obj):
            return None
        return obj

def analyze_energy_csv(csv_file_path, output_dir='analysis_output'):
    try:
        analyzer = EnergyDataAnalyzer(csv_file_path)

        if not analyzer.load_and_prepare_data():
            return {'error': 'Failed to load data'}

        analyzer.create_rolling_averages()
        analyzer.train_linear_regression()
        stats = analyzer.generate_summary_stats()

        # Generate anomalies data first (independent of charts)
        print("DEBUG: Generating anomalies data...")
        anomalies_generated = analyzer._generate_anomalies_data()
        print(f"DEBUG: Anomalies data generation: {'success' if anomalies_generated else 'failed'}")

        # Get anomalies table
        print("DEBUG: About to call get_anomalies_table()")
        anomalies_table = analyzer.get_anomalies_table()
        print(f"DEBUG: get_anomalies_table() returned: {anomalies_table is not None}")
        
        if anomalies_table:
            print(f"DEBUG: Anomalies table has {anomalies_table.get('total_anomalies', 0)} anomalies")
        else:
            print("DEBUG: No anomalies table generated")

        # Generate charts (optional - won't affect anomalies table)
        try:
            analyzer.generate_plots(output_dir)
        except Exception as e:
            print(f"Warning: Chart generation failed: {e}")
            print("Anomalies table was still generated successfully")

        return clean_nan_values({
            'analysis_results': analyzer.analysis_results,
            'stats': stats,
            'output_dir': output_dir,
            'anomalies_table': anomalies_table
        })

    except MemoryError:
        gc.collect()
        return {'error': 'Memory limit exceeded.'}
    except Exception as e:
        return {'error': str(e)}

def analyze_energy_csv_quick(csv_file_path):
    """
    Quick analysis function that generates charts without heavy processing
    """
    try:
        analyzer = EnergyDataAnalyzer(csv_file_path)

        if not analyzer.load_and_prepare_data():
            return {'error': 'Failed to load data'}

        analyzer.create_rolling_averages()
        analyzer.train_linear_regression()
        stats = analyzer.generate_summary_stats()

        # Generate anomalies data first (independent of charts)
        print("DEBUG: Quick analysis - Generating anomalies data...")
        anomalies_generated = analyzer._generate_anomalies_data()
        print(f"DEBUG: Quick analysis - Anomalies data generation: {'success' if anomalies_generated else 'failed'}")

        # Get anomalies table
        print("DEBUG: Quick analysis - About to call get_anomalies_table()")
        anomalies_table = analyzer.get_anomalies_table()
        print(f"DEBUG: Quick analysis - get_anomalies_table() returned: {anomalies_table is not None}")
        
        if anomalies_table:
            print(f"DEBUG: Quick analysis - Anomalies table has {anomalies_table.get('total_anomalies', 0)} anomalies")
        else:
            print("DEBUG: Quick analysis - No anomalies table generated")

        # Generate charts for quick analysis (optional)
        try:
            analyzer.generate_plots('static/charts')
        except Exception as plot_error:
            print(f"Warning: Chart generation failed: {str(plot_error)}")
            print("Anomalies table was still generated successfully")

        return clean_nan_values({
            'analysis_results': analyzer.analysis_results,
            'stats': stats,
            'output_dir': 'static/charts',
            'anomalies_table': anomalies_table
        })

    except MemoryError:
        gc.collect()
        return {'error': 'Memory limit exceeded during quick analysis.'}
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        results = analyze_energy_csv(csv_file)
        print("Analysis completed!")
        print(f"Results: {results}")
    else:
        print("Usage: python energy_analysis.py <csv_file_path>")
