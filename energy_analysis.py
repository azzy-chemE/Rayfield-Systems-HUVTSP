import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (8, 6)

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
            print(f"Loaded {len(self.df)} rows from {self.csv_file_path}")
            
            # Auto-detect datetime column
            datetime_candidates = [col for col in self.df.columns 
                                if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp'])]
            
            if datetime_candidates:
                self.datetime_column = datetime_candidates[0]
                self.df[self.datetime_column] = pd.to_datetime(self.df[self.datetime_column])
            
            # Auto-detect target column
            target_candidates = [col for col in self.df.columns 
                              if any(keyword in col.lower() for keyword in ['power', 'energy', 'output', 'active', 'generated'])]
            
            if target_candidates:
                self.target_column = target_candidates[0]
            else:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.target_column = numeric_cols[-1]
            
            # Auto-detect feature columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in numeric_cols:
                numeric_cols.remove(self.target_column)
            
            self.feature_columns = numeric_cols
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def create_rolling_averages(self, window_sizes=[7, 30]):
        if self.target_column is None:
            return
        
        for window in window_sizes:
            col_name = f'rolling_{window}'
            self.df[col_name] = self.df[self.target_column].rolling(window).mean().fillna(0)
            self.feature_columns.append(col_name)
    
    def train_linear_regression(self, test_size=0.2, random_state=42):
        if not self.feature_columns or self.target_column is None:
            return False
        
        try:
            X = self.df[self.feature_columns].fillna(0)
            y = self.df[self.target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.analysis_results['linear_regression'] = {
                'mse': mse,
                'r2': r2,
                'feature_importance': dict(zip(self.feature_columns, self.model.coef_)),
                'predictions': {
                    'actual': y_test.values,
                    'predicted': y_pred
                }
            }
            
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def _create_simple_chart(self, output_dir, chart_name, data, title, xlabel, ylabel):
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(data)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{chart_name}.png', dpi=100, bbox_inches='tight')
            plt.close()
            return True
        except Exception as e:
            return False

    def generate_plots(self, output_dir='plots', lightweight_mode=False):
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Only generate the 3 most important charts to reduce memory load
            plot_functions = [
                self._plot_regression_results,
                self._plot_rolling_averages,
                self._plot_anomaly_detection
            ]
            
            for plot_func in plot_functions:
                try:
                    plot_func(output_dir)
                except Exception as e:
                    print(f"Warning: Failed to generate {plot_func.__name__}: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Error generating plots: {str(e)}")
        finally:
            plt.close('all')
            gc.collect()
    

    
    def _plot_regression_results(self, output_dir):
        results = self.analysis_results['linear_regression']
        actual = results['predictions']['actual']
        predicted = results['predictions']['predicted']
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with regression line
        plt.subplot(2, 3, 1)
        plt.scatter(actual, predicted, alpha=0.6, s=30, color='blue')
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction vs Actual')
        plt.legend()
        
        # Residuals plot
        plt.subplot(2, 3, 2)
        residuals = actual - predicted
        plt.scatter(predicted, residuals, alpha=0.6, s=30, color='green')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Histogram of residuals
        plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=25, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        # Model performance metrics
        plt.subplot(2, 3, 4)
        metrics_text = f'Linear Regression Model Performance\n\n' \
                      f'MSE: {results["mse"]:.4f}\n' \
                      f'R² Score: {results["r2"]:.4f}\n' \
                      f'Features Used: {len(self.feature_columns)}\n' \
                      f'Data Points: {len(actual)}\n' \
                      f'Mean Residual: {residuals.mean():.4f}\n' \
                      f'Std Residual: {residuals.std():.4f}'
        plt.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.axis('off')
        plt.title('Model Performance Metrics')
        
        # Feature importance (top features)
        plt.subplot(2, 3, 5)
        importance = results['feature_importance']
        if importance:
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            features, importances = zip(*sorted_features)
            
            colors = ['red' if x < 0 else 'blue' for x in importances]
            plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
            plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
            plt.xlabel('Coefficient Value')
            plt.title('Top Feature Importance')
        
        # Prediction accuracy over range
        plt.subplot(2, 3, 6)
        value_ranges = np.linspace(actual.min(), actual.max(), 5)
        accuracies = []
        for i in range(len(value_ranges)-1):
            mask = (actual >= value_ranges[i]) & (actual < value_ranges[i+1])
            if mask.sum() > 0:
                range_residuals = residuals[mask]
                accuracy = 1 - (np.abs(range_residuals) / actual[mask]).mean()
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        plt.bar(range(len(accuracies)), accuracies, alpha=0.7, color='purple')
        plt.xlabel('Value Range')
        plt.ylabel('Prediction Accuracy')
        plt.title('Accuracy by Value Range')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/regression_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    

    
    def _plot_rolling_averages(self, output_dir):
        rolling_cols = [col for col in self.df.columns if col.startswith('rolling_')]
        
        if rolling_cols and self.datetime_column:
            plt.figure(figsize=(14, 8))
            
            plt.plot(self.df[self.datetime_column], self.df[self.target_column], 
                    alpha=0.5, label='Original Data', color='gray', linewidth=1)
            
            colors = ['red', 'blue', 'green', 'orange']
            for i, col in enumerate(rolling_cols):
                plt.plot(self.df[self.datetime_column], self.df[col], 
                        label=col.replace('_', ' ').title(), 
                        color=colors[i % len(colors)], linewidth=2)
            
            mean_val = self.df[self.target_column].mean()
            median_val = self.df[self.target_column].median()
            plt.axhline(y=mean_val, color='purple', linestyle='--', alpha=0.8, 
                       label=f'Mean: {mean_val:.2f}')
            plt.axhline(y=median_val, color='brown', linestyle='--', alpha=0.8, 
                       label=f'Median: {median_val:.2f}')
            
            stats_text = f'Statistics:\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {self.df[self.target_column].std():.2f}\nMin: {self.df[self.target_column].min():.2f}\nMax: {self.df[self.target_column].max():.2f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title(f'{self.target_column} with Rolling Averages and Statistics')
            plt.xlabel('Time')
            plt.ylabel(self.target_column)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/rolling_averages.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_anomaly_detection(self, output_dir):
        if self.target_column is None:
            return
            
        try:
            plt.figure(figsize=(12, 8))
            
            data = self.df[self.target_column].dropna()
            mean_val = data.mean()
            std_val = data.std()
            
            upper_threshold = mean_val + 2 * std_val
            lower_threshold = mean_val - 2 * std_val
            
            if self.datetime_column:
                plt.plot(self.df[self.datetime_column], self.df[self.target_column], 
                        alpha=0.7, label='Original Data', color='blue', linewidth=1)
            else:
                plt.plot(self.df[self.target_column].values, alpha=0.7, 
                        label='Original Data', color='blue', linewidth=1)
            
            anomalies = self.df[
                (self.df[self.target_column] > upper_threshold) | 
                (self.df[self.target_column] < lower_threshold)
            ]
            
            if len(anomalies) > 0:
                if self.datetime_column:
                    plt.scatter(anomalies[self.datetime_column], anomalies[self.target_column], 
                               color='red', s=50, label=f'Anomalies ({len(anomalies)})', zorder=5)
                else:
                    anomaly_indices = anomalies.index
                    plt.scatter(anomaly_indices, anomalies[self.target_column], 
                               color='red', s=50, label=f'Anomalies ({len(anomalies)})', zorder=5)
            
            if self.datetime_column:
                plt.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, 
                           label=f'Upper Threshold (+2σ): {upper_threshold:.2f}')
                plt.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7, 
                           label=f'Lower Threshold (-2σ): {lower_threshold:.2f}')
            else:
                plt.axhline(y=upper_threshold, color='red', linestyle='--', alpha=0.7, 
                           label=f'Upper Threshold (+2σ): {upper_threshold:.2f}')
                plt.axhline(y=lower_threshold, color='red', linestyle='--', alpha=0.7, 
                           label=f'Lower Threshold (-2σ): {lower_threshold:.2f}')
            
            plt.axhline(y=mean_val, color='green', linestyle='-', alpha=0.8, 
                       label=f'Mean: {mean_val:.2f}')
            
            stats_text = f'Anomaly Detection Stats:\nMean: {mean_val:.2f}\nStd: {std_val:.2f}\nAnomalies: {len(anomalies)}\nThreshold: ±2σ'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title(f'Anomaly Detection for {self.target_column}')
            plt.xlabel('Time' if self.datetime_column else 'Data Point')
            plt.ylabel(self.target_column)
            plt.legend()
            if self.datetime_column:
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/anomaly_detection.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self._create_simple_chart(output_dir, 'anomaly_detection', 
                                   self.df[self.target_column].values if self.target_column else [0], 
                                   'Anomaly Detection', 'Index', self.target_column or 'Value')
    

    

    
    def save_model(self, model_path='energy_model.pkl'):
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'analysis_results': self.analysis_results
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
    
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
                'start': self.df[self.datetime_column].min().strftime('%Y-%m-%d'),
                'end': self.df[self.datetime_column].max().strftime('%Y-%m-%d')
            }
        
        if self.target_column:
            mean_val = self.df[self.target_column].mean()
            std_val = self.df[self.target_column].std()
            min_val = self.df[self.target_column].min()
            max_val = self.df[self.target_column].max()
            median_val = self.df[self.target_column].median()
            
            stats['target_stats'] = {
                'mean': float(mean_val) if not pd.isna(mean_val) else None,
                'std': float(std_val) if not pd.isna(std_val) else None,
                'min': float(min_val) if not pd.isna(min_val) else None,
                'max': float(max_val) if not pd.isna(max_val) else None,
                'median': float(median_val) if not pd.isna(median_val) else None
            }
        
        if 'linear_regression' in self.analysis_results:
            mse_val = self.analysis_results['linear_regression']['mse']
            r2_val = self.analysis_results['linear_regression']['r2']
            
            stats['model_performance'] = {
                'mse': float(mse_val) if not pd.isna(mse_val) else None,
                'r2': float(r2_val) if not pd.isna(r2_val) else None
            }
        
        return stats

def analyze_energy_csv(csv_file_path, output_dir='analysis_output', lightweight_mode=False):
    try:
        analyzer = EnergyDataAnalyzer(csv_file_path)
        
        if not analyzer.load_and_prepare_data():
            return {'error': 'Failed to load data'}
        
        analyzer.create_rolling_averages()
        model_results = analyzer.train_linear_regression()
        stats = analyzer.generate_summary_stats()
        
        try:
            analyzer.generate_plots(output_dir, lightweight_mode)
        except Exception as plot_error:
            print(f"Warning: Chart generation failed: {str(plot_error)}")
        
        analysis_results = {
            'linear_regression': model_results,
            'feature_importance': analyzer.analysis_results.get('feature_importance', {}),
            'summary_stats': stats
        }
        
        del analyzer
        gc.collect()
        
        return {
            'analysis_results': analysis_results,
            'stats': stats,
            'output_dir': output_dir,
            'lightweight_mode': lightweight_mode
        }
        
    except MemoryError as e:
        gc.collect()
        return {'error': 'Memory limit exceeded. Try using lightweight mode.'}
    except Exception as e:
        return {'error': str(e)}

def analyze_energy_csv_quick(csv_file_path):
    try:
        analyzer = EnergyDataAnalyzer(csv_file_path)
        
        if not analyzer.load_and_prepare_data():
            return {'error': 'Failed to load data'}
        
        analyzer.create_rolling_averages()
        model_results = analyzer.train_linear_regression()
        stats = analyzer.generate_summary_stats()
        
        analysis_results = {
            'linear_regression': model_results,
            'feature_importance': analyzer.analysis_results.get('feature_importance', {}),
            'summary_stats': stats
        }
        
        del analyzer
        gc.collect()
        
        return {
            'analysis_results': analysis_results,
            'stats': stats,
            'mode': 'quick'
        }
        
    except MemoryError as e:
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
