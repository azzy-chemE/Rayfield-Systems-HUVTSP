import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnergyDataAnalyzer:
    """
    Generalized energy data analyzer that can handle any energy maintenance CSV file
    """
    
    def __init__(self, csv_file_path):
        """
        Initialize the analyzer with a CSV file path
        
        Args:
            csv_file_path (str): Path to the CSV file containing energy data
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = None
        self.datetime_column = None
        self.analysis_results = {}
        
    def load_and_prepare_data(self):
        """
        Load and prepare the CSV data for analysis
        """
        try:
            # Load the CSV file
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(self.df)} rows from {self.csv_file_path}")
            
            # Display basic info
            print(f"Columns: {list(self.df.columns)}")
            print(f"Data types: {self.df.dtypes.to_dict()}")
            
            # Auto-detect datetime column
            datetime_candidates = [col for col in self.df.columns 
                                if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'timestamp'])]
            
            if datetime_candidates:
                self.datetime_column = datetime_candidates[0]
                self.df[self.datetime_column] = pd.to_datetime(self.df[self.datetime_column])
                print(f"Detected datetime column: {self.datetime_column}")
            
            # Auto-detect target column (assume it's the main energy/power column)
            target_candidates = [col for col in self.df.columns 
                              if any(keyword in col.lower() for keyword in ['power', 'energy', 'output', 'active', 'generated'])]
            
            if target_candidates:
                self.target_column = target_candidates[0]
                print(f"Detected target column: {self.target_column}")
            else:
                # If no obvious target, use the last numeric column
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.target_column = numeric_cols[-1]
                    print(f"Using last numeric column as target: {self.target_column}")
            
            # Auto-detect feature columns (all numeric columns except target and datetime)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in numeric_cols:
                numeric_cols.remove(self.target_column)
            
            self.feature_columns = numeric_cols
            print(f"Feature columns: {self.feature_columns}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def create_rolling_averages(self, window_sizes=[7, 30]):
        """
        Create rolling average features for time series analysis
        
        Args:
            window_sizes (list): List of window sizes for rolling averages
        """
        if self.target_column is None:
            print("No target column detected")
            return
        
        for window in window_sizes:
            col_name = f'rolling_{window}'
            self.df[col_name] = self.df[self.target_column].rolling(window).mean().fillna(0)
            self.feature_columns.append(col_name)
            print(f"Created rolling average feature: {col_name}")
    
    def train_linear_regression(self, test_size=0.2, random_state=42):
        """
        Train a linear regression model on the data
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        if not self.feature_columns or self.target_column is None:
            print("No features or target column available")
            return False
        
        try:
            # Prepare features and target
            X = self.df[self.feature_columns].fillna(0)
            y = self.df[self.target_column]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.analysis_results['linear_regression'] = {
                'mse': mse,
                'r2': r2,
                'feature_importance': dict(zip(self.feature_columns, self.model.coef_)),
                'predictions': {
                    'actual': y_test.values,
                    'predicted': y_pred
                }
            }
            
            print(f"Linear Regression Results:")
            print(f"MSE: {mse:.2f}")
            print(f"R²: {r2:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def generate_plots(self, output_dir='plots'):
        """
        Generate various plots for the energy data analysis
        
        Args:
            output_dir (str): Directory to save plots
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set style with memory optimization
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Reduce memory usage by using smaller figure sizes and lower DPI
        plt.rcParams['figure.dpi'] = 100  # Lower DPI for smaller files
        plt.rcParams['figure.figsize'] = (8, 6)  # Smaller default figure size
        
        try:
            # 1. Time series plot
            if self.datetime_column:
                self._plot_time_series(output_dir)
            
            # 2. Linear regression results
            if 'linear_regression' in self.analysis_results:
                self._plot_regression_results(output_dir)
            
            # 3. Feature importance
            if 'linear_regression' in self.analysis_results:
                self._plot_feature_importance(output_dir)
            
            # 4. Rolling averages
            if self.target_column:
                self._plot_rolling_averages(output_dir)
            
            # 5. Correlation heatmap
            self._plot_correlation_heatmap(output_dir)
            
            # 6. Distribution plots
            self._plot_distributions(output_dir)
            
            print(f"All plots saved to {output_dir}/")
            
        except Exception as e:
            print(f"Error generating plots: {str(e)}")
        finally:
            # Clean up matplotlib memory
            plt.close('all')
            import gc
            gc.collect()
    
    def _plot_time_series(self, output_dir):
        """Plot time series of the target variable"""
        plt.figure(figsize=(10, 5))
        # Sample data if too large to reduce memory usage
        if len(self.df) > 10000:
            sample_df = self.df.sample(n=10000, random_state=42)
        else:
            sample_df = self.df
            
        plt.plot(sample_df[self.datetime_column], sample_df[self.target_column], alpha=0.7, linewidth=0.5)
        plt.title(f'{self.target_column} Over Time')
        plt.xlabel('Time')
        plt.ylabel(self.target_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/time_series.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_regression_results(self, output_dir):
        """Plot regression results"""
        results = self.analysis_results['linear_regression']
        actual = results['predictions']['actual']
        predicted = results['predictions']['predicted']
        
        plt.figure(figsize=(8, 6))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(actual, predicted, alpha=0.5, s=20)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction vs Actual')
        
        # Residuals plot
        plt.subplot(2, 2, 2)
        residuals = actual - predicted
        plt.scatter(predicted, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Metrics text
        plt.subplot(2, 2, 3)
        plt.text(0.1, 0.8, f'MSE: {results["mse"]:.2f}', fontsize=10)
        plt.text(0.1, 0.6, f'R²: {results["r2"]:.4f}', fontsize=10)
        plt.text(0.1, 0.4, f'Features: {len(self.feature_columns)}', fontsize=10)
        plt.axis('off')
        plt.title('Model Metrics')
        
        # Histogram of residuals
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/regression_results.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, output_dir):
        """Plot feature importance"""
        results = self.analysis_results['linear_regression']
        importance = results['feature_importance']
        
        # Sort by absolute importance
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'blue' for x in importances]
        plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance (Linear Regression Coefficients)')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rolling_averages(self, output_dir):
        """Plot rolling averages"""
        rolling_cols = [col for col in self.df.columns if col.startswith('rolling_')]
        
        if rolling_cols and self.datetime_column:
            plt.figure(figsize=(12, 6))
            
            # Plot original data
            plt.plot(self.df[self.datetime_column], self.df[self.target_column], 
                    alpha=0.5, label='Original Data', color='gray')
            
            # Plot rolling averages
            colors = ['red', 'blue', 'green', 'orange']
            for i, col in enumerate(rolling_cols):
                plt.plot(self.df[self.datetime_column], self.df[col], 
                        label=col.replace('_', ' ').title(), 
                        color=colors[i % len(colors)], linewidth=2)
            
            plt.title(f'{self.target_column} with Rolling Averages')
            plt.xlabel('Time')
            plt.ylabel(self.target_column)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/rolling_averages.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_correlation_heatmap(self, output_dir):
        """Plot correlation heatmap"""
        # Select numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distributions(self, output_dir):
        """Plot distributions of key variables"""
        # Plot distributions of main features
        key_features = self.feature_columns[:min(6, len(self.feature_columns))]  # Limit to 6 features
        
        if len(key_features) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(key_features):
                if i < len(axes):
                    axes[i].hist(self.df[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{feature} Distribution')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(key_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_model(self, model_path='energy_model.pkl'):
        """Save the trained model"""
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
            print(f"Model saved to {model_path}")
    
    def generate_summary_stats(self):
        """Generate summary statistics for the analysis"""
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
            stats['target_stats'] = {
                'mean': self.df[self.target_column].mean(),
                'std': self.df[self.target_column].std(),
                'min': self.df[self.target_column].min(),
                'max': self.df[self.target_column].max(),
                'median': self.df[self.target_column].median()
            }
        
        if 'linear_regression' in self.analysis_results:
            stats['model_performance'] = {
                'mse': self.analysis_results['linear_regression']['mse'],
                'r2': self.analysis_results['linear_regression']['r2']
            }
        
        return stats

def analyze_energy_csv(csv_file_path, output_dir='analysis_output'):
    """
    Main function to analyze any energy maintenance CSV file
    
    Args:
        csv_file_path (str): Path to the CSV file
        output_dir (str): Directory to save outputs
    
    Returns:
        dict: Analysis results and statistics
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize analyzer
    analyzer = EnergyDataAnalyzer(csv_file_path)
    
    # Load and prepare data
    if not analyzer.load_and_prepare_data():
        return {'error': 'Failed to load data'}
    
    # Create rolling averages
    analyzer.create_rolling_averages()
    
    # Train linear regression model
    if not analyzer.train_linear_regression():
        return {'error': 'Failed to train model'}
    
    # Generate plots
    analyzer.generate_plots(output_dir)
    
    # Save model
    analyzer.save_model(os.path.join(output_dir, 'energy_model.pkl'))
    
    # Generate summary statistics
    stats = analyzer.generate_summary_stats()
    
    return {
        'success': True,
        'stats': stats,
        'analysis_results': analyzer.analysis_results,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        results = analyze_energy_csv(csv_file)
        print("Analysis completed!")
        print(f"Results: {results}")
    else:
        print("Usage: python energy_analysis.py <csv_file_path>") 
