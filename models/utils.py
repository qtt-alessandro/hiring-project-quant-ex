from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 


def remove_outliers_iqr(data: pd.Series, multiplier: float = 1.5):
    """Remove outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    cleaned = data[(data >= lower_bound) & (data <= upper_bound)]
    return cleaned, lower_bound, upper_bound


def create_lag_features(data: pd.Series, n_lags: int):
    """Create lagged features for time series"""
    X_list = []
    y_list = []
    for i in range(n_lags, len(data)):
        X_list.append(data.iloc[i-n_lags:i].values)
        y_list.append(data.iloc[i])
    return np.array(X_list), np.array(y_list)


def plot_residual_diagnostics(results: dict, methods: list = None, max_lags: int = 40, use_pacf: bool = False):
    """Plot residual histogram and autocorrelation for multiple forecasting methods. Ideally, a perfect model would have residuals distributed normally around zero with no autocorrelation. Any remaining autocorrelation in the residuals indicates that the selected model is not able to fully capture the data complexity. 
    
    :param results: dictionary with method names as keys and result dictionaries as values
    :param methods: list of methods to plot. If None, plots all methods in results
    :param max_lags: maximum number of lags for autocorrelation plot, defaults to 40
    :param use_pacf: if True, plot partial autocorrelation instead of autocorrelation
    """
    # Use all methods if none specified
    if methods is None:
        methods = list(results.keys())
    
    # Filter results to only include specified methods
    filtered_results = {k: v for k, v in results.items() if k in methods}
    
    n_methods = len(filtered_results)
    
    # Create subplots: 2 rows (histogram + ACF/PACF), n_methods columns
    fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 10))
    
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    corr_type = "PACF" if use_pacf else "ACF"
    plot_func = plot_pacf if use_pacf else plot_acf
    
    for idx, (method, result) in enumerate(filtered_results.items()):
        residuals = result['data']['target'] - result['data']['prediction']
        
        # Row 1: Histogram
        axes[0, idx].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, idx].set_xlabel('Residuals', fontsize=11)
        axes[0, idx].set_ylabel('Frequency', fontsize=11)
        axes[0, idx].set_title(f'{method.capitalize()}', fontsize=12)
        axes[0, idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, idx].grid(True, alpha=0.3)
        
        # Adding statistics as text in the plot
        mean_res = residuals.mean()
        std_res = residuals.std()
        axes[0, idx].text(0.02, 0.98, f'μ={mean_res:.2f}\nσ={std_res:.2f}', 
                         transform=axes[0, idx].transAxes, 
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Row 2: ACF or PACF based on the input selection defined in the signature
        plot_func(residuals.dropna(), lags=max_lags, ax=axes[1, idx], alpha=0.05)
        axes[1, idx].set_xlabel('Lag', fontsize=11)
        axes[1, idx].set_ylabel(f'{corr_type}', fontsize=11)
        axes[1, idx].set_title(f'{method.capitalize()}', fontsize=10)
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/residual_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_forecast_comparison(results: dict, methods: list = None, n_points: int = 100, 
                             time_to_delivery: str = '0 days 00:01:00'):
    """Plot forecast comparison across multiple methods with predictions, MAE, and sign accuracy.
    
    :param results: dictionary with method names as keys and result dictionaries as values
    :param methods: list of methods to plot. If None, plots all methods in results
    :param n_points: number of most recent points to plot, defaults to 100
    :param time_to_delivery: time to delivery to filter for, defaults to '0 days 00:01:00'
    """
    # Use all methods if none specified
    if methods is None:
        methods = list(results.keys())
    
    # Filter results to only include specified methods
    filtered_results = {k: v for k, v in results.items() if k in methods}
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    sign_accuracies = {}
    
    for method in filtered_results.keys():
        data = results[method]['data']
        
        closest = data[data['time_to_delivery'] == pd.Timedelta(time_to_delivery)]
        closest = closest.sort_values('delivery_start').head(n_points)
        
        # Plot 1: Predictions
        axes[0].plot(closest['delivery_start'], closest['prediction'], 
                     marker='o', label=method, alpha=0.7, markersize=3)
        
        # Plot 2: MAE
        axes[1].plot(closest['delivery_start'], closest['abs_error'], 
                     marker='o', label=method, alpha=0.7, markersize=3)
        
        # Store sign accuracy for histogram
        sign_accuracies[method] = closest['sign_correct'].mean()
    
    # Add actual to predictions plot
    axes[0].plot(closest['delivery_start'], closest['target'], 
                 'k-', linewidth=2, label='Actual', alpha=0.8)
    axes[0].set_ylabel('aFRR Activation', fontsize=11)
    axes[0].set_title(f'Predictions {time_to_delivery} Before Delivery', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].set_ylabel('Absolute Error', fontsize=11)
    axes[1].set_title('Mean Absolute Error', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Sign Accuracy Histogram
    method_names = list(sign_accuracies.keys())
    accuracies = list(sign_accuracies.values())
    
    bars = axes[2].bar(method_names, accuracies, alpha=0.7, edgecolor='black')
    axes[2].axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
                    alpha=0.7, label='Random (50%)')
    axes[2].set_xlabel('Method', fontsize=11)
    axes[2].set_ylabel('Sign Accuracy', fontsize=11)
    axes[2].set_title('Sign Accuracy Comparison by Method', fontsize=12)
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2%}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/forecast_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()




