import pandas as pd 
import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .utils import remove_outliers_iqr, create_lag_features

def naive_mean_forecast(
    recent_activations: pd.Series,
    forecast_horizon: pd.DatetimeIndex,
) -> pd.Series:
    """Naive mean model. Returns a constant prediction over the entire forecast
    horizon equal to the mean of the most recent activations.

    :param forecast_horizon: datetime index with timestamps for which
        predictions are generated
    :param recent_activations: recent aFRR activations as a time-indexed series
    :return: time-indexed pandas Series with predictions for each timestamp
    """
    return pd.Series(recent_activations.mean(), index=forecast_horizon)


def naive_median_forecast(
    recent_activations: pd.Series,
    forecast_horizon: pd.DatetimeIndex,
) -> pd.Series:
    """Naive median model. Returns a constant prediction over the entire forecast
    horizon equal to the median of the most recent activations.

    :param forecast_horizon: datetime index with timestamps for which
        predictions are generated
    :param recent_activations: recent aFRR activations as a time-indexed series
    :return: time-indexed pandas Series with predictions for each timestamp
    """
    return pd.Series(recent_activations.median(), index=forecast_horizon)

def quantile_weighted_forecast(
    recent_activations: pd.Series,
    forecast_horizon: pd.DatetimeIndex,
) -> pd.Series:
    """Quantile weighted model. Returns a constant prediction over the entire forecast
    horizon equal to a weighted sum of the 25th, 50th, and 75th percentiles of the most recent activations.

    Weights:
    - 25th percentile: 0.25
    - 50th percentile: 0.5
    - 75th percentile: 0.25

    :param forecast_horizon: datetime index with timestamps for which
        predictions are generated
    :param recent_activations: recent aFRR activations as a time-indexed series
    :return: time-indexed pandas Series with predictions for each timestamp
    """

    q25 = recent_activations.quantile(0.25)
    q50 = recent_activations.quantile(0.50)  
    q75 = recent_activations.quantile(0.75)
        
    prediction_value = 0.25 * q25 + 0.5 * q50 + 0.25 * q75
    return pd.Series(prediction_value, index=forecast_horizon)

def rolling_median_forecast(
    recent_activations: pd.Series,
    forecast_horizon: pd.DatetimeIndex,
    window_size: int = 5
) -> pd.Series:
    """Rolling median model. Returns a constant prediction over the entire forecast
    horizon equal to the median of the rolling median of the most recent activations.

    :param forecast_horizon: datetime index (front-stamped) with timestamps for which
        predictions are generated
    :param recent_activations: recent aFRR activations as a time-indexed series
    :param window_size: size of the rolling window for median calculation
    :return: time-indexed pandas Series with predictions for each timestamp
    """
    rolling_med = recent_activations.rolling(window=window_size, min_periods=1).median()
    return pd.Series(rolling_med.iloc[-1], index=forecast_horizon)


def ridge_forecast(
    recent_activations: pd.Series,
    forecast_horizon: pd.DatetimeIndex,
    n_lags: int = 10,
    alpha: float = 10.0,
    outlier_multiplier: float = 1.5
) -> pd.Series:
    """Ridge regression model with lagged features for time series forecasting.
    
    :param recent_activations: recent aFRR activations as a time-indexed series
    :param forecast_horizon: datetime index (front-stamped) with timestamps for which
        predictions are generated
    :param n_lags: number of lag features to use, defaults to 10
    :param alpha: regularization strength for Ridge regression, defaults to 10.0
    :param outlier_multiplier: IQR multiplier for outlier removal, defaults to 1.5
    :return: time-indexed pandas Series with predictions for each timestamp
    """
    
    # Remove outliers
    cleaned, lower_bound, upper_bound = remove_outliers_iqr(
        recent_activations, multiplier=outlier_multiplier
    )
    
    # Create lagged features
    X, y = create_lag_features(cleaned, n_lags)
    
    # Scaling the features for numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    
    # Forecast
    preds = []
    last_values = cleaned.iloc[-n_lags:].values
    
    # Iteratively predict each step in the forecast horizon
    for _ in range(len(forecast_horizon)):
        next_X_scaled = scaler.transform(last_values.reshape(1, -1))
        pred = model.predict(next_X_scaled)[0]
        pred = np.clip(pred, lower_bound, upper_bound)
        preds.append(pred)
        last_values = np.append(last_values[1:], pred)
    
    return pd.Series(preds, index=forecast_horizon)


def huber_forecast(
    recent_activations: pd.Series,
    forecast_horizon: pd.DatetimeIndex,
    n_lags: int = 4,
    epsilon: float = 1.35,
    alpha: float = 0.001,
    max_iter: int = 1000,
    outlier_multiplier: float = 1.5
) -> pd.Series:
    """Huber regression model with lagged features for time series forecasting.
    Robust to outliers compared to standard linear regression.
    
    :param recent_activations: recent aFRR activations as a time-indexed series
    :param forecast_horizon: datetime index (front-stamped) with timestamps for which
        predictions are generated
    :param n_lags: number of lag features to use, defaults to 4
    :param epsilon: parameter that controls the threshold for Huber loss, defaults to 1.35
    :param alpha: regularization strength, defaults to 0.001
    :param max_iter: maximum number of iterations, defaults to 1000
    :param outlier_multiplier: IQR multiplier for outlier removal, defaults to 1.5
    :return: time-indexed pandas Series with predictions for each timestamp
    """
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Remove outliers
    cleaned, lower_bound, upper_bound = remove_outliers_iqr(
        recent_activations, multiplier=outlier_multiplier
    )
    
    # Create features
    X, y = create_lag_features(cleaned, n_lags)
    
    # Scale and fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=max_iter)
    model.fit(X_scaled, y)
    
    # Forecast
    preds = []
    last_values = cleaned.iloc[-n_lags:].values
    
    for _ in range(len(forecast_horizon)):
        next_X_scaled = scaler.transform(last_values.reshape(1, -1))
        pred = model.predict(next_X_scaled)[0]
        pred = np.clip(pred, lower_bound, upper_bound)
        preds.append(pred)
        last_values = np.append(last_values[1:], pred)
    
    return pd.Series(preds, index=forecast_horizon)


