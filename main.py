
import pandas as pd
import numpy as np
from requests_cache import CachedSession
from typer import run
from typing import Literal
from enum import StrEnum
from tqdm import tqdm
from models.models import naive_mean_forecast, naive_median_forecast, quantile_weighted_forecast, rolling_median_forecast, ridge_forecast, huber_forecast
from models.utils import plot_residual_diagnostics, plot_forecast_comparison


class BiddingZone(StrEnum):
    DK1 = "DK1"
    DK2 = "DK2"


def get_recent_afrr_activation(
    bidding_zone: BiddingZone, time_from: pd.Timestamp | None = None
) -> pd.Series:
    """Gets the last 100 minutes of published data for aFRR activations from Energinet.

    :param bidding_zone: the bidding zone
    :param time_from: first timestamp to retrieve data for. If None (default),
        the last 100 minutes are fetched
    :return: time-indexed pandas series with minute-by-minute activations.
    """
    url = "https://api.energidataservice.dk/dataset/PowerSystemRightNow"
    params = {}
    if time_from is not None:
        params["start"] = time_from.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M")

    # use cached session to avoid calling the API too often
    session = CachedSession(cache_name=".cache", expire_after=30)
    # the API returns a maximum of 100 records by default
    response = session.get(url, params=params)
    response.raise_for_status()
    res_json = response.json()

    afrr_activation = (
        pd.Series(
            {
                pd.Timestamp(r["Minutes1UTC"]): r[f"aFRR_Activated{bidding_zone}"]
                for r in res_json["records"]
            }
        )
        .astype(float)
        .dropna()
        .sort_index(ascending=False)  # most recent first
    )
    return afrr_activation


def get_forecast_horizon(now: pd.Timestamp, periods: int = 8) -> pd.DatetimeIndex:
    """Gets forecast horizon by creating a datetime index starting
    from the next settlement period and of lenght `periods`

    :param now: current time
    :param periods: number of periods, defaults to 8
    :return: datetime index for the forecast horizon
    """
    return pd.date_range(start=now.ceil("15min"), periods=periods, freq="15min")


def predict(
    forecast_horizon: pd.DatetimeIndex, recent_activations: pd.Series, method: str = "ridge"
) -> pd.Series:
    """Forecasts aFRR activation for the given forecast horizon using the specified method.

    :param forecast_horizon: datetime index with timestamps for which we want
        to make a prediction
    :param recent_activations: recent aFRR activations (minute-by-minute data)
    :param method: forecasting method to use. Available: "naive_mean", "naive_median", 
        "quantile_blend", "rolling_median", "ridge", "huber"
    :return: pandas Series with forecasted activations indexed by forecast_horizon
    """
        
    if method == "naive_mean":
        prediction = naive_mean_forecast(recent_activations, forecast_horizon)

    elif method == "naive_median":
        prediction = naive_median_forecast(recent_activations, forecast_horizon)
    
    elif method == "quantile_weighted":
        prediction = quantile_weighted_forecast(recent_activations, forecast_horizon)
    
    elif method == "rolling_median":
        prediction = rolling_median_forecast(recent_activations, forecast_horizon, window_size=20)
    
    elif method == "ridge":
        prediction = ridge_forecast(recent_activations, forecast_horizon)

    elif method == "huber":
        prediction = huber_forecast(recent_activations, 
                                    forecast_horizon,
                                    n_lags=6,       # number of past lags to use as features
                                    epsilon=1.5,    # Huber loss param
                                    alpha=0.01      # regularization param
                                    )  
        
    else:
        raise ValueError(f"Unknown forecasting method: {method}")

    return prediction


def backtest(activations: pd.DataFrame, method: str = None):
    """Runs a backtest of the naive prediction strategy.
    Computes mean absolute error and sign accuracy.

    :param activations: timeseries of historical activations
    :param method: forecasting method to use
    :return: mean absolute error and sign accuracy metrics as a function of
        forecast horizon (time to delivery)
    """
    # index here is MTU
    # we are trying to predict the average aFRR activation in each quarter
    target = activations.resample("15min").mean().sort_index()

    # ensure activations are most recent first
    activations = activations.sort_index(ascending=False)
    predictions = []
    for prediction_time in tqdm(activations.index):
        if len(activations.loc[prediction_time:]) < 100:
            continue

        # feed recent data to prediction function with the information it would
        # have had if it ran at prediction_time
        recent_activations = activations.loc[prediction_time:].iloc[:100]
        forecast_horizon = get_forecast_horizon(prediction_time + pd.Timedelta("1min"))
        prediction = predict(forecast_horizon, recent_activations, method=method)
        prediction = prediction.to_frame("prediction").reset_index(
            names="delivery_start"
        )
        prediction["prediction_time"] = prediction_time
        predictions.append(prediction)
    predictions = pd.concat(predictions)
    predictions["time_to_delivery"] = (
        predictions["delivery_start"] - predictions["prediction_time"])

    # join with target
    data = predictions.merge(
        target.to_frame("target"),
        left_on="delivery_start",
        right_index=True,
        how="inner",
    )
    data["abs_error"] = data.eval("target - prediction").abs()
    data["sign_correct"] = np.sign(data.target) == np.sign(data.prediction)

    mae = data.groupby("time_to_delivery").abs_error.mean()
    sign_accuracy = data.groupby("time_to_delivery").sign_correct.mean()
    return mae, sign_accuracy, data


def main(
    task: Literal["predict", "backtest"],
    bidding_zone: BiddingZone,
    time_from: str | None = None,
    models: str = "ridge",
    plot_results: bool = True  # Added boolean parameter
):
    """Performs prediction (on live data) or backtests model

    :param task: either "predict" or "backtest"
    :param bidding_zone: "DK1" or "DK2"
    :param time_from: the first timestamp to start the backtest from,
        in format YYYY-MM-DD hh:mm (CET timezone), defaults to None.
    :param models: forecasting method(s) to use. For backtest, use comma-separated list (e.g., "ridge,naive_mean")
    :param plot_results: whether to plot diagnostics and forecast comparison for backtest, defaults to True
    """
    
    # Parse models string into list
    models_list = [m.strip() for m in models.split(",")]
    
    match task:
        case "predict":
            now = pd.Timestamp("now", tz="CET")
            forecast_horizon = get_forecast_horizon(now)
            recent_activations = get_recent_afrr_activation(bidding_zone=bidding_zone)
            # Use first model for prediction
            prediction = predict(forecast_horizon, recent_activations, method=models_list[0])

            print(f"aFRR activation prediction for {bidding_zone}")
            print(prediction.to_string())
            
        case "backtest":
            assert time_from is not None, "time_from cannot be None for backtesting"
            
            time_from = pd.Timestamp(time_from, tz="CET")
            activations = get_recent_afrr_activation(
                bidding_zone=bidding_zone, time_from=time_from
            )
            
            results = {}
            for m in models_list:
                print(f"\n{'='*60}")
                print(f"Running backtest for method: {m.upper()}")
                print(f"{'='*60}")
                
                mae, sign_acc, data = backtest(activations, method=m)
                
                results[m] = {
                    'data': data, 
                    'mae': mae,
                    'sign_accuracy': sign_acc,
                    'activations': activations
                }

            if plot_results: 
                plot_residual_diagnostics(results, methods=models_list, use_pacf=True)
                plot_forecast_comparison(results, methods=models_list, 
                        n_points=300, 
                        time_to_delivery='0 days 00:01:00')    
            
            return results



if __name__ == "__main__":
    run(main)

