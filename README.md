# aFRR Activation Forecasting

Forecasting system for Danish aFRR (automatic Frequency Restoration Reserve) activations. Predicts the next 8 settlement periods (2 hours) with 15-minute granularity.

---

## Project Structure

```
afrr-forecasting/
├── main.py                 
├── models/
│   ├── models.py           # 6 forecasting (simple) models
│   └── utils.py            # plotting functions
├── plots/                  # forecasts and residuals diagnostics
├── requirements.txt
└── README.md
```

---

### Usage

**Live Prediction:**
```bash
python main.py predict DK1 --models ridge
```

**Backtest:**
```bash
python main.py backtest DK1 --time-from "2026-01-15 00:00" --models ridge
```

**Compare Methods:**
```bash
python main.py backtest DK1 --time-from "2026-01-15 00:00" \
  --models "naive_mean,ridge,huber"
```

---

## 🔬 Available Methods

| Method | Description |
|--------|-------------|
| `naive_mean` | Simple average |
| `naive_median` | Robust median |
| `quantile_weighted` | Weighted quantile blend |
| `rolling_median` | Adaptive rolling window |
| `ridge` | Ridge regression |
| `huber` | Robust regression |

---

## Output

- **Prediction**: 8 periods (15-min each) of forecasted activations
- **Backtest**: MAE + sign accuracy by forecast horizon
- **Plots**: Residual diagnostics + forecast comparison (saved to `plots/`)

---

## Key Features

✅ Multiple forecasting methods  
✅ Walk-forward backtesting  
✅ Residuals Diagnostic plots (ACF/PACF, residuals)  

---
