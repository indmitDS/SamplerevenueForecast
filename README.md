# 📈 Revenue Forecasting Pipeline

## Overview

This project builds a modular pipeline to forecast revenue using multiple time series models including **ARIMA, SARIMA, Prophet, and LSTM**.

It covers the full workflow:

* Data preprocessing
* Feature analysis
* Model training & evaluation
* Forecast generation
* Visualization of results

The design is **modular and extensible**, making it easy to plug in additional models or datasets.

---

## 📁 Project Structure

```plaintext
revenueForecast/
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── revenueForecast/
    ├── main.py
    ├── config/
    │   ├── config.yaml
    │   ├── logging.conf
    ├── data/
    ├── logs/
    ├── models/
    ├── plots/
    ├── reports/
    └── src/
        ├── arima_model.py
        ├── sarima_model.py
        ├── prophet_model.py
        ├── lstm_model.py
        ├── data_preprocessing.py
        ├── feature_engineering.py
        ├── feature_analysis.py
        ├── evaluation.py
        └── eda.py
```

---

## 📊 Data

* The project uses **revenue and margin data** aggregated at the customer level
* Final dataset is transformed into a **univariate time series (Revenue vs Time)**

### Key preprocessing steps:

* Data cleaning & filtering
* Aggregation by customer and time (YYYYWW)
* Handling missing periods via **time interpolation**
* Feature ranking and statistical testing (ANOVA)

---

## 🤖 Models Implemented

### ARIMA

Baseline statistical time series model.

### SARIMA

Seasonal extension of ARIMA — **best performing model in this project**.

### Prophet

Captures trend + seasonality with smoother forecasts.

### LSTM

Deep learning model for capturing long-term dependencies in time series.

---

## 📈 Results

* **SARIMA (1,0,1)(1,0,1,52)** performed best based on error metrics
* Prophet captured smoother long-term trends but missed short-term volatility
* LSTM provided reasonable fits but required more tuning

### Insights:

* Revenue data exhibits **seasonality (weekly cycle ~52)**
* Minimal differencing required → data is close to stationary
* Forecast uncertainty increases significantly over longer horizons

### Sample Outputs

![SARIMA Forecast](revenueForecast/revenueForecast/plots/SARIMA_forecast.png)
![Prophet Forecast](revenueForecast/revenueForecast/plots/Prophet_forecast.png)


---

## 🚀 Setup & Run Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd revenueForecast
```

---

### 2. Create virtual environment (Python 3.11 recommended)

```bash
python -m venv .venv
```

---

### 3. Activate environment

**Windows (Command Prompt):**

```bash
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

If blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Data Setup

Place your input dataset here:

```text
revenueForecast/data/finance_sample_data.xlsx
```

> If not included in the repo, manually copy it into the `data/` folder.

---

### 6. Run the pipeline

```bash
cd revenueForecast
python main.py
```

---

## ⚙️ Configuration

Edit:

```text
revenueForecast/config/config.yaml
```

Ensure all paths are **relative**, for example:

```yaml
data_path: data/finance_sample_data.xlsx
output_dir: reports/
model_dir: models/
```

Avoid absolute paths like:

```text
C:/Users/yourname/...
```

---

## 🧠 Key Features

* Modular pipeline architecture
* Multiple model comparison framework
* Logging support via `logging.conf`
* Config-driven execution
* Extendable for additional models and datasets

---

## ⚠️ Notes

* Use **Python 3.11** for best compatibility (TensorFlow / Prophet)
* Logs are generated in `/logs`
* Models and outputs are stored in `/models`, `/plots`, `/reports`
* Ensure correct working directory when running the script

---

## 🔧 Troubleshooting

### Module not found

```bash
pip install -r requirements.txt
```

### File path errors

Make sure you run from:

```bash
cd revenueForecast
python main.py
```

### Missing data file

Check:

```text
revenueForecast/data/
```

---

## 📌 Future Improvements

* Hyperparameter tuning for Prophet & LSTM
* Multivariate forecasting
* Automated model selection
* Deployment as API or dashboard

---

## 📜 License

This project is for educational and demonstration purposes.
