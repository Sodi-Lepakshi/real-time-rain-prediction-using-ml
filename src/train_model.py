import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import os
import logging
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Read the data
    logger.info("Loading historical_rainfall.csv")
    data_path = os.path.join("..", "data", "historical_rainfall.csv")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}. Run generate_data.py first.")
        raise FileNotFoundError(f"Data file not found at {data_path}. Run generate_data.py first.")
    data = pd.read_csv(data_path)
    logger.info(f"Loaded data with {len(data)} rows")

    # Add temporal features
    logger.info("Adding temporal features")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['Sin_Month'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Cos_Month'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Sin_Day'] = np.sin(2 * np.pi * data['DayOfYear'] / 365)
    data['Cos_Day'] = np.cos(2 * np.pi * data['DayOfYear'] / 365)

    # Add previous rainfall features
    logger.info("Adding rainfall features")
    data['Rainfall_prev'] = data.groupby('Region')['Rainfall'].shift(1).fillna(0)
    data['Rainfall_prev_2'] = data.groupby('Region')['Rainfall'].shift(2).fillna(0)
    data['Rainfall_Lag3'] = data.groupby('Region')['Rainfall'].shift(3).fillna(0)
    data['Rainfall_Rolling7'] = data.groupby('Region')['Rainfall'].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # Add additional features
    logger.info("Adding additional features")
    data['Humidity_prev'] = data.groupby('Region')['Humidity'].shift(1).fillna(0)
    data['Humidity_Lag2'] = data.groupby('Region')['Humidity'].shift(2).fillna(0)
    data['Humidity_Rolling7'] = data.groupby('Region')['Humidity'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    data['Temp_Rolling7'] = data.groupby('Region')['Temperature'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    data['Temp_Humidity'] = data['Temperature'] * data['Humidity']
    data['Pressure_Rolling3'] = data.groupby('Region')['Pressure'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    data['Temp_Pressure'] = data['Temperature'] * data['Pressure']

    # Features and target
    features = ["Region", "Temperature", "Humidity", "Humidity_prev", "Humidity_Lag2", "Humidity_Rolling7", 
                "Temp_Rolling7", "Temp_Humidity", "Pressure", "Pressure_Rolling3", "Temp_Pressure", 
                "Rainfall_prev", "Rainfall_prev_2", "Rainfall_Lag3", "Rainfall_Rolling7", 
                "Month", "DayOfYear", "Sin_Month", "Cos_Month", "Sin_Day", "Cos_Day"]
    X = data[features]
    X = pd.get_dummies(X, columns=["Region"])
    y = data["Rainfall"]
    logger.info(f"Prepared {len(features)} features")

    # Train-test split
    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model parameters
    logger.info("Initializing XGBoost model")
    params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'rmse'
    }

    # Train model with manual early stopping
    logger.info("Training model")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evals = [(dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # Wrap the trained model in XGBRegressor for compatibility
    model_regressor = XGBRegressor()
    model_regressor._Booster = model

    # Save the model
    logger.info("Saving model")
    model_dir = os.path.join("..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rainfall_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_regressor, f)

    # Print model accuracy
    r2_score = model_regressor.score(X_test, y_test)
    logger.info(f"Model trained! Accuracy (R^2): {r2_score:.2f}")

except Exception as e:
    logger.error(f"Error in train_model.py: {str(e)}")
    raise