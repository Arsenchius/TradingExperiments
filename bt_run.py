import pandas as pd


def backtest_run():
    pass


import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
import yaml

# Load data
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

# Load best params from YAML file
with open('best_params.yaml', 'r') as f:
    best_params = yaml.load(f, Loader=yaml.FullLoader)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Train model with best hyperparameters
model = lgb.train(best_params, train_data, num_boost_round=1000, valid_sets=[train_data, test_data], early_stopping_rounds=100, verbose_eval=False)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
