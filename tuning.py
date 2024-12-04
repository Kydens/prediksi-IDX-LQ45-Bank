import yfinance as yf
import pandas as pd
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from alive_progress import alive_bar

rs = RobustScaler()

def create_lag(df, days):
    df_copy = df.copy()
        
    for feature in ['Open','High','Low','Volume']:
        df_copy[f'{feature.lower()}_lag'] = df[feature].shift(periods=days, freq='B')
    
    return df_copy

def stocks_ticker(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='5y', interval='1d')
    
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    df = df.drop(['Dividends','Stock Splits'], axis=1)
    
    return df


def preprocessing_data(df):
    df.dropna(inplace=True)

    features = df.drop(columns=['Close'], axis=1)
    target = df['Close']

    X_scaled = rs.fit_transform(features)
    y = target.values.reshape(-1,1).ravel()
    
    return X_scaled, y


def tuning_model(cv):
    rf = RandomForestRegressor()
    xgb = XGBRegressor()
    
    rf_params = {
        'n_estimators': [100,150,200,250,300],
        'max_depth': [4,6,8,10,12,15,20],
        'min_samples_split': [2,5,10,15,20],
        'min_samples_leaf': [2,5,10,15,20],
        'max_features': [4,5,6,7,8]
    }
    
    xgb_params = {
        'learning_rate': [0.01,0.05,0.1,0.15,0.2,0.25],
        'max_depth': [3,4,5,6,7,8,9,10],
        'subsample': [0.3,0.5,0.7,1.0],
        'n_estimators': [100,150,200,250,300],
    }
    
    grid_rf = GridSearchCV(rf, rf_params, scoring='neg_root_mean_squared_error', cv=TimeSeriesSplit(n_splits=cv))
    grid_xgb = GridSearchCV(xgb, xgb_params, scoring='neg_root_mean_squared_error', cv=TimeSeriesSplit(n_splits=cv))

    return grid_rf, grid_xgb


def predict_data(X, y, cv):
    rf, xgb = tuning_model(cv)
    
    rf.fit(X,y)
    xgb.fit(X,y)
    
    y_pred_rf = rf.predict(X)
    y_pred_xgb = xgb.predict(X)
    
    print(f'Best Params for RF : {rf.best_params_}')
    print(f'Best Params for XGB : {xgb.best_params_}')
    
    return y_pred_rf, y_pred_xgb

    
cv = [5,10]

tickers = ['ARTO.JK', 'BBCA.JK', 'BBNI.JK', 'BBRI.JK', 'BBTN.JK', 'BMRI.JK', 'BRIS.JK']

with alive_bar(len(cv) * len(tickers), title="Processing Stocks") as bar:
    for i in range(len(cv)):
        for ticker in tickers:
            print(f'\nProcessing Ticker: {ticker} | CV: {cv[i]}')
            df = stocks_ticker(ticker)
            df_lags = create_lag(df, days=7)
            X, y = preprocessing_data(df_lags)
            print(f'Tuning Model with GridSearch (CV={cv[i]})')
            tuning_model(cv[i])
            print(f'Predicting for Ticker: {ticker} | CV: {cv[i]}')
            predict_data(X, y, cv[i])
            bar()

print('Experiment Done.')