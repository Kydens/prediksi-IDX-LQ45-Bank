import yfinance as yf
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
from typing import List, Union
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

class ModelPredict:
    def __init__(self,ticker):
        self.ticker: str = ticker
        self.scaler: RobustScaler = RobustScaler()
        self.close_actual: np.ndarray = np.array([])
        self.dates: List[str] = []
        self.dates_test: List[str] = []
        self.X_train: np.ndarray = np.array([])
        self.X_test: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.y_test: np.ndarray = np.array([])
        self.model: Union[VotingRegressor, None] = None
        self.combined_dates: Union[pd.DatetimeIndex, List[str]] = pd.DatetimeIndex([])
        self.combined_close: np.ndarray = np.array([])
        self.sma: List[float] = []
        self.upper: List[float] = []
        self.lower: List[float] = []
        self.status: List[str] = []
        
        
    def stocks_ticker(self)->tuple[pd.DataFrame,List[str],List[float]]:
        ticker_market = yf.Ticker(self.ticker)
        
        df = ticker_market.history(period='5y')
        
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        self.dates = df.index.strftime('%Y-%m-%d').tolist()
        
        self.close_actual = df['Close'].values
        
        return df, self.dates, self.close_actual.tolist()
    
    
    def preprocessing_data(self, df: pd.DataFrame)->tuple[np.ndarray,np.ndarray]:
        df = df.drop(['Dividends','Stock Splits'], axis=1)
        
        features = df[['Open','High','Low','Volume']]
        target = df[['Close']]
        
        features = self.scaler.fit_transform(features)
        target = self.scaler.fit_transform(target)
        
        return features, target
    
    
    def train_test_data(self, features_norm: np.ndarray, target_norm: np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        split_data = int(len(features_norm)*0.9)
        
        self.dates_test = self.dates[split_data:]
        
        self.X_train, self.X_test = features_norm[:split_data], features_norm[split_data:]
        self.y_train, self.y_test = target_norm[:split_data].ravel(), target_norm[split_data:].ravel()
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    
    def voting_model(self)->VotingRegressor:
        rf = RandomForestRegressor(n_estimators=100,
                                max_depth=8,
                                max_features=4,
                                min_samples_leaf=2,
                                min_samples_split=2)
        xgb = XGBRegressor(n_estimators=100,
                        eta=0.15,
                        max_depth=8,
                        subsample=0.3)
        
        self.model = VotingRegressor(estimators=[
            ('rf', rf),
            ('xgb', xgb)
        ], weights=[2,1])
        
        return self.model
    
    
    def fit_model(self)->None:
        self.model.fit(self.X_train, self.y_train)
    
    
    def predict_data(self)->tuple[np.ndarray,np.ndarray]:
        y_pred = self.model.predict(self.X_test)

        y_test_reversed = self.scaler.inverse_transform(self.y_test.reshape(-1,1))
        y_pred_reversed = self.scaler.inverse_transform(y_pred.reshape(-1,1))
        
        return y_test_reversed.flatten(), y_pred_reversed.flatten()
    
    
    def evaluation_data(self, y_test: np.ndarray, y_pred: np.ndarray)->tuple[float, float, float]:
        rmse = root_mean_squared_error(y_test,y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test,y_pred)
        
        return rmse, mae, r2
    
    
    def predict_future_value(self, data:np.ndarray, days: int)->tuple[pd.DatetimeIndex,np.ndarray]:
        last_features = data[-days:]
        
        y_pred = self.model.predict(last_features)
        
        y_pred_reversed = self.scaler.inverse_transform(y_pred.reshape(-1,1))
        
        last_date = datetime.strptime(self.dates_test[-1], '%Y-%m-%d')
        dates_future = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
        
        return dates_future, y_pred_reversed.flatten()
    
    
    def combine_actual_predict(self, data:np.ndarray, days: int)->tuple[List[str], List[float], List[str], List[float]]:
        dates_pred, close_pred = self.predict_future_value(data, days)
        
        dates_pred_list = dates_pred.strftime('%Y-%m-%d')
        
        self.combined_dates = pd.DatetimeIndex(self.dates + dates_pred.strftime('%Y-%m-%d').tolist()).strftime('%Y-%m-%d')
        self.combined_close = np.append(self.close_actual, close_pred)
        
        return dates_pred_list.tolist(), close_pred.tolist(), self.combined_dates.tolist(), self.combined_close.tolist()
    
    
    def bollinger_bands(self, data: List[float], size:int)->tuple[List[float],List[float],List[float]]:
        window = pd.Series(data).rolling(size)
        std = window.std()
        sma = window.mean()
        
        upper = sma + (2*std)
        lower = sma - (2*std)
        
        sma_list = sma.tolist()
        upper_list = upper.tolist()
        lower_list = lower.tolist()
        
        self.sma = sma_list[size - 1:]
        self.upper = upper_list[size - 1:]
        self.lower = lower_list[size - 1:]
        
        return self.sma, self.upper, self.lower
    
    
    def status_predict(self, y_pred: List[float])->List[str]:
        data = np.array(y_pred)
        
        condition = [
            (data > self.upper[-len(data):]),
            (data < self.lower[-len(data):]),
            (data >= self.sma[-len(data):]) & (data <= self.upper[-len(data):]),
            (data <= self.sma[-len(data):]) & (data >= self.lower[-len(data):]),
        ]
        
        choices = ['Naik Signifikan', 'Turun Signifikan', 'Naik', 'Turun']
        
        self.status = np.select(condition, choices, default='Stabil')
    
        return self.status.tolist()
    
    
if __name__ == '__main__':
    ticker = 'BBCA.JK'
    days = 30
    window = 20
    
    predict = ModelPredict(ticker)
    
    df, _, close = predict.stocks_ticker()
    
    features, target = predict.preprocessing_data(df)
    
    predict.train_test_data(features, target)

    predict.voting_model()

    predict.fit_model()

    y_test_reversed, y_pred_reversed = predict.predict_data()
    
    rmse, mae, r2 = predict.evaluation_data(y_test_reversed,y_pred_reversed)
    
    dates_future, close_future, combined_dates, combined_close = predict.combine_actual_predict(days)
    
    sma, upper, lower = predict.bollinger_bands(combined_close, window)
    
    status = predict.status_predict(close_future)
    
    # print(close)
    
    # print(rmse)
    # print(mae)
    # print(r2)
    
    # print(dates_future)
    # print(close_future)
    
    # print(combined_dates, combined_close)
    
    # print(len(combined_close))
    # print(sma)
    # print(upper)
    # print(lower)
    
    # print(status)
        