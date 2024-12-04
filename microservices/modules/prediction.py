import yfinance as yf
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
from typing import List, Union

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

class ModelPredict:
    def __init__(self,ticker):
        self.ticker: str = ticker
        self.scaler: RobustScaler = RobustScaler()
        self.close_actual: np.ndarray = np.array([])
        self.dates: List[str] = []
        self.dates_test: List[str] = []
        self.X: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])
        self.model: VotingRegressor = None
        self.combined_dates: Union[pd.DatetimeIndex, List[str]] = pd.DatetimeIndex([])
        self.combined_close: np.ndarray = np.array([])
        self.sma: List[float] = []
        self.upper: List[float] = []
        self.lower: List[float] = []
        self.status: List[str] = []
      
    
    def create_lag(self, df: pd.DataFrame, days: int)->pd.DataFrame:
        df_copy = df.copy()
         
        for feature in ['Open','High','Low']:
            df_copy[f'{feature.lower()}_lag'] = df[feature].shift(periods=days, freq='B')
            
        return df_copy
    
    
    def prepare_future_data(self, df: pd.DataFrame, days_predict: int)->tuple[pd.DataFrame, pd.DatetimeIndex]:
        last_date = pd.to_datetime(df.index.max())
        
        future_dates = pd.date_range(start=last_date+pd.Timedelta(days=1), periods=days_predict, freq='B')
        
        future_df = pd.DataFrame(index=future_dates, columns=df.columns)
        
        combined_df = pd.concat([df, future_df])
        combined_df = self.create_lag(combined_df, days=days_predict)
        
        future_feature_df = combined_df.loc[future_dates]
        features_pred = ['open_lag','high_lag','low_lag']
        
        return future_feature_df[features_pred], future_dates
        
        
    def stocks_ticker(self)->tuple[pd.DataFrame,List[str],List[float]]:
        ticker_market = yf.Ticker(self.ticker)
        
        df = ticker_market.history(period='5y')
        
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        df = df.drop(['Volume','Dividends','Stock Splits'], axis=1)
        
        self.dates = df.index.strftime('%Y-%m-%d').tolist()
        self.close_actual = df['Close'].values
        
        return df, self.dates, self.close_actual.tolist()
    
    
    def preprocessing_data(self, df: pd.DataFrame)->tuple[np.ndarray,np.ndarray]:
        df.dropna(inplace=True)
        
        features = ['open_lag','high_lag','low_lag']
        target = ['Close']
        
        X = df[features]
        self.y = df[target].values.ravel()
        
        # self.X = self.robust_normalize(X.values)
        self.X = self.scaler.fit_transform(X)
        
        return self.X, self.y
    
    
    def voting_model(self, ticker: str)->VotingRegressor:
        data = {
            'BBCA.JK': [{
                'n_estimators': 100,
                'max_depth': 10,
                'max_features': 8,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
            },{
                'n_estimators': 100,
                'eta': 0.25,
                'max_depth': 9,
                'subsample': 0.5,
            },{'weights':[1,2]}],
            'ARTO.JK': [{
                'n_estimators': 100,
                'max_depth': 10,
                'max_features': 7,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
            },{
                'n_estimators': 300,
                'eta': 0.05,
                'max_depth': 6,
                'subsample': 0.5,
            },{'weights':[1,2]}],
            'BMRI.JK': [{
                'n_estimators': 100,
                'max_depth': 10,
                'max_features': 8,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
            },{
                'n_estimators': 300,
                'eta': 0.25,
                'max_depth': 9,
                'subsample': 0.3,
            },{'weights':[1,2]}],
            'BBNI.JK': [{
                'n_estimators': 100,
                'max_depth': 10,
                'max_features': 6,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
            },{
                'n_estimators': 100,
                'eta': 0.25,
                'max_depth': 7,
                'subsample': 0.3,
            },{'weights':[1,2]}],
            'BBRI.JK': [{
                'n_estimators': 200,
                'max_depth': 20,
                'max_features': 8,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
            },{
                'n_estimators': 150,
                'eta': 0.1,
                'max_depth': 10,
                'subsample': 0.5,
            },{'weights':[1,2]}],
            'BBTN.JK': [{
                'n_estimators': 300,
                'max_depth': 20,
                'max_features': 6,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
            },{
                'n_estimators': 300,
                'eta': 0.05,
                'max_depth': 4,
                'subsample': 0.7,
            },{'weights':[2,1]}],
            'BRIS.JK': [{
                'n_estimators': 150,
                'max_depth': 10,
                'max_features': 6,
                'min_samples_leaf': 2,
                'min_samples_split': 5,
            },{
                'n_estimators': 250,
                'eta': 0.1,
                'max_depth': 3,
                'subsample': 0.3,
            },{'weights':[2,1]}]
        }
        
        rf = RandomForestRegressor(**(data[self.ticker][0]))
        xgb = XGBRegressor(**(data[self.ticker][1]))
        
        self.model = VotingRegressor(estimators=[
            ('rf', rf),
            ('xgb', xgb)
        ], **(data[ticker][2]))
        
        return self.model
    
    
    def fit_model(self)->None:
        self.model.fit(self.X, self.y)
    
    
    def predict_data(self)->tuple[np.ndarray]:
        y_pred = self.model.predict(self.X)
        
        return y_pred
    
    
    def evaluation_data(self, y_pred: np.ndarray)->tuple[float, float, float]:
        ss_res = np.sum((self.y - y_pred)**2)
        ss_tot = np.sum((self.y - np.mean(self.y))**2)
        
        rmse = np.sqrt(np.mean((self.y - y_pred)**2))
        mae = np.mean(np.abs(self.y - y_pred))
        r2 = (1 - (ss_res / ss_tot))
        
        return rmse, mae, r2
    
    
    def predict_future_value(self, df: pd.DataFrame, days: int)->tuple[List[str], List[float]]:
        future_features, future_dates = self.prepare_future_data(df, days)
        
        prediction_features = ['open_lag', 'high_lag', 'low_lag']
        future_features[prediction_features] = future_features[prediction_features].astype('float64')
        
        # future_features_scaled = self.robust_normalize(future_features[prediction_features].values)
        future_features_scaled = self.scaler.transform(future_features[prediction_features] )
        y_pred_future = self.model.predict(future_features_scaled)
        
        return future_dates.tolist(), y_pred_future.tolist()
    
    
    def combine_actual_predict(self, df: pd.DataFrame, days: int)->tuple[List[str], List[float], List[str], List[float]]:
        future_dates, close_pred = self.predict_future_value(df, days)
        
        future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        self.combined_close = np.append(self.close_actual, close_pred)
        self.combined_dates = self.dates + future_dates_str
        
        return future_dates_str, close_pred, self.combined_dates, self.combined_close.tolist()
    
    
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
            (data > self.sma[-len(data):]) & (data < self.upper[-len(data):]),
            (data < self.sma[-len(data):]) & (data > self.lower[-len(data):]),
        ]
        
        choices = ['Naik Signifikan', 'Turun Signifikan', 'Naik', 'Turun']
        
        self.status = np.select(condition, choices, default='Stabil')
    
        return self.status.tolist()
        