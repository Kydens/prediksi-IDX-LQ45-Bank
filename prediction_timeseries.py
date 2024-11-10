import yfinance as yf
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from microservices.modules.custom_model import XGBRegressor

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


class model_predict:
    def __init__(self, ticker):
        self.ticker = ticker
        self.mms = RobustScaler()
        self.sma = []
        self.std = []
        self.upper = []
        self.lower = []


    def stocks_ticker(self):
        ticker_market = yf.Ticker(self.ticker)
        df = ticker_market.history(period='5y')
        
        close = df['Close'].values.reshape(-1,1).flatten()
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        dates = df.index
        
        return df, dates, close


    def preprocessing_data(self):
        df, dates , _ = self.stocks_ticker()
        df = df.drop(['Dividends','Stock Splits'], axis=1)

        features = df[['Open','High','Low','Volume']]
        target = df[['Close']]

        features_norm = self.mms.fit_transform(features)
        target_norm = self.mms.fit_transform(target)
        
        return dates, features_norm, target_norm


    def train_test_data(self):
        dates, features_norm, target_norm = self.preprocessing_data()
        split_data = int(len(features_norm)*0.9)
        
        dates_test = dates[split_data:]
        
        X_train, X_test = features_norm[:split_data], features_norm[split_data:]
        y_train, y_test = target_norm[:split_data], target_norm[split_data:]
        
        return dates_test, X_train, X_test, y_train, y_test

    def voting_model(self):
        rf = RandomForestRegressor(n_estimators=100,
                                max_depth=8,
                                max_features=4,
                                min_samples_leaf=2,
                                min_samples_split=2)
        xgb = XGBRegressor(n_estimators=100,
                        eta=0.15,
                        max_depth=8,
                        subsample=0.3)
        
        return VotingRegressor(estimators=[
            ('rf', rf),
            ('xgb', xgb),
        ], weights=[2,1])


    def predict_data(self):
        dates_test, X_train, X_test, y_train, y_test = self.train_test_data()
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        
        model = self.voting_model()
        
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        y_test_transform = self.mms.inverse_transform(y_test.reshape(-1,1))
        y_pred_transform = self.mms.inverse_transform(y_pred.reshape(-1,1))
        
        return dates_test, y_test_transform.flatten(), y_pred_transform.flatten()


    def evaluation_data(self):
        _, y_test, y_pred = self.predict_data()
        
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return rmse, mae, r2
    
    
    def predict_future_value(self,days):
        dates_test, X_train, X_test, y_train, y_test = self.train_test_data()
        
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        
        model = self.voting_model()
        model.fit(X_train,y_train)
        
        last_date = dates_test[-1]
        dates_pred = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
        
        last_features = X_test[-1:]
        future_features = np.tile(last_features, (days, 1))
        
        y_pred = model.predict(future_features)

        y_pred_transform = self.mms.inverse_transform(y_pred.reshape(-1,1))
        
        return dates_pred, y_pred_transform.flatten()


    def combine_actual_predict(self,days):
        _, dates_actual, close_actual = self.stocks_ticker()
        dates_pred, close_pred = self.predict_future_value(days)
        
        combined_dates = pd.DatetimeIndex.append(dates_actual,dates_pred)
        combined_df = np.append(close_actual, close_pred)
        
        
        return dates_pred, close_pred, combined_dates, combined_df
    
    