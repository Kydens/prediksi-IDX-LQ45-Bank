import yfinance as yf
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

class ModelPredict:
    def __init__(self,ticker):
        self.ticker = ticker
        self.scaler = RobustScaler()
        self.close_actual = None,
        self.dates, self.dates_test = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.combined_dates, self.combined_close = None, None
        self.sma, self.upper, self.lower = [], [], []
        
        
    def stocks_ticker(self):
        ticker_market = yf.Ticker(self.ticker)
        
        df = ticker_market.history(period='5y')
        
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        self.dates = df.index
        
        self.close_actual = df['Close'].values.reshape(-1,1).flatten()
        
        return df, self.dates, self.close_actual
    
    
    def preprocessing_data(self, df):
        df = df.drop(['Dividends','Stock Splits'], axis=1)
        
        features = df[['Open','High','Low','Volume']]
        target = df[['Close']]
        
        features = self.scaler.fit_transform(features)
        target = self.scaler.fit_transform(target)
        
        return features, target
    
    
    def train_test_data(self, features_norm, target_norm):
        split_data = int(len(features_norm)*0.9)
        
        self.dates_test = self.dates[split_data:]
        
        self.X_train, self.X_test = features_norm[:split_data], features_norm[split_data:]
        self.y_train, self.y_test = target_norm[:split_data].ravel(), target_norm[split_data:].ravel()
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    
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
        
        self.model = VotingRegressor(estimators=[
            ('rf', rf),
            ('xgb', xgb)
        ], weights=[2,1])
        
        return self.model
    
    
    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)
    
    
    def predict_data(self):
        y_pred = self.model.predict(self.X_test)

        y_test_reversed = self.scaler.inverse_transform(self.y_test.reshape(-1,1))
        y_pred_reversed = self.scaler.inverse_transform(y_pred.reshape(-1,1))
        
        return y_test_reversed.flatten(), y_pred_reversed.flatten()
    
    
    def evaluation_data(self, y_test, y_pred):
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test,y_pred)
        
        return rmse, mae, r2
    
    
    def predict_future_value(self, days):
        last_features = self.X_test[-days:]
        
        y_pred = self.model.predict(last_features)
        
        y_pred_reversed = self.scaler.inverse_transform(y_pred.reshape(-1,1))
        
        last_date = self.dates_test[-1]
        dates_future = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
        
        return dates_future, y_pred_reversed.flatten()
    
    
    def combine_actual_predict(self, days):
        dates_pred, close_pred = self.predict_future_value(days)
        
        self.combined_dates = pd.DatetimeIndex.append(self.dates, dates_pred)
        self.combined_close = np.append(self.close_actual, close_pred)
        
        return dates_pred, close_pred, self.combined_dates, self.combined_close
    
    
    def bollinger_bands(self, data, size):
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
    
    
    print(close)
    
    print(rmse)
    print(mae)
    print(r2)
    
    print(dates_future)
    print(close_future)
    
    print(combined_dates, combined_close)
    
    print(len(combined_close))
    print(len(sma))
    print(len(upper))
    print(len(lower))
        