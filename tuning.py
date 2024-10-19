import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error, r2_score

mms = MinMaxScaler()

def stocks_ticker():
    bbca = yf.Ticker('BBCA.JK')
    bbca_df = bbca.history(period='5y', interval='1d')
    
    return bbca_df


def preprocessing_data():
    bbca_df = stocks_ticker()
    bbca_df = bbca_df.drop(['Dividends','Stock Splits'], axis=1)

    features = bbca_df[['Open','High','Low','Volume']]
    target = bbca_df[['Close']]

    features_norm = mms.fit_transform(features)
    target_norm = mms.fit_transform(target)
    return features_norm, target_norm


def train_test_data():
    features_norm, target_norm = preprocessing_data()
    split_data = int(len(features_norm)*0.8)
    
    X_train, X_test = features_norm[:split_data], features_norm[split_data:]
    y_train, y_test = target_norm[:split_data], target_norm[split_data:]
    
    return X_train, X_test, y_train, y_test

def tuning_model():
    rf = RandomForestRegressor()
    xgb = XGBRegressor()
    
    rf_params = {
        'n_estimators': [50,100,150,200,250,300],
        'max_depth': [4,6,8,10,12,15,20],
        'min_samples_split': [2,5,10,15,20],
        'min_samples_leaf': [2,5,10,15,20],
        'max_features': [4,5,6,7,8]
    }
    
    xgb_params = {
        'learning_rate': [0.01,0.05,0.1,0.15,0.2,0.25,0.3],
        'max_depth': [3,4,5,6,7,8,9,10,15],
        'subsample': [0.3,0.5,0.7,1.0],
        'n_estimators': [50,100,150,200,300],
    }
    
    grid_rf = GridSearchCV(rf, rf_params, scoring='neg_root_mean_squared_error', cv=10)
    grid_xgb = GridSearchCV(xgb, xgb_params, scoring='neg_root_mean_squared_error', cv=10)

    return grid_rf, grid_xgb


def predict_data():
    X_train, X_test, y_train, y_test = train_test_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    rf, xgb = tuning_model()
    
    rf.fit(X_train,y_train)
    xgb.fit(X_train,y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_pred_xgb = xgb.predict(X_test)
    
    print(rf.best_params_)
    print(xgb.best_params_)
    return y_test, y_pred_rf, y_pred_xgb

    

stocks_ticker()
preprocessing_data()
train_test_data()
print(tuning_model())
print(predict_data())