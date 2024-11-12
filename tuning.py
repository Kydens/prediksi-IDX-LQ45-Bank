import yfinance as yf
import pandas as pd
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestRegressor
from microservices.modules.custom_model import XGBRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error, r2_score

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


# def train_test_data():
#     features_norm, target_norm = preprocessing_data()
#     split_data = int(len(features_norm)*0.9)
    
#     X_train, X_test = features_norm[:split_data], features_norm[split_data:]
#     y_train, y_test = target_norm[:split_data], target_norm[split_data:]
    
#     return X_train, X_test, y_train, y_test

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
    
    grid_rf = GridSearchCV(rf, rf_params, scoring='neg_root_mean_squared_error', cv=cv)
    grid_xgb = GridSearchCV(xgb, xgb_params, scoring='neg_root_mean_squared_error', cv=cv)

    return grid_rf, grid_xgb
    # return grid_xgb


def predict_data(X, y, cv):
    # X_train, X_test, y_train, y_test = train_test_data()
    # y_train = y_train.ravel()
    # y_test = y_test.ravel()
    
    rf, xgb = tuning_model(cv)
    # rf = tuning_model(cv)
    
    # rf.fit(X_train,y_train)
    # xgb.fit(X_train,y_train)
    
    rf.fit(X,y)
    xgb.fit(X,y)
    
    # y_pred_rf = rf.predict(X_test)
    # y_pred_xgb = xgb.predict(X_test)
    
    y_pred_rf = rf.predict(X)
    y_pred_xgb = xgb.predict(X)
    
    print(f'Best Params for RF : {rf.best_params_}')
    print(f'Best Params for XGB : {xgb.best_params_}')
    return y_pred_rf, y_pred_xgb
    # return y_pred_rf

    
cv = [5,10]

for i in range(len(cv)):
    ticker = 'BRIS.JK'
    print(ticker)
    print(f'Experiment on going using CV {cv[i]}:')  
    df = stocks_ticker(ticker)
    df_lags = create_lag(df, days=30)
    X, y = preprocessing_data(df_lags)
    # train_test_data()
    print(F'Model going to be tune with GridSearch (CV {cv[i]})')
    tuning_model(cv[i])
    print(f'Result with {cv[i]} is :')
    predict_data(X, y, cv[i])

print(f'Experiment Done.\n\n\n') 

ticker = 'BBTN.JK'
print(ticker)
print(f'Experiment on going using CV {cv[0]}:')  
df = stocks_ticker(ticker)
df_lags = create_lag(df, days=30)
X, y = preprocessing_data(df_lags)
# train_test_data()
print(F'Model going to be tune with GridSearch (CV {cv[0]})')
tuning_model(cv[0])
print(f'Result with {cv[0]} is :')
predict_data(X, y, cv[0])   

print('Experiment Done.') 

