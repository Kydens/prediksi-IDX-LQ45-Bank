import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from microservices.modules.custom_model import XGBRegressor

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

mms = MinMaxScaler()

def stocks_ticker():
    bbca = yf.Ticker('BBCA.JK')
    bbca_df = bbca.history(period='5y')
    
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
    split_data = int(len(features_norm)*0.7)
    
    X_train, X_test = features_norm[:split_data], features_norm[split_data:]
    y_train, y_test = target_norm[:split_data], target_norm[split_data:]
    
    return X_train, X_test, y_train, y_test

def voting_model():
    rf = RandomForestRegressor(n_estimators=50,
                               max_depth=20,
                               max_features=4,
                               min_samples_leaf=2,
                               min_samples_split=2)
    xgb = XGBRegressor(n_estimators=100,
                       eta=0.15,
                       max_depth=10,
                       subsample=0.7)
    
    return VotingRegressor(estimators=[
        ('rf', rf),
        ('xgb', xgb),
    ], weights=None)


def predict_data():
    X_train, X_test, y_train, y_test = train_test_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    model = voting_model()
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred


def evaluation_data():
    y_test, y_pred = predict_data()
    
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rmse, mae, r2
    

stocks_ticker()
preprocessing_data()
train_test_data()
voting_model()
predict_data()
print(evaluation_data())