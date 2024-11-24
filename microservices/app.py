import modules

app = modules.Flask(__name__)
cors = modules.CORS(app, resources={r'/api/*':{'origins': 'http://127.0.0.1:8000/'}})
app.json.sort_keys = False


# Setup Default Ticker
ticker_available = ['ARTO.JK','BBCA.JK','BBNI.JK','BBRI.JK','BBTN.JK','BMRI.JK','BRIS.JK']


@app.route('/')
def index():
    return 'This is API to show and predict stock market IDX ARTO, BBCA, BBNI, BBTN, BRIS, BMRI, BBRI.'


@app.route('/api/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    if ticker not in ticker_available:
        response = modules.jsonify({
            'success': False,
            'message': f'Ticker {ticker} is not available.',
        })
        
        response.headers['Content-Type'] = 'application/json'
        response.headers['X-Error-Code'] = 500
        
        return response, 500
    
    try:
        ticker_market = modules.ModelPredict(ticker)
        df, dates, close = ticker_market.stocks_ticker()
        
        response = modules.jsonify({
            'success': True,
            'message': 'Fetch data successfully',
            'data': {
                'index': dates,
                'close': close,
            }
        })
        
        response.headers['Content-Type'] = 'application/json'
        response.headers['Accept'] = 200
        
        return response, 200
    
    except Exception as e:
        response = modules.jsonify({
            'success': False,
            'message': 'Cannot fetch data',
            'error': str(e),
        })
        
        response.headers['Content-Type'] = 'application/json'
        response.headers['X-Error-Code'] = 503
        
        return response, 503


@app.route('/api/<ticker>/predict', methods=['GET'])
def get_predict_stock_data(ticker):
    if ticker not in ticker_available:
        response = modules.jsonify({
            'success': False,
            'message': f'Ticker {ticker} is not available.',
        })
        
        response.headers['Content-Type'] = 'application/json'
        response.headers['X-Error-Code'] = 500
        
        return response, 500
    
    try:
        # Parameter URL
        days = modules.request.args.get('days', type=int)
        window = modules.request.args.get('window', type=int)
        
        # Get the data from API
        ticker_market = modules.ModelPredict(ticker)
        df, _, _ = ticker_market.stocks_ticker()
        
        # Create Lag Features
        df_with_lags = ticker_market.create_lag(df, days=days)
        
        # Preprocessing Data
        ticker_market.preprocessing_data(df_with_lags)
        
        # Model
        ticker_market.voting_model(ticker)
        ticker_market.fit_model()
        
        # Predict data with test data
        y_pred = ticker_market.predict_data()
        
        # Evaluation result
        rmse, mae, r2 = ticker_market.evaluation_data(y_pred)
        
        # Actual + Predict t+days
        dates_pred, close_pred, combined_dates, combined_close = ticker_market.combine_actual_predict(df, days)
        
        # Bollinger Bands
        sma, upper, lower = ticker_market.bollinger_bands(combined_close, window)
        
        # Status
        status = ticker_market.status_predict(close_pred)
        
        response = modules.jsonify({
            'success': True,
            'message': 'Predict successfully',
            'data': {
                'ticker': ticker,
                'index_pred': dates_pred,
                'close_pred': close_pred,
                'index_combined': combined_dates,
                'df_combined': combined_close,
                'upper_band': upper,
                'sma_band': sma,
                'lower_band': lower,
                'status': status,
                'evaluation_model': {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                },
            },
        })
        
        response.headers['Content-Type'] = 'application/json'
        response.headers['Accept'] = 200
        
        return response, 200
    
    except Exception as e:
        response = modules.jsonify({
            'success': False,
            'error': str(e),
            'message': 'We have an error or we are just updating data',
        })
        
        response.headers['Content-Type'] = 'application/json'
        response.headers['X-Error-Code'] = 503
        
        return response, 503


if __name__ == '__main__':
    app.run(debug=True, port=5000)