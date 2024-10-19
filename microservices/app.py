import modules

app = modules.Flask(__name__)
cors = modules.CORS(app, resources={r'/api/*':{'origins': 'http://127.0.0.1:8000/'}})
app.json.sort_keys = False


# Setup Cache
# cache = modules.Cache()
# cache.init_app(app, config={'CACHE_TYPE': 'simple'})


# Setup Default Ticker
ticker_available = ['ARTO.JK','BBCA.JK','BBNI.JK','BBRI.JK','BBTN.JK','BMRI.JK','BRIS.JK']


@app.route('/')
def index():
    return 'Hello World'


@app.route('/api/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    if ticker not in ticker_available:
        return modules.jsonify({
            'success': False,
            'message': f'Ticker {ticker} is not available.',
        }), 503
    
    try:
        ticker_market = modules.ModelPredict(ticker)
        df, dates, close = ticker_market.stocks_ticker()
        df_close = close.tolist()
        df_dates = dates.strftime('%Y-%m-%d').tolist()
        
        return modules.jsonify({
            'success': True,
            'message': 'Fetch data successfully',
            'data': {
                'index': df_dates,
                'close': df_close,
            }
        }), 200
    except Exception as e:
        return modules.jsonify({
            'success': False,
            'message': 'Cannot fetch data',
            'error': str(e),
        }), 503


@app.route('/api/<ticker>/predict', methods=['GET'])
# @cache.cached(timeout=86400)
def get_predict_stock_data(ticker):
    if ticker not in ticker_available:
        return modules.jsonify({
            'success': False,
            'message': f'Ticker {ticker} is not available.',
        }), 503
    
    try:
        # Parameter URL
        days = modules.request.args.get('days', type=int)
        window = modules.request.args.get('window', type=int)
        
        # Get the data from API
        ticker_market = modules.ModelPredict(ticker)
        df, dates, close = ticker_market.stocks_ticker()
        
        # Preprocessing Data
        features, target = ticker_market.preprocessing_data(df)
        ticker_market.train_test_data(features, target)
        
        # Model
        ticker_market.voting_model()
        ticker_market.fit_model()
        
        # Predict data with test data
        y_test_reversed, y_pred_reversed = ticker_market.predict_data()
        
        # Evaluation result
        rmse, mae, r2 = ticker_market.evaluation_data(y_test_reversed, y_pred_reversed)
        
        # Actual + Predict t+days
        dates_pred, close_pred, combined_dates, combined_close = ticker_market.combine_actual_predict(days)
        
        # Turn dates and close (predict) to list
        dates_pred_list = [date.strftime('%Y-%m-%d') for date in dates_pred]
        close_pred_list = close_pred.tolist()
        
        # Turn dates and close (actual + predict) to list
        combined_dates_list = [date.strftime('%Y-%m-%d') for date in combined_dates]
        combined_df_list = combined_close.tolist()
        
        # Evaluation Model to list
        eva = [rmse,mae,r2]
        
        # Bollinger Bands
        sma, upper, lower = ticker_market.bollinger_bands(combined_close, window)
        
        return modules.jsonify({
            'success': True,
            'message': 'Predict successfully',
            'data': {
                'ticker': ticker,
                'index_pred': dates_pred_list,
                'close_pred': close_pred_list,
                'index_combined': combined_dates_list,
                'df_combined': combined_df_list,
                'upper_band': upper,
                'sma_band': sma,
                'lower_band': lower,
                'evaluation_model': eva,
            },
        }), 200
    except Exception as e:
        return modules.jsonify({
            'success': False,
            'error': str(e),
            'message': 'We have an error or we are just updating data',
        }), 503


if __name__ == '__main__':
    app.run(debug=True, port=5000)