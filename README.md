# IDX LQ45 Banking Stock Signal Movement Prediction
## Description
This application is an application to predict the movement of IDX LQ45 stocks in the banking sector for the next 7 days. This application prediction uses machine learning methods, namely Voting Regression (combining random forest and xgboost) and technical indicators, namely Bollinger Bands.

## Prequisites
* Minimum Operating Systems: Windows 10
* Minimum Required Software: 
    * PHP 8.2
    * Python 3.12
    * Laravel 10
    * Composer 2.6.6
* Minimum Hardware:
    * Processor: AMD Ryzen 5 4600H or Intel Core i5
    * RAM: 8.0 GB
    * Disk Space: 2GB

## Installation Methods
### Windows
```
# Clone the Repository
> git clone https://github.com/Kydens/prediksi-IDX-LQ45-Bank.git

# Navigate to the project directory
> cd prediksi-IDX-LQ45-Bank

# Install all dependencies Python and Laravel
# Enter Microservices for Python
> cd microservices
> pip install -r requirement.txt
> flask --debug run
# This can be access at http://localhost:5000

# Enter Full Stack for Laravel
> cd fullStack
> php artisan key:generate
> php artisan serve
# This can be access at http://localhost:<your-localhost-port>
```

**Optional Installation**
```
# Run the python code in virtual environment
> python3 -m venv venv
> .\venv\Scripts\activate
```

## Configurations
### Enviroment Variables
Copy ```.env``` file in the fullStack folder root and add with the following variables:
```
MICROSERVICES_APP_URL=<your-localhost-python-microservices>
APIKEY_SERPAPIGOOGLENEWS='<your-serpAPI-apiKey>'
```
To get API Key of ```APIKEY_SERPAPIGOOLENEWS=```, you should have serpAPI account to get serpAPI APIKey. This APIKey is to show news in homepage or newspage.