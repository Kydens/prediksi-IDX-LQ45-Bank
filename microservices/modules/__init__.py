from modules.prediction import ModelPredict
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_caching import Cache