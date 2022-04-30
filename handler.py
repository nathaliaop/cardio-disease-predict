import os
import pandas as pd
import pickle
from flask import Flask, request

from api_preprocessing import ApiPreprocessing

# import model
classifier = pickle.load(open('deploy/cardio_kernel_svm.pkl', 'rb'))

# instanciate flask
app = Flask(__name__)

# route for predictions
@app.route('/predict', methods=['POST'])
def predict():
  response_json = request.get_json()
  df = pd.json_normalize(response_json)
  # instanciate data preprocessing
  pipeline = ApiPreprocessing()
  # data preprocessing
  df_preprocessed = pipeline.data_preprocessing(df, response_json)
  # predict
  pred = classifier.predict(df_preprocessed)

  # return prediction
  return str(pred[0])

if __name__ == '__main__':
  # start flask
  port = os.environ.get('PORT', 5000)
  app.run(host='0.0.0.0', port=port)