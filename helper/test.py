import pickle
import xgboost as xgb
import pandas as pd

def test_predict():

	# Load model
	open_file = open('model/xgboost.pkl', "rb")
	model = pickle.load(open_file)
	open_file.close()

	# Create dummy input value and predict
	dummy = pd.DataFrame([[0]*25])
	result = model.predict_proba(dummy)

	# Check type of model
	assert isinstance(model, xgb.XGBClassifier)

	# Check for positive and negative sentiment probabilities
	assert len(result[0]) == 2