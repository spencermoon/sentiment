import os
import streamlit as st 
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from helper.preprocess import preprocess
import pickle
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from xgboost import XGBClassifier
import nltk
nltk.download('stopwords')

# Set font formatting
font = {'family' : 'normal',
        'weight' : 'light',
        'size' : 7,
        }
rc('font', **font)

# Load Doc2Vec model
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
d2v = Doc2Vec.load('model/d2v.model')

# Load tree model
open_file = open('model/xgboost.pkl', "rb")
model = pickle.load(open_file)
open_file.close()

# Create user input box
tweet = st.text_input('Write your own tweet here:', key='tweet', max_chars=140)

# Surface sentiment analysis
if len(tweet)>0: 

	# Preprocess data
	text = preprocess(tweet)
	text = d2v.infer_vector(text)
	text = pd.DataFrame([text])

	# Make prediction
	prediction = model.predict_proba(text)
	results = pd.DataFrame({'Sentiment': ['Negative', 'Positive'],
			   				'Percentage': [prediction[0][0]*100, 
			   							   prediction[0][1]*100]})

	# Draw chart for probabilities
	fig, ax = plt.subplots()
	fig.patch.set_alpha(0.0)
	ax.barh(results['Sentiment'], results['Percentage'], 
			color=['#F9B5AC', '#9DBF9E'])
	ax.bar_label(ax.containers[0], padding=3, fmt='%.1f%%')
	ax.axes.get_xaxis().set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_color('#262730')
	ax.tick_params(axis=u'both', which=u'both',length=0)
	st.pyplot(fig)