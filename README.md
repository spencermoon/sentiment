# Twitter Sentiment App

This application determines the sentiment of a user-provided tweet. It can be 
accessed [here](https://share.streamlit.io/spencermoon/sentiment/main/app.py).
The app serves as the final capstone project submission for the [ML engineering
track](https://www.springboard.com/courses/ai-machine-learning-career-track/) 
on Springboard.

## Approach

The first step in training the model was creating Doc2Vec embeddings from the 
Twitter dataset. The embeddings were split into training and test datasets for 
XGBoost. The model achieved a test accuracy rate of 67%. The script to train 
this model can be found in `/prework/prototype.ipynb`. 

Tuning a pretrained ALBERT model was also another approach that was explored. 
Training the model for just one epoch resulted in a test accuracy rate of over 
85%, a significant increase from the performace of the XGBoost model. A next 
step for improving this project is to implement the tuned ALBERT model.

Streamlit was used to serve the model online. The web app takes a text input
from the user and runs it through the XGBoost model to generate probabilities 
for positive and negative sentiment, which are then surfaced as a bar chart. 
The code to create and serve the app can be found in `app.py`.

## Running Locally

1. Clone the repo and set directory path to the repo in terminal.
2. Activate a virtual environment.
3. Run `pip install -r requirements.txt` to install necessary packages.
4. Conduct a test by running `pytest helper/test.py`	
5. Run `streamlit run app.py` to initiate the app locally.