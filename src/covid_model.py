# linear regression for multioutput regression
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def run_the_model():
    df = pd.read_excel('dataset.xlsx')
    X = df[['Age', 'heart beat in bpm', 'Gender']]
    y = df[['Number of steps', 'Sleep']]

    # define model
    model = LinearRegression()

    # fit model
    model.fit(X, y)

    pickle.dump(model, open('model.pkl', 'wb'))



def predict(data):
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict([np.array(data)])
    prediction = [el.round(0) for el in prediction]
    return prediction

