from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

import numpy as np
from typing import Any, Tuple

import plotly.express as px
import plotly

def regression_scale_fit_mse(
    model, 
    params:dict, 
    X_train:np.ndarray, 
    y_train:np.ndarray, 
    X_test:np.ndarray, 
    y_test:np.ndarray
    ) -> Tuple[float, np.ndarray, Any]:
    """
    This function first scales the training dataset and then fits an sklearn model to it.
    Then makes the prediction on the test dataset and calculates the MSE. 

    Args:
        model (estimator instance): sklearn estimator instance
        params (dict): parameters of the estimator instance
        X_train (np.ndarray): training sample for X
        y_train (np.ndarray): training sample for y
        X_test (np.ndarray): test sample for X
        y_test (np.ndarray): test sample for y

    Returns:
        tuple: tuple of MSE, prediction of y on the test dataset, pipeline object
    """
    model = model(**params)
    scaler = MinMaxScaler(feature_range=(0,1))
    pipe = Pipeline([('scaler', scaler), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return mse, y_pred, pipe


def draw_actual_vs_predicted(
    y_test:np.ndarray, 
    y_pred:np.ndarray
    ):

    """
    Draws the graph of the actual and predicted values.
    """

    cor = np.corrcoef(y_test, y_pred)[0][1]
    fig = px.scatter(x=y_test, y=y_pred)
    fig.update_layout(
        title=f"Correlation between the actual and predicted values: {round(cor, 5)}", 
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=800, width=800)

    return fig