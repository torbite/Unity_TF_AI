#~/Tiago/Python/gmae4/

#import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
import data_manipulator as dm
from copy import deepcopy


def create_model(input_shape) -> Sequential:
    new_model = Sequential([
        Dense(128, "relu", input_shape=input_shape),
        Dense(256, "relu"),
        Dense(128, "relu"),
        Dense(64, "relu"),
        Dense(32),
        Dense(1)
    ])

    new_model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["accuracy"]
    )

    return new_model


def train_model(model: Sequential, epochs : int, data_file: str, learning_rate : float):

    trained_model = clone_model(model)
    optmizer = Adam(learning_rate = learning_rate)
    trained_model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["accuracy"]
    )
    data = dm.load_data(data_file)

    # print(len(data["X"]))
    # print(len(data["y"]))

    X = np.array(data["X"])  # Directly convert to NumPy array
    if X.ndim == 1:  # If it's 1D, reshape it to (n_samples, 1)
        X = X.reshape(-1, 4)
    y = np.array(data["y"])

    # print(len(X))
    # print(len(y))
    # input()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    early_stop = EarlyStopping(
        monitor = "val_loss",
        patience = 5,
        restore_best_weights = True
    )

    trained_model.fit(
        X_train, 
        y_train, 
        epochs = epochs, 
        validation_data=(X_test, y_test), 
        # callbacks=early_stop
    )

    return trained_model


if __name__ == "__main__":
    model = create_model()
    test_prediction = model.predict([[1]])
    model = train_model(model, 200, "test_file", 0.00001)
    x_test = [[i] for i in range(200)]
    test_prediction_2 = model.predict(x_test)
    
    dm.load_data("test_file")
