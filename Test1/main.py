import flask
from flask import request, jsonify
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
import data_manipulator as dm
import model_manipulator as ml
from copy import deepcopy

# input = [xpos-txpos, ypos-trypos, xinput, yinput]
# example: [-4, 4, 1, 1]
# example: [5, 12, 1, 0]
# example: [-2, -3, -1, 1]


app = flask.Flask(__name__)

models = {}

@app.route('/start_model/<id>', methods=['GET'])
def start_model(id):
    model = ml.create_model((4,))
    modelName = f"model{id}.h5"
    model.save(modelName)
    models[id] = modelName
    return {"message": "Model started successfully"}

@app.route('/predict/<id>', methods=['POST'])
def predict(id):
    model = load_model(f"model{id}.h5")
    data = request.get_json()["floats"]
    possibilities = []
    pairs = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            poss = (x, y)
            #print(data)
            #input()
            pairs.append(poss)
            input_data = np.array([data + list(poss)])
            possibilities.append(input_data)
    
    #convert the predictions to a list of dictionaries
    input_batch = np.vstack(possibilities)  # Stack all possibilities into a single batch
    
    # Predict the possibilities with the model
    predictions = model.predict(input_batch, batch_size=9)
    
    #print(predictions)

    

    best_index = np.argmax(predictions)  # Get the index of the highest prediction
    
    # Get the corresponding possibility based on the highest prediction
    best_possibility = pairs[best_index]
    print(input_batch)
    print(predictions)
    print(best_index, best_possibility)
    print()
    return jsonify({"values" : best_possibility})

@app.route('/send_data', methods=['POST'])
def send_data():
    data = request.get_json()
    data = data["dict"]
    X = []
    y = []
    for dic in data:
        X.append(dic["key"])
        y.append(dic["value"])
    # for values in X:

    #X = data["key"]
    #y = data["value"]
    
    a = dm.add_data("trainData", {"X": X, "y": y})
    
    return {"message": "Data sent successfully"}

@app.route('/train_models', methods=['GET'])
def train_model():
    print("Train model")
    print(models.items())
    for id, model_name in models.items():
        model = load_model(model_name)
        trained_model = ml.train_model(model, 100, "trainData", 0.0001)
        trained_model.save(model_name)
    print("models trained")
    # input()
    return {"message": "Models trained successfully"}


    
if __name__ == '__main__':
    app.run(port=5000, debug=True)
    