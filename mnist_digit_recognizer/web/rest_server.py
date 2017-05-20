from flask import Flask, request, send_from_directory, jsonify
from sklearn.externals import joblib
from keras.models import load_model
import numpy as np
app = Flask(__name__)


sknn = joblib.load('../trained/sklearn_neural_network.pkl')
pca = joblib.load('../trained/sklearn_pca.pkl')
knn = load_model('../trained/keras_neural_network.h5')


@app.route("/")
def hello():
    return send_from_directory(".", "test.html")


@app.route("/guess", methods=['POST'])
def guess():
    pixels = request.form.getlist('pixels[]', type=float)
    pixels = np.array(pixels) / 255
    test_data = pca.transform(pixels)
    sknn_prediction = sknn.predict(test_data).ravel()[0]
    knn_prediction = np.argmax(knn.predict(test_data), axis=1).ravel()[0]
    return jsonify({"guess": {
        "sknn": int(sknn_prediction),
        "knn": int(knn_prediction)
    }})


if __name__ == "__main__":
    app.run(port=8080)

