from flask import Flask, request, send_from_directory, jsonify
from flask_socketio import SocketIO, send, emit
import io
import numpy as np
from PIL import Image
from keras.models import load_model
from keras import backend as K

app = Flask(__name__)
socket = SocketIO(app)
model = load_model('../trained/keras_conv_neural_network.h5')


@socket.on('blob')
def blob(data):
    image = Image.open(io.BytesIO(data))
    im = np.array(image)
    h, w, c = im.shape
    p = float(h / w)
    move = (w - float(w * p))
    s = int(move / 2)
    e = int(w - move / 2)
    imn = im[0:h, s:e]
    image = Image.fromarray(imn)
    image.thumbnail((96, 96), Image.ANTIALIAS)
    image = np.array(image)
    X = image[0:96, 0:96, 1] / 255
    X = X.reshape(1, 96, 96, 1)
    y_pred = model.predict(X)

    out = y_pred.reshape((15, 2)) * 48 + 48

    emit('blob-saved', out.tolist())


@app.route("/")
def hello():
    return send_from_directory(".", "test.html")

if __name__ == "__main__":
    socket.run(app, port=8080, host='0.0.0.0')

