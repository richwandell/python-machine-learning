import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K

df = pd.read_csv("../data/test.csv.gz")
X = np.array(df, dtype=float) / 255

model = load_model('../trained/keras_conv_neural_network.h5')

if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X = X.reshape(X.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

predicted = np.argmax(model.predict(X), axis=1)

np.savetxt('submission.csv', np.c_[range(1, predicted.shape[0] + 1), predicted], delimiter=',',
           header='ImageId,Label', comments='', fmt='%d')


