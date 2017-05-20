import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('data/training.csv.gz')

image_1 = np.array(list(float(x) for x in train['Image'].values[0].split(" ")), dtype=float)

img = np.reshape(
    # greyscale images using the same value for R G B
    np.column_stack(
        (image_1, image_1, image_1)
    ),
    (96, 96, 3)
)

plt.imshow(img)

plt.show()