from tensorflow.keras.applications import resnetv50
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

model = resnetv50(
    weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
)
