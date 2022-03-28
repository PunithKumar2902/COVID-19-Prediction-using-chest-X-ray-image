## Predicting Models

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model=load_model('x-ray_model.h5')
img=image.load_image('val/PNEUMONIA/person1946_bacteria_4874.jpeg',target_size=(224,224))
x=image.imd_to_array(img)
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)

class=model.predict(img_data)

print(class)
