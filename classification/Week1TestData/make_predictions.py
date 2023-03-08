from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import os


IMAGES_PATH = r"..\FileManipulation\PDR2_i20_images.dat"
NUM_IMAGES = 100

IMG_HEIGHT = 64
IMG_WIDTH = 64

# Filenames for Model
MODEL_PATH = "model.json"
MODEL_WEIGHTS_PATH = ".\checkpoints\ 39.h5"

def loadData():
    
    # Load & Reshape Testing Images
    testing_images = np.genfromtxt(IMAGES_PATH, dtype=np.single, max_rows=NUM_IMAGES*IMG_WIDTH*IMG_WIDTH)
    testing_images = np.reshape(testing_images, (NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, 1))
    
    return testing_images
    

def loadModel():
    json_file = open(MODEL_PATH, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(MODEL_WEIGHTS_PATH)

    return loaded_model
  
def makePredictions(model, images):
    type_dict = {0: 'Non-Spiral', 1: 'Spiral'}
  
    predictions = model.predict(images)
  
    for idx, image in enumerate(images):
      galaxy_type = type_dict[round(predictions[idx, 0])]
      plt.imshow(image, cmap='gray')
      plt.title(f"Model Prediction: {galaxy_type}, Confidence: {predictions[idx,0]:.3f}") 
      plt.savefig(os.path.join(os.getcwd(), "PDR2_i20_predictions",f"{idx}_prediction"))
  
  
model = loadModel()
images = loadData()

makePredictions(model, images)