"""

This is for Generating Confusion Matrices for CNN classification models
By Patrick Roe, on 2022/07/30

"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Input Data Locations
PREDICTION_OUT_PATH = "test.out"
TESTING_LABELS_PATH = "nam_labels_test.dat"

# Confusion Plot Details
CONFUSION_MATRIX_PATH = "confusion_matrices_2/confusion_100epoch_100batch.png"
CONFUSION_MATRIX_TITLE = "Confusion Matrix (100 epochs & 100 batch size)"

# Confusion Matrix Config
CLASS_LABELS = ["E", "S0", "Sp"]

def loadData():
    
    # Converter to subtract 1 from all labels
    label_converter = lambda x: int(x) - 1;   
    
    # Load Testing Labels
    testing_labels = np.genfromtxt(TESTING_LABELS_PATH, dtype=np.uint8, converters={0: label_converter})
    
    # Load Prediction Labels
    prediction_labels = np.genfromtxt(PREDICTION_OUT_PATH, dtype=np.uint8, skip_header=1, usecols=(0))

    return (testing_labels, prediction_labels)


def saveConfusionMatrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)

    disp.plot(cmap=plt.cm.Blues)
    plt.title(CONFUSION_MATRIX_TITLE)
    
    #plt.show()
    
    plt.savefig(CONFUSION_MATRIX_PATH)
    


y_test, y_pred = loadData()
saveConfusionMatrix(y_test, y_pred)