"""

This is for Generating Confusion Matrices for CNN classification models
By Patrick Roe, on 2022/07/30

"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Input Data Locations
PREDICTION_OUT_PATH = "test.out"
TESTING_LABELS_PATH = "..\FileManipulation\Tadaki_labels.dat"

# Confusion Plot Details
CONFUSION_MATRIX_PATH = "confusion_matrices_3\confusion_Tadaki.png"
CONFUSION_MATRIX_TITLE = "Confusion Matrix for Tadaki Binary Classification"

# Confusion Matrix Config
CLASS_LABELS = ["Non-Spiral", "Spiral"]

def loadData():
    
    # Dictionary for converting to binary (Spiral & Non-Spiral) classification
    label_dict = {
        b"1": 0,
        b"2": 0,
        b"3": 1
    }
    
    # Converter to subtract 1 from all labels
    label_converter = lambda x: label_dict[x];       
    
    # Load Testing Labels
    testing_labels = np.genfromtxt(TESTING_LABELS_PATH, dtype=np.uint8, max_rows=10000)# converters={0: label_converter})
    
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