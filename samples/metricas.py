from sklearn.metrics import classification_report
from matplotlib 	 import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np

target_names = ['Positivo', 'Negativo']

y_pred= [1,1,2,2,1]
y_true= [1,1,2,1,1]

print(classification_report(y_true, y_pred, target_names=target_names))


print(confusion_matrix(y_true, y_pred))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.show()