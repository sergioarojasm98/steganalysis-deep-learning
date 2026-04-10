import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os

# Esta función lee los datos de un archivo CSV y devuelve los valores y_true y y_pred_prob
def read_csv(file_path):
    data = pd.read_csv(file_path)
    frr = data['FPR']  
    tpr = data['TPR']  
    thresholds = data['Thresholds']  
    return frr, tpr, thresholds

# Esta función grafica la curva ROC para los datos proporcionados
def plot_roc_curve(file_path, label):
    fpr, tpr, thresholds = read_csv(file_path)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")

# Configuración inicial de la figura
plt.figure(figsize=(10, 7))

# Graficar las curvas ROC para cada archivo CSV
plot_roc_curve('/Users/sergiorojas/Documents/GitHub/tg2/Resultados/CSV_Files/ROC_Data_CNN_Test_10.csv', 'CNN 1')
plot_roc_curve('/Users/sergiorojas/Documents/GitHub/tg2/Resultados/CSV_Files/ROC_Data_CNN_Test_11.csv', 'CNN 2')
plot_roc_curve('/Users/sergiorojas/Documents/GitHub/tg2/Resultados/CSV_Files/ROC_Data_CNN_Test_12.csv', 'CNN 3')

# Configuración adicional del gráfico
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC ROC para el Conjunto de Prueba")
plt.legend(loc="lower right")

# Guardar la figura
file_path = os.environ.get("TG2_HOME", "/home/srojas/tg2") + "/Resultados/ROC_Test_Combined.pdf"
plt.savefig(file_path)
plt.close()
