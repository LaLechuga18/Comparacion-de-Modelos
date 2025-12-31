#------------- PRACTICA 8 -------------
# Angel Sebastian Garnica Carbajal
# Comparar la prescision de un arbol de decision y la regresion logistica
# 31/03/2025


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# Cargar el archivo CSV
df = "Pacientes.csv" 
df = pd.read_csv(df)

# Verificar las primeras filas del DataFrame
print(df.head())

# Graficar el pairplot
sns.pairplot(df, hue="problema_cardiaco")

# Variables independientes (edad y colesterol)
variablex = df[["edad", "colesterol"]].values
print("\n Solo edad y colesterol : \n", variablex[:10])

# Variable dependiente (problema cardíaco)
variabley = df["problema_cardiaco"].values.reshape(-1, 1)
print(variabley[:10])

# Crear los datasets de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(variablex, variabley, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árbol de decisión
modelTree = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
modelTree.fit(x_train, y_train)

# Hacer predicciones
y_pred = modelTree.predict(x_test)

# Calcular la precisión y F1-score
f1 = f1_score(y_test, y_pred)  
print(f"El modelo obtuvo un indice F1 de: {f1}")

percent = modelTree.score(x_test, y_test)
print(f"El modelo obtuvo un {percent*100:.2f}% de precision para clasificar")

# Mostrar el árbol de decisión
plt.figure(figsize=(12, 6))
tree.plot_tree(modelTree, feature_names=["Edad", "Colesterol"])
print(tree.export_text(modelTree,feature_names=["edad", "colesterol"]))
plt.show()

# Matriz de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
cm_display.plot(cmap=plt.cm.Blues)
plt.show()
