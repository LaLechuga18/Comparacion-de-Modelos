# Comparacion-de-Modelos-de-Clasificacion
# Practica 8 – Comparación de Modelos de Clasificación  
**Angel Sebastian Garnica Carbajal**  
**Fecha:** 31/03/2025  

Este proyecto compara el rendimiento entre dos modelos clásicos de Machine Learning para la detección de problemas cardiacos:  
- **Árbol de Decisión (DecisionTreeClassifier)**  

El análisis se realiza utilizando un dataset de pacientes, evaluando la relación entre **edad**, **colesterol** y el diagnóstico de **problema cardíaco**.

---

## Objetivo  
Evaluar qué tan bien un modelo basado en árbol de decisión puede clasificar casos positivos y negativos de enfermedades cardíacas utilizando métricas como **F1-Score**, **Accuracy** y la **Matriz de Confusión**.

---

## Dataset  
El código utiliza un archivo llamado **Pacientes.csv**, el cual debe incluir al menos las siguientes columnas:

- `edad`
- `colesterol`
- `problema_cardiaco` (variable objetivo)

Asegúrate de tener el archivo en la misma carpeta que el script.

---

##  Tecnologías utilizadas

- Python 3  
- Pandas  
- NumPy  
- Seaborn  
- Matplotlib  
- Scikit-learn  

---

##  Pasos del analisis

1. **Carga y visualización del dataset**  
   - Se muestran las primeras filas.
   - Se genera un `pairplot` para observar correlaciones entre variables.

2. **Selección de variables**
   - Variables independientes: `edad`, `colesterol`
   - Variable dependiente: `problema_cardiaco`

3. **División entre entrenamiento y prueba**
   ```python
   train_test_split(test_size=0.2, random_state=42)

