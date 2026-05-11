import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Modelos_de_Regresion import modelo_regresion_SGD


def evaluar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2


# Cargar el conjunto de datos de California Housing
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="MedHouseVal")


# Dividir en entrenamiento y prueba (70%-30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Estandarizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cargar los modelos entrenados previamente
modelo_arbol = joblib.load(
    "modelo_arbol_regresion.pkl"
)  # Cargar el modelo de Árbol de Decisión
modelo_svm = joblib.load("modelo_svm_regresion.pkl")  # Cargar el modelo SVM
modelo_regresion = joblib.load(
    "modelo_regresion_SGD.pkl"
)  # Cargar el modelo de Regresión Logística
modelo_nn = tf.keras.models.load_model(
    "modelo_red_neuronal_regresion.h5"
)  # Cargar la red neuronal (modelo Keras)

# Evaluar los modelos  optimizados
mse_sgd, rmse_sgd, mae_sgd, r2_sgd = evaluar_modelo(
    modelo_regresion, X_test, y_test
)
mse_arbol, rmse_arbol, mae_arbol, r2_arbol = evaluar_modelo(
    modelo_arbol, X_test, y_test
)
mse_svm, rmse_svm, mae_svm, r2_svm = evaluar_modelo(modelo_svm, X_test, y_test)
mse_nn, rmse_nn, mae_nn, r2_nn = evaluar_modelo(modelo_nn, X_test, y_test)

# Métricas para los modelos
metrics_models = {
    "Modelo": ["Regresión Lineal", "Árbol de Decisión", "SVM", "Red Neuronal"],
    "MSE": [mse_sgd, mse_arbol, mse_svm, mse_nn],
    "RMSE": [rmse_sgd, rmse_arbol, rmse_svm, rmse_nn],
    "MAE": [mae_sgd, mae_arbol, mae_svm, mae_nn],
    "R²": [r2_sgd, r2_arbol, r2_svm, r2_nn],
}

# Crear DataFrames para las métricas
df_classic = pd.DataFrame(metrics_models)
# Mostrar la comparación
print(df_classic)

# Graficar la comparación de las métricas (MSE, RMSE, MAE, R²)
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# MSE
ax[0, 0].bar(df_classic["Modelo"], df_classic["MSE"], color="skyblue")
ax[0, 0].set_title("Comparación de MSE")
ax[0, 0].set_ylabel("MSE")

# RMSE
ax[0, 1].bar(df_classic["Modelo"], df_classic["RMSE"], color="lightcoral")
ax[0, 1].set_title("Comparación de RMSE")
ax[0, 1].set_ylabel("RMSE")

# MAE
ax[1, 0].bar(df_classic["Modelo"], df_classic["MAE"], color="lightgreen")
ax[1, 0].set_title("Comparación de MAE")
ax[1, 0].set_ylabel("MAE")

# R²
ax[1, 1].bar(
    df_classic["Modelo"], df_classic["R²"], color="lightgoldenrodyellow"
)
ax[1, 1].set_title("Comparación de R²")
ax[1, 1].set_ylabel("R²")

# Ajustes de la gráfica
plt.tight_layout()
plt.show()
