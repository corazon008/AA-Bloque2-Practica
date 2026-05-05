from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# Función para evaluar un modelo
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

from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# REGRESION LINEAL CON SGD
max_iter_sgd = 10  # Número máximo de iteraciones
alpha_sgd = 0.1  # Regularización
eta0_sgd = 0.1  # Tasa de aprendizaje inicial

modelo_regresion_SGD = SGDRegressor(
    max_iter=max_iter_sgd, alpha=alpha_sgd, eta0=eta0_sgd, random_state=42
)
# Entrenamos el modelo
modelo_regresion_SGD.fit(X_train, y_train)
mse, rmse, mae, r2 = evaluar_modelo(modelo_regresion_SGD, X_test, y_test)
print("Modelo:Regresión Lineal con SGD")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}\n")

# Graficar las predicciones vs valores reales para cada modelo
plt.figure(figsize=(10, 8))
y_pred = modelo_regresion_SGD.predict(X_test)
plt.scatter(y_test, y_pred, color="blue", label="Predicciones")
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    linestyle="--",
)  # Línea ideal
plt.title("Regresión Lineal con SGD - Predicciones vs Reales")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()
plt.tight_layout()
plt.show()

# ÁRBOL  DE REGRESIÓN
max_depth_tree = 1000
min_samples_split_tree = 3000
min_samples_leaf_tree = 500
modelo_arbol_re = DecisionTreeRegressor(
    max_depth=max_depth_tree,
    min_samples_split=min_samples_split_tree,
    min_samples_leaf=min_samples_leaf_tree,
    random_state=42,
)
modelo_arbol_re.fit(X_train, y_train)

mse, rmse, mae, r2 = evaluar_modelo(modelo_arbol_re, X_test, y_test)
print("Modelo:Árbol de Regresión")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}\n")

# Graficar las predicciones vs valores reales para cada modelo
plt.figure(figsize=(10, 8))
y_pred = modelo_arbol_re.predict(X_test)
plt.scatter(y_test, y_pred, color="blue", label="Predicciones")
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    linestyle="--",
)  # Línea ideal
plt.title("Árbol de regresión - Predicciones vs Reales")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()
plt.tight_layout()
plt.show()

# MODELO SVM
C_svr = 1
epsilon_svr = 1
kernel_svr = "linear"
modelo_svm = SVR(C=C_svr, epsilon=epsilon_svr, kernel=kernel_svr)
modelo_svm.fit(X_train, y_train)

mse, rmse, mae, r2 = evaluar_modelo(modelo_svm, X_test, y_test)
print("Modelo:SVM en Regresión")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}\n")

# Graficar las predicciones vs valores reales para cada modelo
plt.figure(figsize=(10, 8))
y_pred = modelo_svm.predict(X_test)
plt.scatter(y_test, y_pred, color="blue", label="Predicciones")
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    linestyle="--",
)  # Línea ideal
plt.title("SVM en Regresión - Predicciones vs Reales")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()
plt.tight_layout()
plt.show()

# Guardamos los modelos
joblib.dump(modelo_arbol_re, "modelo_arbol_regresion.pkl")
joblib.dump(modelo_svm, "modelo_svm_regresion.pkl")
joblib.dump(modelo_regresion_SGD, "modelo_regresion_SGD.pkl")
