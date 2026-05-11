from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD

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

# Modelo
modelo = Sequential(
    [
        # Capa de entrada con 4 neuronas y función de activación ReLU
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ]
)

# Compilar y entrenar el modelo
modelo.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
history = modelo.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    verbose=1,
    validation_data=(X_test, y_test),
)
# Guardar el modelo en formato .h5
modelo.save("modelo_red_neuronal_regresion.h5")
mse, rmse, mae, r2 = evaluar_modelo(modelo, X_test, y_test)
print(
    f"Red Neuronal - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
)


# Graficar el proceso de entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Pérdida de entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida de validación")
plt.title("Pérdida durante el entrenamiento de la Red Neuronal")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.show()
