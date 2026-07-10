import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# 1. Definición del Dataset (24 estudiantes)
data = {
    'Estudiante': [
        'Ana Perez', 'Juan Rodriguez', 'Maria Gomez', 'Pedro Martinez',
        'Laura Diaz', 'Carlos Santos', 'Sofia Ramirez', 'Luis Herrera',
        'Valentina Cruz', 'Miguel Lopez', 'Camila Torres', 'Andres Castillo',
        'Daniela Reyes', 'Jose Fernandez', 'Paula Morales', 'Diego Vargas',
        'Gabriela Nunez', 'Javier Medina', 'Elena Rosario', 'Adrian Pena',
        'Natalia Jimenez', 'Samuel Ortiz', 'Isabella Guerrero', 'Kevin Batista'
    ],
    'Asistencia': [
        92, 65, 95, 58, 88, 72, 97, 60, 90, 55, 85, 68,
        93, 62, 89, 70, 96, 52, 87, 66, 91, 59, 94, 74
    ],
    'Tareas_Anteriores_Entregadas': [
        9, 4, 10, 3, 8, 5, 10, 4, 9, 2, 7, 5,
        9, 4, 8, 6, 10, 2, 8, 5, 9, 3, 10, 6
    ],
    'Recordatorio_Enviado': [
        1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1
    ],
    'Entrego': [
        1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1
    ]
}

df = pd.DataFrame(data)

# ==========================================
# MODELO 1: Regresión Logística (Clasificación)
# ==========================================
# Predecir 'Entrego' usando ['Asistencia', 'Tareas_Anteriores_Entregadas', 'Recordatorio_Enviado']
X_cls = df[['Asistencia', 'Tareas_Anteriores_Entregadas', 'Recordatorio_Enviado']]
y_cls = df['Entrego']

# División entrenamiento/prueba (70/30) con estratificación
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.30, random_state=42, stratify=y_cls
)

# Escalado de características (esencial para Regresión Logística debido a regularización)
scaler = StandardScaler()
X_train_cls_scaled = scaler.fit_transform(X_train_cls)
X_test_cls_scaled = scaler.transform(X_test_cls)

# Ajuste del modelo
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_cls_scaled, y_train_cls)

# Evaluación
y_pred_cls = logistic_model.predict(X_test_cls_scaled)
acc = accuracy_score(y_test_cls, y_pred_cls)

print("--- REGRESIÓN LOGÍSTICA ---")
print(f"Exactitud en prueba: {acc * 100:.1f}%")
print(f"Coeficientes: {logistic_model.coef_[0]}")
print(f"Intercepto: {logistic_model.intercept_[0]}")

# ==========================================
# MODELO 2: Regresión Lineal (Predicción Continua)
# ==========================================
# Predecir la variable continua 'Tareas_Anteriores_Entregadas' usando ['Asistencia', 'Recordatorio_Enviado']
X_reg = df[['Asistencia', 'Recordatorio_Enviado']]
y_reg = df['Tareas_Anteriores_Entregadas']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.30, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)

# Evaluación
y_pred_reg = linear_model.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n--- REGRESIÓN LINEAL ---")
print(f"MSE en prueba: {mse:.4f}")
print(f"RMSE en prueba: {np.sqrt(mse):.4f}")
print(f"R² en prueba: {r2:.4f}")
print(f"Coeficientes (Asistencia, Recordatorio): {linear_model.coef_}")
print(f"Intercepto: {linear_model.intercept_}")

# ==========================================
# MODELO 3: Modelo de Probabilidad Lineal (LPM)
# ==========================================
# Predecir 'Entrego' usando Regresión Lineal directamente para comparar con Regresión Logística
lpm_model = LinearRegression()
lpm_model.fit(X_train_cls, y_train_cls) # Usamos X_train sin escalar para facilitar la interpretación directa
y_pred_lpm = lpm_model.predict(X_test_cls)
mse_lpm = mean_squared_error(y_test_cls, y_pred_lpm)

print("\n--- MODELO DE PROBABILIDAD LINEAL (LPM) ---")
print(f"MSE en prueba: {mse_lpm:.4f}")
print(f"Coeficientes (Asistencia, Tareas, Recordatorio): {lpm_model.coef_}")
print(f"Intercepto: {lpm_model.intercept_}")

# Guardar los modelos y el escalador para Streamlit
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(linear_model, f)
with open('lpm_model.pkl', 'wb') as f:
    pickle.dump(lpm_model, f)

# Guardar también el DataFrame completo para usar en la app si es necesario
with open('student_data.pkl', 'wb') as f:
    pickle.dump(df, f)

print("\n[OK] Todos los modelos han sido entrenados y guardados exitosamente.")
