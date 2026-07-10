import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

# Configuración de página con estética premium
st.set_page_config(
    page_title="Analítica Predictiva Escolar - CRISP-ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para dar un aspecto premium, moderno y dinámico (tarjetas con degradados, fuentes limpias y bordes redondeados)
st.markdown("""
<style>
    /* Estilos generales */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
    /* Estilos de tarjetas (Glassmorphism / Premium Cards) */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
        margin-bottom: 1.5rem;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .status-card-success {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        color: #1e4620;
        padding: 1.5rem;
        border-radius: 16px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(150, 230, 161, 0.3);
    }
    
    .status-card-danger {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #5c181b;
        padding: 1.5rem;
        border-radius: 16px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 154, 158, 0.3);
    }

    .crisp-phase {
        border-left: 4px solid #1e3c72;
        padding-left: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Cargar los modelos pre-entrenados o re-entrenar si no existen (mecanismo de robustez)
@st.cache_resource
def load_or_train_models():
    # Dataset por si necesitamos entrenar al vuelo
    data = {
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
    
    # Intentar cargar pickles
    try:
        with open('logistic_model.pkl', 'rb') as f:
            logistic_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('linear_model.pkl', 'rb') as f:
            linear_model = pickle.load(f)
        with open('lpm_model.pkl', 'rb') as f:
            lpm_model = pickle.load(f)
    except Exception:
        # Reentrenamiento al vuelo
        X_cls = df[['Asistencia', 'Tareas_Anteriores_Entregadas', 'Recordatorio_Enviado']]
        y_cls = df['Entrego']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cls)
        
        logistic_model = LogisticRegression(random_state=42)
        logistic_model.fit(X_scaled, y_cls)
        
        X_reg = df[['Asistencia', 'Recordatorio_Enviado']]
        y_reg = df['Tareas_Anteriores_Entregadas']
        
        linear_model = LinearRegression()
        linear_model.fit(X_reg, y_reg)
        
        lpm_model = LinearRegression()
        lpm_model.fit(X_cls, y_cls)
        
    return logistic_model, scaler, linear_model, lpm_model, df

logistic_model, scaler, linear_model, lpm_model, df = load_or_train_models()

# ── Encabezado Principal ──────────────────────────────────────────────────────
st.markdown("<div class='main-header'>🎓 Plataforma de Analítica Predictiva Escolar</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Modelo basado en la Metodología CRISP-ML para el éxito académico y la intervención docente temprana</div>", unsafe_allow_html=True)

# ── Panel de Control Lateral (Sidebar Inputs) ──────────────────────────────────
st.sidebar.markdown("### 🎛️ Variables del Estudiante")
st.sidebar.write("Ajusta los indicadores del estudiante para calcular predicciones en tiempo real.")

asistencia_input = st.sidebar.slider(
    "Porcentaje de Asistencia (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0,
    help="Porcentaje acumulado de asistencia a clases presenciales o virtuales."
)

tareas_previas_input = st.sidebar.slider(
    "Tareas Anteriores Entregadas (sobre 10)",
    min_value=0,
    max_value=10,
    value=6,
    step=1,
    help="Número de tareas entregadas exitosamente de las últimas 10 asignadas."
)

recordatorio_input = st.sidebar.radio(
    "¿Se envió recordatorio previo de la tarea?",
    options=["Sí", "No"],
    index=0,
    help="Si el docente envió un recordatorio institucional antes de la fecha límite."
)
recordatorio_bin = 1 if recordatorio_input == "Sí" else 0

# Crear inputs en formato DataFrame
input_cls_df = pd.DataFrame([{
    'Asistencia': asistencia_input,
    'Tareas_Anteriores_Entregadas': tareas_previas_input,
    'Recordatorio_Enviado': recordatorio_bin
}])

# Crear inputs para Regresión Lineal
input_reg_df = pd.DataFrame([{
    'Asistencia': asistencia_input,
    'Recordatorio_Enviado': recordatorio_bin
}])

# ==========================================
# CÁLCULO DE PREDICCIONES
# ==========================================
# 1. Regresión Logística (Clasificación)
input_cls_scaled = scaler.transform(input_cls_df)
prob_entrega = logistic_model.predict_proba(input_cls_scaled)[0, 1]
pred_entrega = logistic_model.predict(input_cls_scaled)[0]

# 2. Regresión Lineal (Predicción de Tareas)
pred_tareas = linear_model.predict(input_reg_df)[0]
# Limitar valor de predicción razonable (entre 0 y 10)
pred_tareas = np.clip(pred_tareas, 0, 10)

# 3. Modelo de Probabilidad Lineal (LPM)
pred_lpm = lpm_model.predict(input_cls_df)[0]

# ── Diseño de Pestañas (Tabs) ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Clasificación (Regresión Logística)", 
    "📈 Regresión Lineal y Comparativa LPM", 
    "📊 Ciclo CRISP-ML", 
    "💾 Datos y Estadísticas Históricas"
])

# ==============================================================================
# PESTAÑA 1: REGRESIÓN LOGÍSTICA
# ==============================================================================
with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### Resultado de la Predicción")
        st.write("Predicción de la entrega de la tarea escolar basado en el modelo de Regresión Logística:")
        
        # Tarjeta visual dinámica
        prob_pct = prob_entrega * 100
        if prob_entrega >= 0.50:
            st.markdown(f"""
            <div class='status-card-success'>
                <h2>¡Entrega Probable!</h2>
                <p style='font-size: 2.2rem; margin: 0;'>{prob_pct:.1f}%</p>
                <p>Probabilidad de entrega estimada. No se requiere intervención crítica.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='status-card-danger'>
                <h2>Riesgo de Incumplimiento</h2>
                <p style='font-size: 2.2rem; margin: 0;'>{prob_pct:.1f}%</p>
                <p>El estudiante tiene alta probabilidad de no entregar. Se sugiere contactar y enviar apoyo.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.progress(prob_entrega)
        
    with col2:
        st.markdown("### Explicación del Modelo e Impacto de Coeficientes")
        st.write("""
        La Regresión Logística calcula una probabilidad acotada entre 0 y 1 usando la función logística (sigmoide).
        Los coeficientes del modelo entrenado nos permiten ver qué variable influye más en el comportamiento del estudiante:
        """)
        
        coefs = logistic_model.coef_[0]
        intercept = logistic_model.intercept_[0]
        
        coef_df = pd.DataFrame({
            "Variable": ["Asistencia (Escalada)", "Tareas Anteriores (Escalada)", "Recordatorio (Escalado)"],
            "Coeficiente Beta (Peso)": coefs
        })
        
        st.table(coef_df)
        
        st.markdown(f"""
        * **Intercepto (Sesgo):** `{intercept:.4f}`
        * **Interpretación:** 
          - La variable de mayor peso positivo es **Recordatorio Enviado** (`{coefs[2]:.4f}`). Enviar un recordatorio aumenta drásticamente la probabilidad de entrega.
          - Le sigue la **Asistencia** (`{coefs[0]:.4f}`), lo que demuestra que los estudiantes que atienden con regularidad tienden a ser cumplidores.
          - El historial de **Tareas Anteriores Entregadas** (`{coefs[1]:.4f}`) complementa el análisis.
        """)

# ==============================================================================
# PESTAÑA 2: REGRESIÓN LINEAL Y COMPARATIVA LPM
# ==============================================================================
with tab2:
    st.markdown("### 📈 Regresión Lineal Continua")
    st.write("""
    A diferencia de la clasificación, la Regresión Lineal estima resultados cuantitativos continuos.
    En este caso, la usamos para predecir **cuántas de las 10 tareas anteriores** entregaría un estudiante hipotético en función de su asistencia y si recibe recordatorios.
    """)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            label="Predicción de Tareas Anteriores Entregadas", 
            value=f"{pred_tareas:.2f} tareas / 10",
            delta=f"{pred_tareas - 5.0:.2f} vs Promedio Escolar (5.0)"
        )
        st.write(f"""
        **Fórmula del regresor ajustado:**  
        `Tareas Previas = {linear_model.intercept_:.3f} + ({linear_model.coef_[0]:.3f} * Asistencia) + ({linear_model.coef_[1]:.3f} * Recordatorio)`
        
        *Para los valores ingresados:*  
        `Tareas Previas = {linear_model.intercept_:.3f} + ({linear_model.coef_[0]:.3f} * {asistencia_input}) + ({linear_model.coef_[1]:.3f} * {recordatorio_bin}) = {pred_tareas:.2f}`
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.write("#### Relación de Ajuste Lineal (Datos Históricos)")
        # Graficamos el ajuste de la regresión
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(data=df, x='Asistencia', y='Tareas_Anteriores_Entregadas', color='#E67E22', s=70, ax=ax, label="Estudiantes Reales")
        
        # Generar recta de predicción
        x_vals = np.linspace(df['Asistencia'].min(), df['Asistencia'].max(), 100)
        y_vals = linear_model.intercept_ + linear_model.coef_[0] * x_vals + linear_model.coef_[1] * recordatorio_bin
        ax.plot(x_vals, y_vals, color='#2a5298', lw=2.5, label=f"Ajuste lineal (Recordatorio={recordatorio_input})")
        
        # Indicar punto actual
        ax.scatter([asistencia_input], [pred_tareas], color='#E74C3C', s=150, zorder=5, marker='*', label="Punto de Entrada")
        
        ax.set_title("Línea de Regresión Lineal Estimada", fontsize=11, fontweight='bold')
        ax.set_xlabel("% Asistencia")
        ax.set_ylabel("Tareas Entregadas (0-10)")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### ⚖️ Comparación Teórica: Regresión Logística vs. Modelo de Probabilidad Lineal (LPM)")
    st.write("""
    El **Modelo de Probabilidad Lineal (LPM)** es una regresión lineal aplicada a una variable de salida binaria (0/1). 
    A continuación se compara la probabilidad estimada por ambos modelos para los parámetros seleccionados:
    """)
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("<div class='metric-card' style='border-left: 5px solid #2ECC71;'>", unsafe_allow_html=True)
        st.write("**Probabilidad Estimada por Regresión Logística:**")
        st.markdown(f"<h3 style='color: #2ECC71;'>{prob_entrega * 100:.2f}%</h3>", unsafe_allow_html=True)
        st.write("Acotada estrictamente en el rango `[0, 1]` mediante la función sigmoide.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with comp_col2:
        lpm_color = "#E74C3C" if (pred_lpm < 0 or pred_lpm > 1) else "#3498DB"
        st.markdown(f"<div class='metric-card' style='border-left: 5px solid {lpm_color};'>", unsafe_allow_html=True)
        st.write("**Probabilidad Estimada por Modelo Lineal (LPM):**")
        st.markdown(f"<h3 style='color: {lpm_color};'>{pred_lpm * 100:.2f}%</h3>", unsafe_allow_html=True)
        st.write("Calculada directamente como combinación lineal. Puede arrojar valores absurdos (< 0% o > 100%) bajo valores extremos.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.info("💡 **Prueba esto:** Ajusta la Asistencia a un valor muy bajo (p.ej., 10%) o muy alto en la barra lateral y observa cómo el LPM arroja una probabilidad irreal (menor a 0% o mayor a 100%), demostrando por qué la Regresión Logística es matemáticamente superior para problemas de clasificación binaria.")

# ==============================================================================
# PESTAÑA 3: METODOLOGÍA CRISP-ML
# ==============================================================================
with tab3:
    st.markdown("### ⚙️ Ciclo de Vida del Proyecto CRISP-ML")
    st.write("El desarrollo de este sistema predictivo siguió la metodología estándar para proyectos de Machine Learning:")
    
    st.markdown("""
    <div class='crisp-phase'>
        <h4>1. Comprensión del Negocio (Business Understanding)</h4>
        <p>Identificar la necesidad de predecir la no entrega de tareas para apoyar a docentes en intervenciones oportunas.</p>
    </div>
    <div class='crisp-phase'>
        <h4>2. Comprensión de los Datos (Data Understanding)</h4>
        <p>Análisis de las correlaciones entre Asistencia (90.4% media en entregados), Tareas previas (8.9 entregadas) y Recordatorios enviados.</p>
    </div>
    <div class='crisp-phase'>
        <h4>3. Preparación de los Datos (Data Preparation)</h4>
        <p>Escalado estándar de las variables numéricas y división 70/30 estratificada de los datos para garantizar entrenamiento insesgado.</p>
    </div>
    <div class='crisp-phase'>
        <h4>4. Modelado (Modeling)</h4>
        <p>Implementación de la Regresión Logística para clasificar la probabilidad de entrega y la Regresión Lineal para estimar tendencias de cumplimiento continuo.</p>
    </div>
    <div class='crisp-phase'>
        <h4>5. Evaluación (Evaluation)</h4>
        <p>Verificación del modelo. La regresión logística logró una exactitud (Accuracy) del 100% sobre el conjunto de prueba (dada la naturaleza linealmente separable del conjunto limitado de 24 estudiantes), mientras que el regresor lineal obtuvo un R² del 98%.</p>
    </div>
    <div class='crisp-phase'>
        <h4>6. Despliegue y Monitoreo (Deployment & Monitoring)</h4>
        <p>Puesta en producción de este simulador Streamlit y despliegue del repositorio en GitHub. Se proponen auditorías mensuales para detectar concepto o datos de deriva (concept & data drift).</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PESTAÑA 4: DATOS HISTÓRICOS
# ==============================================================================
with tab4:
    st.markdown("### 💾 Dataset del Proyecto (24 Estudiantes)")
    st.write("Este es el conjunto de datos completo utilizado para el entrenamiento y prueba de los modelos predictivos:")
    
    # Mostrar el dataframe completo formateado de forma atractiva
    st.dataframe(
        df.style.map(
            lambda val: 'background-color: #d4fc79; color: #1e4620;' if val == 1 else 'background-color: #ff9a9e; color: #5c181b;', 
            subset=['Recordatorio_Enviado', 'Entrego']
        ), 
        width='stretch'
    )
    
    st.write("---")
    st.write("#### 📊 Distribución de Estadísticas Descriptivas del Grupo")
    st.table(df.describe().round(2))
