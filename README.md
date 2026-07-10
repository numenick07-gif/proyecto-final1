# 🎓 Plataforma Predictiva de Entrega de Tareas Escolares

Este proyecto final integrador, desarrollado para la asignatura **Fundamentos de IA y Aprendizaje Automático** en la **Maestría en Inteligencia Artificial Aplicada a la Educación e Investigación (UFHEC)**, implementa y despliega un sistema predictivo supervisado basado en la metodología estándar **CRISP-ML**.

El sistema predice si un estudiante entregará o no una tarea escolar basándose en su nivel de asistencia general, historial de cumplimiento y si se le envió un recordatorio institucional previo a la fecha límite. Además, incorpora la estimación de tendencias continuas mediante regresión lineal y el despliegue del proyecto con **Streamlit** y una **Landing Page** inmersiva de Realidad Aumentada (RA).

---

## 📁 Estructura del Proyecto

El repositorio está estructurado de la siguiente forma:

- **`Prediccion_Entrega_Tarea_Colab (1).ipynb`**: Cuaderno Jupyter estructurado en base a las 6 fases de la metodología CRISP-ML. Contiene el desarrollo de los modelos de Regresión Logística, Regresión Lineal y el Modelo de Probabilidad Lineal (LPM), además del Análisis Exploratorio de Datos (EDA).
- **`app.py`**: Aplicación interactiva web desarrollada en Streamlit que funciona como un simulador en tiempo real para predecir el riesgo académico y mostrar explicabilidad.
- **`index.html`**: Landing Page premium desarrollada en HTML y CSS que explica el ciclo de vida del proyecto bajo CRISP-ML y expone el impacto visual de la Realidad Aumentada en el aula.
- **`train_models.py`**: Script de entrenamiento y persistencia que genera y guarda los modelos entrenados en archivos pickle (`.pkl`).
- **`assets/`**: Directorio que contiene las imágenes de Realidad Aumentada generadas por Inteligencia Artificial para la Landing Page.
- **`requirements.txt`**: Librerías y dependencias necesarias para la ejecución local de la aplicación y el cuaderno.
- **`.gitignore`**: Archivo de configuración para omitir archivos temporales y archivos binarios pesados en el control de versiones.

---

## 🛠️ Instalación y Configuración Local

Sigue los siguientes pasos para ejecutar el proyecto en tu máquina local:

### 1. Clonar el repositorio (o descargar los archivos)
```bash
git clone <url_de_tu_repositorio_de_github>
cd "proyecto final"
```

### 2. Crear y activar un entorno virtual de Python (Opcional, pero recomendado)
En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar las dependencias
Instala todas las dependencias necesarias que se especifican en el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Instrucciones de Ejecución

### 1. Entrenar y Generar los Modelos
El proyecto está diseñado para funcionar de inmediato, pero si deseas volver a ajustar los modelos o verificar el script de entrenamiento, ejecuta:
```bash
python train_models.py
```
Este comando generará los archivos serializados `.pkl` en el directorio raíz.

### 2. Ejecutar la aplicación interactiva de Streamlit
Para inicializar el servidor local y abrir el simulador interactivo en tu navegador, ejecuta:
```bash
streamlit run app.py
```

### 3. Visualizar la Landing Page
Para explorar la Landing Page interactiva, simplemente abre el archivo `index.html` en cualquier navegador web moderno. Puedes hacerlo dando doble clic sobre el archivo o arrastrándolo a la ventana del navegador.

---

## ⚙️ Resumen Metodológico: CRISP-ML

1. **Comprensión del Negocio:** Definición de objetivos escolares para reducir el rezago en la entrega de tareas mediante la toma de decisiones pedagógicas basada en datos.
2. **Comprensión de los Datos:** Análisis exploratorio (EDA) de la relación entre asistencia, recordatorios e historial de cumplimiento de 24 estudiantes simulados.
3. **Preparación de los Datos:** Escalado estándar (`StandardScaler`) y particionado de datos (70/30) con muestreo estratificado.
4. **Modelado:** Ajuste de Regresión Logística (clasificación binaria) y Regresión Lineal (tendencia continua y LPM).
5. **Evaluación:** Validación cruzada en el set de prueba obteniendo una exactitud del 100% (para Regresión Logística) y un $R^2$ de 98.5% (para Regresión Lineal).
6. **Despliegue y Monitoreo:** Puesta en producción con Streamlit y monitoreo recomendado mensual de la deriva de concepto (concept drift) y de datos (data drift).

---

## 👥 Créditos del Proyecto

* **Estudiante:** Liskeidy Rosario Pérez  
* **Asignatura:** Fundamentos de IA y Aprendizaje Automático  
* **Institución:** Universidad Federico Henríquez y Carvajal (UFHEC)  
* **Asesor:** Dr. Darwin Muñoz, Ph.D
