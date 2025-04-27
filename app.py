# ===============================
# 🔵 Chatbot - Calidad de Sueño en Streamlit Web
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# ======================================
# 1. 📌 Cargar el conjunto de datos desde GitHub
# ======================================

url = 'https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/Sleep_health_and_lifestyle_dataset.csv'

st.write("🔹 Cargando dataset...")
df = pd.read_csv(url)

feature_columns = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
X = df[feature_columns]
y = df['Quality of Sleep']

# ======================================
# 2. 📌 Preprocesamiento
# ======================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ======================================
# 3. 📌 Entrenamiento del modelo
# ======================================

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ======================================
# 4. 📌 Evaluación del modelo
# ======================================

st.subheader("📈 Evaluación del Modelo de Predicción")
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
st.write(f"🔵 Precisión promedio con validación cruzada: {np.mean(cv_scores):.2f}")

y_pred = model.predict(X_test)
st.write("🔵 Matriz de Confusión:")
st.text(confusion_matrix(y_test, y_pred))

st.write("🔵 Informe de Clasificación:")
st.text(classification_report(y_test, y_pred))

# ======================================
# 5. 📌 Chatbot de predicción interactiva
# ======================================

st.title("🤖 Asistente de Calidad del Sueño - TalentoTech 💤")

st.write("Completa los siguientes datos para obtener una predicción:")

# --- Inputs visuales en la web ---
age = st.number_input("📝 ¿Qué edad tienes?", min_value=0, max_value=120, step=1)
sleep_duration = st.number_input("🛏️ ¿Cuántas horas duermes por noche?", min_value=0.0, max_value=24.0, step=0.5)
physical_activity = st.number_input("🏃‍♂️ ¿Cuántos minutos de actividad física haces por semana?", min_value=0, max_value=2000, step=10)
stress_level = st.slider("😰 ¿Nivel de estrés de 1 a 10?", min_value=1, max_value=10)

# Botón para predecir
if st.button("Predecir Calidad de Sueño"):
    # Crear dataframe del usuario
    user_data = pd.DataFrame([[age, sleep_duration, physical_activity, stress_level]], columns=feature_columns)
    
    # Escalar datos
    user_data_scaled = scaler.transform(user_data)

    # Hacer predicción
    predicted_quality = model.predict(user_data_scaled)[0]

    st.success(f"🌟 Predicción de Calidad de Sueño: {predicted_quality}")

    # --- Consejos basados en la predicción ---
    st.subheader("💡 Consejos Personalizados:")
    if predicted_quality >= 8:
        st.info("¡Excelente! Tu calidad de sueño es alta. Mantén tus hábitos saludables. 😴✨")
    elif predicted_quality >= 6:
        st.warning("Tu sueño es aceptable, pero podrías mejorarlo un poco.")
        if sleep_duration < 7:
            st.write("- Intenta dormir al menos entre 7 y 9 horas.")
        if physical_activity < 150:
            st.write("- Aumenta tu actividad física semanal para mejorar el descanso.")
        if stress_level > 5:
            st.write("- Considera técnicas para reducir el estrés como meditación o ejercicio moderado.")
    else:
        st.error("Tu calidad de sueño parece baja. ¡Es momento de cuidar tu descanso!")
        if sleep_duration < 7:
            st.write("- Aumenta tus horas de sueño diario.")
        if physical_activity < 150:
            st.write("- Haz más actividad física para mejorar tu estado de ánimo y descanso.")
        if stress_level > 5:
            st.write("- Busca maneras de controlar el estrés: yoga, respiración, pausas activas.")


