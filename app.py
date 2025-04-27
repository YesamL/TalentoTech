# ===============================
# 🤖 Chatbot Real Mejorado - Calidad de Sueño - TalentoTech (corregido rerun)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# ======================================
# 1. 📌 Cargar datos y entrenar modelo
# ======================================

url = 'https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/Sleep_health_and_lifestyle_dataset.csv'

df = pd.read_csv(url)
feature_columns = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
X = df[feature_columns]
y = df['Quality of Sleep']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ======================================
# 2. 📌 Streamlit App en forma de Chatbot
# ======================================

st.title("🤖 Asistente Virtual de Calidad del Sueño - TalentoTech 💤")

# Inicializar estado de sesión
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.name = ""
    st.session_state.answers = []

# --- Flujo del chatbot ---
preguntas = [
    "¿Qué edad tienes? 🎂",
    "¿Cuántas horas duermes por noche? 🛏️",
    "¿Cuántos minutos de actividad física haces por semana? 🏃‍♂️",
    "En una escala del 1 al 10, ¿cuánto estrés sientes? 😰"
]

# Mostrar mensajes anteriores
for i in range(len(st.session_state.answers)):
    if i == 0 and st.session_state.name:
        st.chat_message("assistant").write(f"{st.session_state.name}, {preguntas[i]}")
    else:
        st.chat_message("assistant").write(preguntas[i])
    st.chat_message("user").write(st.session_state.answers[i])

# Flujo del chatbot
if st.session_state.step == 0:
    st.chat_message("assistant").write("¡Hola! 👋 ¿Cómo te llamas?")
    name_input = st.chat_input("Escribe tu nombre...")
    if name_input:
        st.session_state.name = name_input
        st.session_state.step += 1
elif st.session_state.step <= len(preguntas):
    pregunta_actual = preguntas[st.session_state.step - 1]
    user_input = st.chat_input(f"{st.session_state.name}, {pregunta_actual}")

    if user_input:
        st.session_state.answers.append(user_input)
        st.session_state.step += 1
else:
    # Hacer predicción
    try:
        edad = float(st.session_state.answers[0])
        horas_sueno = float(st.session_state.answers[1])
        actividad_fisica = float(st.session_state.answers[2])
        nivel_estres = float(st.session_state.answers[3])

        user_data = pd.DataFrame([[edad, horas_sueno, actividad_fisica, nivel_estres]], columns=feature_columns)
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)[0]

        st.chat_message("assistant").success(f"🌟 {st.session_state.name}, según mis cálculos, tu calidad de sueño es: **{prediction}**")

        if prediction >= 8:
            st.chat_message("assistant").info(f"¡Excelente {st.session_state.name}! Tu calidad de sueño es alta. 😴✨ ¡Sigue así!")
        elif prediction >= 6:
            st.chat_message("assistant").warning(f"{st.session_state.name}, tu sueño es aceptable, pero podrías mejorarlo un poco.")
            if horas_sueno < 7:
                st.chat_message("assistant").write("- Intenta dormir entre 7 y 9 horas por noche.")
            if actividad_fisica < 150:
                st.chat_message("assistant").write("- Aumenta tu actividad física semanal.")
            if nivel_estres > 5:
                st.chat_message("assistant").write("- Reduce tu nivel de estrés.")
        else:
            st.chat_message("assistant").error(f"{st.session_state.name}, tu calidad de sueño parece baja. ¡Debes actuar! 🌙")
            if horas_sueno < 7:
                st.chat_message("assistant").write("- Prioriza más horas de sueño.")
            if actividad_fisica < 150:
                st.chat_message("assistant").write("- Haz más ejercicio moderado.")
            if nivel_estres > 5:
                st.chat_message("assistant").write("- Busca técnicas para manejar el estrés.")
    except Exception as e:
        st.chat_message("assistant").error(f"❌ Error procesando tus datos: {e}")
