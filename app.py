# ===============================================
# 🤖 Asistente Virtual de Calidad del Sueño - Streamlit Chatbot (versión corregida)
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

# --- Configuración de la página ---
st.set_page_config(page_title="Asistente de Calidad del Sueño", layout="centered")
st.title("🤖 Asistente Virtual de Calidad del Sueño 💤")
st.caption("Impulsado por datos de TalentoTech")

# ======================================
# 1. 📌 Cargar y entrenar el modelo
# ======================================
@st.cache_resource
def load_data_train_model(data_url: str):
    df = pd.read_csv(data_url)
    feature_columns = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
    target_column = 'Quality of Sleep'

    X = df[feature_columns]
    y = df[target_column]

    scaler = MinMaxScaler()
    scaler.fit(X)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, feature_columns

DATA_URL = 'https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/Sleep_health_and_lifestyle_dataset.csv'
model, scaler, feature_columns = load_data_train_model(DATA_URL)

# ======================================
# 2. 📌 Lógica de Consejos
# ======================================
def get_advice_messages(predicted_score: float, user_inputs: dict) -> list[str]:
    messages = []
    edad = user_inputs['Age']
    horas_sueno = user_inputs['Sleep Duration']
    actividad_fisica = user_inputs['Physical Activity Level']
    nivel_estres = user_inputs['Stress Level']

    messages.append("Basado en mi predicción y tus datos:")

    if predicted_score >= 8:
        messages.append("✨ ¡Excelente! Tu calidad de sueño es alta. Sigue así. 😴")
    elif predicted_score >= 6:
        messages.append("👍 Tu sueño es aceptable, pero podrías mejorarlo un poco.")
        if horas_sueno < 7:
            messages.append("- Duerme entre 7 y 9 horas por noche.")
        if actividad_fisica < 150:
            messages.append("- Aumenta tu actividad física semanal (al menos 150 minutos).")
        if nivel_estres > 5:
            messages.append("- Trabaja en técnicas de manejo de estrés.")
    else:
        messages.append("⚠️ Tu calidad de sueño parece baja. Es importante actuar.")
        if horas_sueno < 7:
            messages.append("- Aumenta tus horas de sueño.")
        if actividad_fisica < 150:
            messages.append("- Haz más actividad física regularmente.")
        if nivel_estres > 5:
            messages.append("- Busca estrategias efectivas para controlar el estrés.")
    return messages

# ======================================
# 3. 📌 Chatbot - Flujo de Conversación
# ======================================

# --- Estado de sesión ---
if "chat_state" not in st.session_state:
    st.session_state.chat_state = "inicio"
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# --- Mostrar mensajes anteriores ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Flujo de conversación ---
def preguntar(pregunta):
    with st.chat_message("assistant"):
        st.markdown(pregunta)
    st.session_state.messages.append({"role": "assistant", "content": pregunta})

# Inicio de la conversación
if st.session_state.chat_state == "inicio":
    preguntar("¡Hola! 👋 Soy tu asistente para mejorar tu calidad de sueño. ¿Cómo te llamas?")
    st.session_state.chat_state = "pidiendo_nombre"

# Entrada del usuario
user_input = st.chat_input("Escribe aquí tu respuesta...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.chat_state == "pidiendo_nombre":
        st.session_state.user_name = user_input.split()[0].capitalize()
        preguntar(f"¡Mucho gusto, {st.session_state.user_name}! ¿Qué edad tienes? 🎂")
        st.session_state.chat_state = "pidiendo_edad"

    elif st.session_state.chat_state == "pidiendo_edad":
        try:
            edad = float(user_input)
            if not (1 <= edad <= 120):
                raise ValueError()
            st.session_state.user_data["Age"] = edad
            preguntar(f"{st.session_state.user_name}, ¿cuántas horas duermes por noche? 🛏️")
            st.session_state.chat_state = "pidiendo_sueno"
        except ValueError:
            preguntar("❌ Por favor ingresa una edad válida entre 1 y 120 años.")

    elif st.session_state.chat_state == "pidiendo_sueno":
        try:
            horas = float(user_input)
            if not (0 < horas <= 24):
                raise ValueError()
            st.session_state.user_data["Sleep Duration"] = horas
            preguntar(f"{st.session_state.user_name}, ¿cuántos minutos de actividad física haces por semana? 🏃‍♂️")
            st.session_state.chat_state = "pidiendo_actividad"
        except ValueError:
            preguntar("❌ Por favor ingresa una cantidad de horas de sueño válida (0-24).")

    elif st.session_state.chat_state == "pidiendo_actividad":
        try:
            actividad = float(user_input)
            if actividad < 0:
                raise ValueError()
            st.session_state.user_data["Physical Activity Level"] = actividad
            preguntar(f"{st.session_state.user_name}, en una escala del 1 al 10, ¿qué nivel de estrés sientes? 😰")
            st.session_state.chat_state = "pidiendo_estres"
        except ValueError:
            preguntar("❌ Por favor ingresa minutos de actividad física válidos (positivo).")

    elif st.session_state.chat_state == "pidiendo_estres":
        try:
            estres = float(user_input)
            if not (1 <= estres <= 10):
                raise ValueError()
            st.session_state.user_data["Stress Level"] = estres
            st.session_state.chat_state = "recolectando"

            # ✅ Forzar el flujo a predicción directamente después de completar datos
            st.experimental_rerun()
        except ValueError:
            preguntar("❌ Por favor ingresa un nivel de estrés entre 1 y 10.")

    elif st.session_state.chat_state == "preguntando_reiniciar":
        if user_input.lower() in ["empezar", "sí", "si"]:
            st.session_state.chat_state = "pidiendo_edad"
            st.session_state.user_data = {}
            st.session_state.messages = []
            st.experimental_rerun()
        elif user_input.lower() in ["salir", "no", "terminar"]:
            preguntar("¡Gracias por usar el asistente! Que tengas un excelente descanso. 🌙")
            st.balloons()
            st.stop()
        else:
            preguntar("Por favor escribe 'empezar' para una nueva predicción o 'salir' para terminar.")

# --- Si ya recolectó todos los datos, hacer predicción ---
if st.session_state.chat_state == "recolectando":
    with st.spinner("🔎 Analizando tus datos..."):
        time.sleep(2)
        user_data_df = pd.DataFrame([st.session_state.user_data], columns=feature_columns)
        user_data_scaled = scaler.transform(user_data_df)
        prediction = model.predict(user_data_scaled)[0]

        prediction_message = f"🌟 {st.session_state.user_name}, según mis cálculos, tu calidad de sueño es: **{int(prediction)}**"
        with st.chat_message("assistant"):
            st.markdown(prediction_message)
        st.session_state.messages.append({"role": "assistant", "content": prediction_message})

        advice_messages = get_advice_messages(prediction, st.session_state.user_data)
        for advice in advice_messages:
            with st.chat_message("assistant"):
                st.markdown(advice)
            st.session_state.messages.append({"role": "assistant", "content": advice})

        final_message = f"¿Te gustaría hacer otra predicción ('empezar') o prefieres terminar ('salir'), {st.session_state.user_name}?"
        with st.chat_message("assistant"):
            st.markdown(final_message)
        st.session_state.messages.append({"role": "assistant", "content": final_message})

        st.session_state.chat_state = "preguntando_reiniciar"
