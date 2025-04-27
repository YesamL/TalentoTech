# ===============================================
# 🤖 Asistente Virtual de Calidad del Sueño Mara y Samir 💤
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

# --- Configuración de la página ---
st.set_page_config(page_title="Asistente de Calidad del Sueño", layout="centered")
st.title("🤖 Asistente Virtual de Calidad del Sueño Mara y Samir 💤")
st.caption("Impulsado por datos de Mara y Samir")

# ======================================
# 1. 📌 Cargar, limpiar (winsorization) y entrenar el modelo
# ======================================
@st.cache_resource
def load_model_and_scaler(url):
    df = pd.read_csv(url)
    features = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
    target = 'Quality of Sleep'

    # --- Limpieza de datos ---
    df = df.dropna(subset=features + [target])       # eliminar nulos
    df = df.drop_duplicates()                        # eliminar duplicados
    for col in features:  # winsorización al 5-95%
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = df[col].clip(lower, upper)
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=features)

    X = df[features]
    y = df[target]
    scaler = MinMaxScaler().fit(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    cv_scores = cross_val_score(model, X, y, cv=5)
    return model, scaler, features, train_acc, cv_scores

DATA_URL = (
    "https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/"
    "Sleep_health_and_lifestyle_dataset.csv"
)
with st.spinner("🔄 Cargando y limpiando datos..."):
    model, scaler, feature_columns, train_acc, cv_scores = load_model_and_scaler(DATA_URL)
    st.success("✅ Datos limpios y modelo entrenado.")

# ======================================
# 2. 📌 Generador de consejos mejorado
# ======================================
def get_advice(score, data):
    msgs = ["Basado en mi predicción y tus datos:"]
    h, a, s = data['Sleep Duration'], data['Physical Activity Level'], data['Stress Level']
    if score >= 8:
        msgs.append("✨ ¡Excelente! Tu calidad de sueño es alta. Sigue así. 😴")
        if s > 5:
            msgs.append("⚠️ Aunque duermes bien, tu nivel de estrés es alto; trabaja en técnicas de relajación para mejorar tu descanso.")
    elif score >= 6:
        msgs.append("👍 Tu sueño es aceptable, pero podrías mejorarlo.")
        if h < 7: msgs.append(f"- Duerme entre 7 y 9 horas (hoy {h:.1f}h).")
        if a < 150: msgs.append(f"- Al menos 150 min de actividad física (hoy {a:.0f} min).")
        if s > 5: msgs.append(f"- Práctica de relajación; estrés {s:.1f}.")
    else:
        msgs.append("⚠️ Tu calidad de sueño parece baja. Es importante actuar.")
        if h < 7: msgs.append(f"- Aumenta tus horas de sueño (solo {h:.1f}h).")
        if a < 150: msgs.append(f"- Incrementa actividad física (hoy {a:.0f} min).")
        if s > 5: msgs.append(f"- Reduce el estrés (nivel {s:.1f}).")
    msgs.append("💡 Mantén hidratación, pausas activas y meditación aun con buen sueño.")
    return msgs

# ======================================
# 3. 📌 Estado de sesión y chat histórico
# ======================================
if "state" not in st.session_state:
    st.session_state.state = "inicio"
    st.session_state.history = []
    st.session_state.user_data = {}
    st.session_state.name = ""
for msg in st.session_state.history:
    with st.chat_message(msg["role"]): st.markdown(msg["text"])

# ======================================
# 4. 📌 Función de bot
# ======================================
def bot_say(text):
    with st.chat_message("assistant"): st.markdown(text)
    st.session_state.history.append({"role":"assistant","text":text})

# ======================================
# 5. 📌 Input de usuario
# ======================================
user_input = st.chat_input("Escribe aquí…")
if user_input:
    with st.chat_message("user"): st.markdown(user_input)
    st.session_state.history.append({"role":"user","text":user_input})
    st.session_state.last = user_input.strip().lower()

# ======================================
# 6. 📌 Máquina de estados
# ======================================
state = st.session_state.state
greetings = ["hola","buenos","buenas","hey","hi","saludos"]

if state == "inicio":
    bot_say("¡Hola! 👋 ¿Cómo estás hoy? 😊")
    st.session_state.state = "saludo"
elif state == "saludo" and "last" in st.session_state:
    resp = st.session_state.last
    if any(w in resp for w in ["bien","genial","excelente"]):
        bot_say("¡Me alegra oír eso! 🎉 Vamos a medir tu calidad de sueño.")
    elif any(g in resp for g in greetings):
        bot_say("¡Hola de nuevo! 🤗 Gracias por saludar.")
    else:
        bot_say("Lo siento que no estés al 100%. Vamos a revisar tu sueño para mejorar tu ánimo.")
    bot_say("¿Cómo te llamas?")
    st.session_state.state = "pidiendo_nombre"
elif state == "pidiendo_nombre" and "last" in st.session_state:
    nm = st.session_state.last.split()[0].capitalize()
    st.session_state.name = nm
    bot_say(f"Encantado, {nm}! Precisión entrenamiento: {train_acc:.2%}, CV media: {cv_scores.mean():.2%}, desv: {cv_scores.std():.2%}.")
    bot_say(f"{nm}, ¿qué edad tienes? 🎂")
    st.session_state.state = "pidiendo_edad"
elif state == "pidiendo_edad" and "last" in st.session_state:
    try:
        v = float(st.session_state.last); assert 1<=v<=120
        st.session_state.user_data['Age']=v
        bot_say(f"{st.session_state.name}, ¿horas de sueño? 🛏️")
        st.session_state.state = "pidiendo_sueno"
    except:
        bot_say("❌ Edad inválida (1–120). Intenta de nuevo.")
elif state == "pidiendo_sueno" and "last" in st.session_state:
    try:
        v=float(st.session_state.last); assert 0<v<=24
        st.session_state.user_data['Sleep Duration']=v
        bot_say(f"{st.session_state.name}, ¿minutos de actividad semanal? 🏃‍♂️")
        st.session_state.state="pidiendo_actividad"
    except:
        bot_say("❌ Horas inválidas (0–24). Intenta de nuevo.")
elif state == "pidiendo_actividad" and "last" in st.session_state:
    try:
        v=float(st.session_state.last); assert v>=0
        st.session_state.user_data['Physical Activity Level']=v
        bot_say(f"{st.session_state.name}, nivel de estrés 1-10? 😰")
        st.session_state.state="pidiendo_estres"
    except:
        bot_say("❌ Valor inválido. Intenta de nuevo.")
elif state=="pidiendo_estres" and "last" in st.session_state:
    try:
        v=float(st.session_state.last); assert 1<=v<=10
        st.session_state.user_data['Stress Level']=v
        st.session_state.state="recolectando"
    except:
        bot_say("❌ Nivel inválido (1–10). Intenta de nuevo.")

# ======================================
# 7. 📌 Predicción y consejos
# ======================================
if st.session_state.state=="recolectando":
    with st.spinner("🔎 Analizando..."):
        time.sleep(1.5)
        df_in=pd.DataFrame([st.session_state.user_data],columns=feature_columns)
        scaled=scaler.transform(df_in)
        pred=int(model.predict(scaled)[0])
    lbl="Excelente" if pred>=8 else "Aceptable" if pred>=6 else "Baja"
    bot_say(f"🌟 {st.session_state.name}, tu calidad de sueño es **{pred}** ({lbl}).")
    for m in get_advice(pred,st.session_state.user_data): bot_say(m)
    bot_say("💡 Aun con buen sueño, mantén hidratación, pausas activas y meditación.")
    bot_say("¿Otra predicción? 'empezar' o 'salir'.")
    st.session_state.state="preguntando_reiniciar"

# ======================================
# 8. 📌 Reinicio o fin
# ======================================
if state=="preguntando_reiniciar" and "last" in st.session_state:
    ans=st.session_state.last
    if any(k in ans for k in ["empezar","si"]):
        st.session_state.user_data={}
        st.session_state.history=[]
        st.session_state.state="pidiendo_edad"
        bot_say(f"¡Empezamos de nuevo, {st.session_state.name}! ¿Qué edad tienes? 🎂")
    elif any(k in ans for k in ["salir","no"]):
        bot_say(f"¡Gracias, {st.session_state.name}! Que descanses. 🌙")
        st.balloons()
        st.stop()
    else:
        bot_say("Escribe 'empezar' o 'salir'.")
