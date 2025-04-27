# ===============================
# 🤖 Chatbot Mejorado con Streamlit y Caching - Calidad de Sueño
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
# time es útil para pausas cortas en la interacción del bot
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
# Importaciones que no se usan en el frontend pero son parte del análisis original
# from sklearn.metrics import classification_report, confusion_matrix, cross_val_score

# ======================================
# 1. 📌 Cargar datos y entrenar modelo (Cacheado)
# ======================================
@st.cache_resource # Carga y entrena solo una vez
def load_data_train_model(data_url: str):
    """Carga los datos desde una URL, entrena el modelo y fitea el scaler."""
    try:
        df = pd.read_csv(data_url)
        feature_columns = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
        target_column = 'Quality of Sleep'

        # Asegurarse de que las columnas necesarias existan
        if not all(col in df.columns for col in feature_columns + [target_column]):
             missing = [col for col in feature_columns + [target_column] if col not in df.columns]
             st.error(f"Error: El archivo CSV no contiene las columnas necesarias. Faltan: {missing}")
             st.stop()

        X = df[feature_columns]
        y = df[target_column]

        # Escalado de características: fit en todos los datos originales
        scaler = MinMaxScaler()
        scaler.fit(X)

        # División y entrenamiento del modelo (solo necesitamos el modelo entrenado)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        st.success("Datos cargados y modelo entrenado. ¡Listo para chatear!") # Mensaje de confirmación

        return model, scaler, feature_columns

    except Exception as e:
        st.error(f"Ocurrió un error al cargar datos o entrenar el modelo: {e}")
        st.stop() # Detiene la aplicación en caso de error grave

# URL del dataset en GitHub
DATA_URL = 'https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/Sleep_health_and_lifestyle_dataset.csv'

# Cargar modelo, scaler y columnas usando la función cacheada
model, scaler, feature_columns = load_data_train_model(DATA_URL)

# --- Lógica de Consejos (Adaptada para mensajes de chat) ---
def get_advice_messages(predicted_score: float, user_inputs: dict) -> list[str]:
    """Genera una lista de mensajes de consejos basados en la predicción y inputs."""
    messages = []
    messages.append("Basado en mi predicción y tus datos:")

    # Acceder a los datos del usuario directamente desde el diccionario
    edad = user_inputs[feature_columns[0]]
    horas_sueno = user_inputs[feature_columns[1]]
    actividad_fisica = user_inputs[feature_columns[2]]
    nivel_estres = user_inputs[feature_columns[3]]

    if predicted_score >= 8:
        messages.append("✨ ¡Excelente! Tu calidad de sueño es alta. 😴✨")
        messages.append("✅ Continúa con tus hábitos saludables de sueño, actividad física y manejo del estrés.")
        messages.append("🛌 Mantener un horario de sueño regular es clave.")
    elif predicted_score >= 6:
        messages.append("👍 Tu calidad de sueño es aceptable, pero podrías mejorarlo un poco.")
        messages.append("Ideas para mejorar:")
        if horas_sueno < 7:
            messages.append(f"- Intenta dormir entre 7 y 9 horas por noche (actualmente ~{horas_sueno:.1f}h).")
        if actividad_fisica < 150: # Valor de referencia (ej: minutos moderados por semana)
             messages.append(f"- Aumenta tu actividad física semanal (actualmente ~{actividad_fisica:.0f} min/sem). Busca al menos 150 min moderados.")
        if nivel_estres > 5: # Valor de referencia
            messages.append(f"- Explora técnicas de manejo del estrés (nivel actual ~{nivel_estres:.1f}).")
        messages.append("✨ Establece una rutina relajante antes de acostarte.")
    else: # predicted_score < 6
        messages.append("⚠️ Tu calidad de sueño parece baja. ¡Es importante tomar medidas! 🌙")
        messages.append("Pasos recomendados:")
        if horas_sueno < 7:
            messages.append(f"- Prioriza aumentar tus horas de sueño (actualmente ~{horas_sueno:.1f}h). Intenta horarios regulares.")
        if actividad_fisica < 150:
             messages.append(f"- La actividad física regular es vital (actualmente ~{actividad_fisica:.0f} min/sem). Busca incorporar ejercicio diario.")
        if nivel_estres > 5:
            messages.append(f"- Trabaja en reducir tu estrés diario (nivel actual ~{nivel_estres:.1f}). Busca apoyo o estrategias efectivas.")
        messages.append("🏠 Revisa tu higiene del sueño: habitación oscura, silenciosa y fresca.")
        messages.append("🚫 Evita cafeína, alcohol y pantallas electrónicas antes de dormir.")
        messages.append("👩‍⚕️ Si los problemas persisten, considera consultar a un médico o especialista en sueño.")

    return messages


# ======================================
# 2. 📌 Streamlit App en forma de Chatbot - Lógica de Interacción
# ======================================

st.title("🤖 Asistente Virtual de Calidad del Sueño 💤")
st.caption("Impulsado por datos de TalentoTech")

# Inicializar estado de sesión (si aún no existen)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_state" not in st.session_state:
    # Estados: inicio, pidiendo_nombre, pidiendo_edad, pidiendo_sueno, ..., recolectando, completo, preguntando_reiniciar, fin
    st.session_state.chat_state = "inicio"
if "user_data_collector" not in st.session_state:
    st.session_state.user_data_collector = {}
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# --- Mostrar mensajes históricos ---
# Este bucle se ejecuta en cada rerun para mostrar toda la conversación
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Lógica Principal del Chatbot ---

# Lógica del primer mensaje y cambio de estado inicial
if st.session_state.chat_state == "inicio":
    initial_message = "¡Hola! 👋 Soy tu asistente personal para analizar y mejorar tu sueño. ¿Cómo te llamas?"
    with st.chat_message("assistant"):
        st.markdown(initial_message)
    st.session_state.messages.append({"role": "assistant", "content": initial_message})
    st.session_state.chat_state = "pidiendo_nombre" # Siguiente estado


# Usar st.chat_input para obtener la entrada del usuario
# Este widget solo aparece si no se llama a st.stop()
user_input = st.chat_input("Escribe aquí tu respuesta...")

if user_input:
    # Añadir la entrada del usuario a la historia inmediatamente
    st.session_state.messages.append({"role": "user", "content": user_input})

    # --- Procesar la entrada según el estado actual ---
    current_state = st.session_state.chat_state
    user_data = st.session_state.user_data_collector
    user_name = st.session_state.user_name
    next_state = current_state # Por defecto, el estado no cambia a menos que sea exitoso
    bot_response = None # Mensaje del bot a mostrar después de procesar input
    process_input_success = False # Bandera para saber si el input actual fue válido y procesado

    # Diccionario para mapear estados a preguntas y nombres de columnas
    questions = {
        "pidiendo_edad": (f"{user_name}, ¿qué edad tienes? 🎂 ({feature_columns[0]})", feature_columns[0]),
        "pidiendo_sueno": (f"{user_name}, ¿cuántas horas duermes normalmente por noche? 🛏️ ({feature_columns[1]})", feature_columns[1]),
        "pidiendo_actividad": (f"{user_name}, ¿cuántos minutos de actividad física haces por semana? 🏃‍♂️ ({feature_columns[2]})", feature_columns[2]),
        "pidiendo_estres": (f"{user_name}, en una escala del 1 al 10, ¿cuánto estrés sientes? 😰 ({feature_columns[3]})", feature_columns[3]),
    }

    # --- Lógica de Estados ---
    if current_state == "pidiendo_nombre":
        if len(user_input.strip()) > 0:
            user_name = user_input.strip().split()[0] # Tomar solo el primer nombre si hay varios
            st.session_state.user_name = user_name
            bot_response = f"¡Mucho gusto, {user_name}! 😄 Ahora sí, comencemos con las preguntas para tu predicción. ¿Qué edad tienes? 🎂 ({feature_columns[0]})"
            next_state = "pidiendo_edad"
            process_input_success = True
        else:
            bot_response = "¿No me dices tu nombre? No hay problema. ¿Empezamos con las preguntas? ¿Qué edad tienes? 🎂"
            st.session_state.user_name = "usuario" # Nombre por defecto
            next_state = "pidiendo_edad"
            process_input_success = True # Considerar como éxito para avanzar

    elif current_state in ["pidiendo_edad", "pidiendo_sueno", "pidiendo_actividad", "pidiendo_estres"]:
        pregunta, col_name = questions[current_state]
        try:
            value = float(user_input)
            # Validaciones básicas
            if col_name == feature_columns[0] and (value <= 0 or value > 120): # Edad
                 raise ValueError("La edad debe ser entre 1 y 120 años.")
            elif col_name == feature_columns[1] and (value <= 0 or value > 24): # Sueño
                 raise ValueError("La duración del sueño debe ser positiva (horas).")
            elif col_name == feature_columns[2] and value < 0: # Actividad
                 raise ValueError("La actividad física no puede ser negativa.")
            elif col_name == feature_columns[3] and (value < 1 or value > 10): # Estrés
                 raise ValueError("El nivel de estrés debe ser un número entre 1 y 10.")

            user_data[col_name] = value # Almacenar el valor validado
            process_input_success = True # Marcar como éxito

            # Determinar el siguiente estado y la siguiente pregunta
            current_index = feature_columns.index(col_name)
            if current_index < len(feature_columns) - 1:
                next_col_name = feature_columns[current_index + 1]
                # Encontrar el estado que corresponde a la siguiente columna
                next_state = [state for state, info in questions.items() if info[1] == next_col_name][0]
                next_question, _ = questions[next_state]
                bot_response = f"Entendido. {next_question}"
            else:
                # Todos los datos recolectados
                bot_response = f"¡Gracias, {user_name}! Ya tengo todos los datos necesarios. Permíteme un momento para hacer la predicción."
                next_state = "recolectando" # Estado intermedio antes de 'completo'

        except ValueError as e:
            # Error específico de valor o tipo
            bot_response = f"🤔 Ups. Parece que '{user_input}' no es una respuesta válida para la pregunta sobre {col_name.lower().replace(' level', '')}. {e} Por favor, intenta de nuevo con un número."
            process_input_success = False # Mantener estado actual

        except Exception as e:
            # Otros posibles errores
            bot_response = f"❌ Ocurrió un error inesperado al procesar tu respuesta sobre {col_name.lower()}: {e}"
            process_input_success = False # Mantener estado actual


    # --- Si llegamos al estado 'recolectando' después de un input exitoso, hacemos la predicción ---
    # Usamos un estado intermedio para asegurar que el último mensaje de confirmación se muestre ANTES de la predicción
    if current_state == "recolectando" and process_input_success:
        # Mostrar el último mensaje de confirmación si existe
        if bot_response:
             with st.chat_message("assistant"):
                 st.markdown(bot_response)
             st.session_state.messages.append({"role": "assistant", "content": bot_response})
             # Limpiar el bot_response para que no se duplique abajo
             bot_response = None
        # Importante: No cambiar next_state aún, la predicción lo hará

        with st.spinner("Analizando tus datos..."):
             time.sleep(1.5) # Simula tiempo de procesamiento

             # Crear DataFrame para predicción desde los datos recolectados
             user_data_df = pd.DataFrame([user_data], columns=feature_columns)

             # Escalar datos del usuario
             # Usamos transform() con el scaler que fue fit() en load_data_train_model
             user_data_scaled = scaler.transform(user_data_df)

             # Realizar la predicción
             predicted_quality = model.predict(user_data_scaled)[0]

             # Mostrar predicción
             prediction_message = f"🌟 {user_name}, según mis cálculos, tu calidad de sueño es: **{int(predicted_quality)}**"
             with st.chat_message("assistant"):
                 st.markdown(prediction_message)
             st.session_state.messages.append({"role": "assistant", "content": prediction_message})

             # Obtener y mostrar consejos
             advice_messages = get_advice_messages(predicted_quality, user_data_df)
             with st.chat_message("assistant"):
                 st.markdown("💡 Aquí tienes algunos consejos para ti:") # Encabezado de consejos
             st.session_state.messages.append({"role": "assistant", "content": "💡 Aquí tienes algunos consejos para ti:"})
             # Mostrar cada consejo como un mensaje separado para mejor lectura
             for i, advice_msg in enumerate(advice_messages):
                 # Opcional: añadir un pequeño retraso entre consejos
                 # if i > 0: time.sleep(0.2) # Pausa corta
                 with st.chat_message("assistant"):
                     st.markdown(advice_msg)
                 st.session_state.messages.append({"role": "assistant", "content": advice_msg})

             # Preguntar si quiere otra predicción o terminar
             final_message = f"¿Te gustaría hacer otra predicción ('empezar') o prefieres terminar ('salir')?"
             with st.chat_message("assistant"):
                  st.markdown(final_message)
             st.session_state.messages.append({"role": "assistant", "content": final_message})

             # Resetear el estado y datos para una posible nueva conversación
             next_state = "preguntando_reiniciar"
             st.session_state.user_data_collector = {} # Limpiar datos recolectados


    elif current_state == "preguntando_reiniciar":
         if "empezar" in user_input.lower() or "si" in user_input.lower() or "otra" in user_input.lower():
             bot_response = f"¡Excelente! Empecemos de nuevo. ¿Qué edad tienes, {user_name}? 🎂 ({feature_columns[0]})"
             next_state = "pidiendo_edad"
             st.session_state.user_data_collector = {} # Asegurarse de que esté limpio
             st.session_state.messages = [] # Opcional: Limpiar la conversación anterior al reiniciar
             st.session_state.initial_message_shown = True # Mostrar el primer mensaje de pregunta directamente
         elif "salir" in user_input.lower() or "no" in user_input.lower() or "terminar" in user_input.lower():
             bot_response = f"De acuerdo, {user_name}. ¡Que tengas un excelente descanso y un buen día! Adiós. 👋"
             next_state = "fin"
         else:
             bot_response = f"No entendí. ¿Quieres hacer otra predicción ('empezar') o terminar ('salir'), {user_name}?"
             # next_state remains 'preguntando_reiniciar'


    # --- Actualizar el estado y datos si el procesamiento del input fue exitoso ---
    # Excepción: El estado 'recolectando' maneja su propia transición
    if process_input_success and current_state != "recolectando":
         st.session_state.chat_state = next_state
         st.session_state.user_data_collector = user_data # Guardar datos actualizados
    # Si el estado actual *era* 'recolectando' y tuvo éxito, ya se cambió el estado a 'preguntando_reiniciar' dentro de ese bloque

    # --- Mostrar la respuesta del bot generada (si existe) ---
    # Excluimos el mensaje del estado 'recolectando' que se muestra dentro de su bloque
    if bot_response and current_state != "recolectando":
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # --- Manejar el estado final ---
    if st.session_state.chat_state == "fin":
        st.balloons() # Opcional: añadir algo visual al finalizar
        st.stop() # Detiene la ejecución del script Streamlit aquí.