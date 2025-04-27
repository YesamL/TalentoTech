# ===============================================
# 🤖 Asistente Virtual de Calidad del Sueño - Streamlit Chatbot
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import time # Para simular que el bot 'piensa'
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
# Importaciones que no se usan en el frontend pero son parte del análisis original
# from sklearn.metrics import classification_report, confusion_matrix, cross_val_score

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Asistente de Calidad del Sueño", layout="centered")

st.title("🤖 Asistente Virtual de Calidad del Sueño 💤")
st.caption("Impulsado por datos de TalentoTech")

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

        # st.success("Datos cargados y modelo entrenado. ¡Listo para chatear!") # Mensaje de confirmación en la UI si deseas

        return model, scaler, feature_columns

    except Exception as e:
        st.error(f"Ocurrió un error al cargar datos o entrenar el modelo: {e}")
        st.stop() # Detiene la aplicación en caso de error grave

# URL del dataset en GitHub
DATA_URL = 'https://raw.githubusercontent.com/YesamL/TalentoTech/main/data/Sleep_health_and_lifestyle_dataset.csv'

# Cargar modelo, scaler y columnas usando la función cacheada
# Estas variables estarán disponibles en todo el script después de esta línea
model, scaler, feature_columns = load_data_train_model(DATA_URL)

# --- Lógica de Consejos (Adaptada para devolver mensajes de chat) ---
def get_advice_messages(predicted_score: float, user_inputs: dict) -> list[str]:
    """Genera una lista de mensajes de consejos basados en la predicción y inputs."""
    messages = []

    # Acceder a los datos del usuario directamente desde el diccionario
    # Asegurarse de que todas las claves existan antes de acceder
    try:
        edad = user_inputs[feature_columns[0]]
        horas_sueno = user_inputs[feature_columns[1]]
        actividad_fisica = user_inputs[feature_columns[2]]
        nivel_estres = user_inputs[feature_columns[3]]
    except KeyError as e:
        # Esto no debería pasar si la lógica de recolección es correcta, pero es un fallback
        messages.append(f"Error interno al obtener datos para consejos: falta la clave {e}. No se pueden dar consejos detallados.")
        return messages # Devolver solo el mensaje de error

    messages.append("Basado en mi predicción y tus datos:")

    if predicted_score >= 8:
        messages.append("✨ ¡Excelente! Tu calidad de sueño es alta. 😴✨")
        messages.append("✅ Continúa con tus hábitos saludables de sueño, actividad física y manejo del estrés.")
        messages.append("🛌 Mantener un horario de sueño regular es clave.")
    elif predicted_score >= 6:
        messages.append("👍 Tu calidad de sueño es aceptable, pero podrías mejorarlo un poco.")
        messages.append("Ideas para mejorar:")
        if horas_sueno < 7: # Valor de referencia
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

# Inicializar estado de sesión (si aún no existen)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_state" not in st.session_state:
    # Estados: inicio, pidiendo_nombre, pidiendo_edad, pidiendo_sueno, ..., recolectando, preguntando_reiniciar, fin
    st.session_state.chat_state = "inicio"
if "user_data_collector" not in st.session_state:
    st.session_state.user_data_collector = {}
if "user_name" not in st.session_state:
    st.session_state.user_name = ""


# --- Mostrar mensajes históricos ---
# Este bucle se ejecuta en cada rerun para mostrar toda la conversación guardada
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
# Este widget solo aparece si no se llama a st.stop() y si no hay un user_input pendiente de procesar
user_input = st.chat_input("Escribe aquí tu respuesta...")

# Solo procesar el input si hay uno
if user_input:
    # Añadir la entrada del usuario a la historia inmediatamente
    st.session_state.messages.append({"role": "user", "content": user_input})

    # --- Procesar la entrada según el estado actual ---
    current_state = st.session_state.chat_state
    user_data = st.session_state.user_data_collector
    user_name = st.session_state.user_name # Obtener nombre actual
    next_state = current_state # Por defecto, el estado no cambia a menos que el input sea válido
    bot_response = None # Mensaje del bot a mostrar después de procesar input (si no es un error)
    input_handled = False # Bandera para saber si el input actual fue procesado (válido o inválido con mensaje)

    # Diccionario para mapear estados a preguntas y nombres de columnas
    # Se construye aquí para usar el nombre de usuario si ya se conoce
    questions = {
        "pidiendo_edad": (f"¿Qué edad tienes{', ' + user_name if user_name else ''}? 🎂 ({feature_columns[0]})", feature_columns[0]),
        "pidiendo_sueno": (f"¿Cuántas horas duermes normalmente por noche{', ' + user_name if user_name else ''}? 🛏️ ({feature_columns[1]})", feature_columns[1]),
        "pidiendo_actividad": (f"¿Cuántos minutos de actividad física haces por semana, aproximadamente{', ' + user_name if user_name else ''}? 🏃‍♂️ ({feature_columns[2]})", feature_columns[2]),
        "pidiendo_estres": (f"En una escala del 1 al 10, ¿cuánto estrés sientes{', ' + user_name if user_name else ''}? 😰 ({feature_columns[3]})", feature_columns[3]),
    }


    # --- Lógica de Estados ---
    if current_state == "pidiendo_nombre":
        if len(user_input.strip()) > 0:
            user_name = user_input.strip().split()[0].capitalize() # Tomar solo el primer nombre y capitalizar
            st.session_state.user_name = user_name
            bot_response = f"¡Mucho gusto, {user_name}! 😄 Ahora sí, comencemos con las preguntas para tu predicción. {questions['pidiendo_edad'][0]}"
            next_state = "pidiendo_edad"
            input_handled = True # Indica que el input fue procesado y se puede avanzar
        else:
            bot_response = "¿No me dices tu nombre? No hay problema. Puedes llamarme Bot. 😉 Empecemos con las preguntas. {questions['pidiendo_edad'][0]}"
            st.session_state.user_name = "usuario" # Nombre por defecto
            next_state = "pidiendo_edad"
            input_handled = True # Considerar como manejado para avanzar


    elif current_state in ["pidiendo_edad", "pidiendo_sueno", "pidiendo_actividad", "pidiendo_estres"]:
        pregunta, col_name = questions[current_state]
        try:
            value = float(user_input)
            # Validaciones básicas
            if col_name == feature_columns[0] and (value <= 0 or value > 120): # Edad
                 raise ValueError("La edad debe ser entre 1 y 120 años.")
            elif col_name == feature_columns[1] and (value <= 0 or value > 24): # Duración Sueño
                 raise ValueError("La duración del sueño debe ser positiva (horas) y razonable.")
            elif col_name == feature_columns[2] and value < 0: # Actividad
                 raise ValueError("La actividad física no puede ser negativa.")
            elif col_name == feature_columns[3] and (value < 1 or value > 10): # Estrés
                 raise ValueError("El nivel de estrés debe ser un número entre 1 y 10.")

            # Si la validación pasa:
            user_data[col_name] = value # Almacenar el valor validado en el diccionario
            input_handled = True # Marcar como éxito

            # Determinar el siguiente estado y la siguiente pregunta
            current_index = feature_columns.index(col_name)
            if current_index < len(feature_columns) - 1:
                next_col_name = feature_columns[current_index + 1]
                 # Encontrar el estado que corresponde a la siguiente columna.
                 # Esto busca en el diccionario 'questions' el estado que tiene 'next_col_name'
                next_state_key = [state for state, info in questions.items() if info[1] == next_col_name]
                if next_state_key:
                    next_state = next_state_key[0]
                    next_question, _ = questions[next_state] # Obtener el texto de la siguiente pregunta
                    bot_response = f"Entendido. {next_question}"
                else:
                    # Esto no debería pasar si questions está bien definido
                     raise ValueError(f"Error interno: No se encontró el siguiente estado para {next_col_name}")
            else:
                # Todos los datos recolectados
                # Aquí ya tenemos todos los datos, cambiamos al estado de predicción
                bot_response = f"¡Gracias, {user_name}! Ya tengo todos los datos necesarios. Permíteme un momento para hacer la predicción."
                next_state = "recolectando" # Estado intermedio antes de mostrar resultados


        except ValueError as e:
            # Error específico de valor o tipo
            bot_response = f"🤔 Ups. Parece que '{user_input}' no es una respuesta válida. {e} Por favor, intenta de nuevo con un número."
            # next_state sigue siendo el current_state
            input_handled = True # El input fue manejado (con un error)

        except Exception as e:
            # Otros posibles errores durante el procesamiento del input
            bot_response = f"❌ Ocurrió un error inesperado al procesar tu respuesta: {e}"
            # next_state sigue siendo el current_state
            input_handled = True # El input fue manejado (con un error)

    # --- Si llegamos al estado 'recolectando' (después de un input exitoso de estrés) ---
    # Este bloque se activa en el rerun DESPUÉS de que el estado cambió a 'recolectando'
    if current_state == "recolectando":
         # Mostrar el último mensaje de confirmación (viene del estado anterior) antes del spinner
         # st.chat_message("assistant").markdown(f"¡Gracias, {user_name}! Ya tengo todos los datos necesarios...") # Opcional, ya está en history


         # --- INICIO DEL BLOQUE TRY PARA CAPTURAR ERRORES EN PREDICCIÓN/CONSEJOS ---
         try:
             with st.spinner("Analizando tus datos..."):
                  time.sleep(1.5) # Simula tiempo de procesamiento

                  # Crear DataFrame para predicción desde los datos recolectados
                  # Asegurarse de que user_data_collector tiene todas las claves esperadas antes de crear el DF
                  if len(user_data) != len(feature_columns):
                       raise ValueError("Error interno: No se recolectaron todos los datos esperados para la predicción.")

                  user_data_df = pd.DataFrame([user_data], columns=feature_columns)

                  # Escalar datos del usuario
                  user_data_scaled = scaler.transform(user_data_df)

                  # Realizar la predicción
                  predicted_quality = model.predict(user_data_scaled)[0]

                  # Mostrar predicción
                  prediction_message = f"🌟 {user_name}, según mis cálculos, tu calidad de sueño es: **{int(predicted_quality)}**"
                  with st.chat_message("assistant"):
                      st.markdown(prediction_message)
                  st.session_state.messages.append({"role": "assistant", "content": prediction_message})

                  # Obtener y mostrar consejos
                  # Pasamos el DataFrame user_data_df a la función de consejos
                  advice_messages = get_advice_messages(predicted_quality, user_data) # Pasamos el diccionario user_data
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
                  final_message = f"¿Te gustaría hacer otra predicción ('empezar') o prefieres terminar ('salir'), {user_name}?"
                  with st.chat_message("assistant"):
                       st.markdown(final_message)
                  st.session_state.messages.append({"role": "assistant", "content": final_message})

                  # Si todo salió bien, cambiar el estado para esperar la respuesta final
                  next_state = "preguntando_reiniciar"
                  st.session_state.user_data_collector = {} # Limpiar datos recolectados

         except Exception as e:
             # --- SI OCURRE CUALQUIER ERROR EN EL BLOQUE TRY (PREDICCIÓN/CONSEJOS) ---
             error_message = f"❌ Lo siento, {user_name}. Ocurrió un error inesperado al procesar tu solicitud: {e}"
             with st.chat_message("assistant"):
                 st.error(error_message) # Mostrar el error en rojo
             st.session_state.messages.append({"role": "assistant", "content": error_message})

             # Guiamos al usuario a reiniciar o salir después de un error
             retry_message = f"Por favor, puedes intentar de nuevo ('empezar') o terminar ('salir'), {user_name}?"
             with st.chat_message("assistant"):
                  st.markdown(retry_message)
             st.session_state.messages.append({"role": "assistant", "content": retry_message})
             next_state = "preguntando_reiniciar" # Ir al estado donde puede reiniciar o salir
             st.session_state.user_data_collector = {} # Limpiar datos incompletos, ya no son confiables


    # --- Manejar el estado de Reinicio o Salida ---
    elif current_state == "preguntando_reiniciar":
         if "empezar" in user_input.lower() or "si" in user_input.lower() or "otra" in user_input.lower():
             # Reiniciar conversación
             bot_response = f"¡Excelente! Empecemos de nuevo, {user_name}. ¿Qué edad tienes? 🎂 ({feature_columns[0]})"
             next_state = "pidiendo_edad"
             st.session_state.user_data_collector = {} # Asegurarse de que esté limpio
             st.session_state.messages = [] # Opcional: Limpiar la conversación anterior al reiniciar

             # Agregar el primer mensaje de la nueva conversación
             st.session_state.messages.append({"role": "assistant", "content": bot_response})
             # No necesitamos mostrarlo inmediatamente con st.chat_message aquí
             # Se mostrará en el bucle de display al inicio del script en el próximo rerun
             bot_response = None # Limpiamos para que no se duplique

         elif "salir" in user_input.lower() or "no" in user_input.lower() or "terminar" in user_input.lower():
             bot_response = f"De acuerdo, {user_name}. ¡Que tengas un excelente descanso y un buen día! Adiós. 👋"
             next_state = "fin"
         else:
             bot_response = f"No entendí. ¿Quieres hacer otra predicción ('empezar') o terminar ('salir'), {user_name}?"
             # next_state remains 'preguntando_reiniciar'
             input_handled = True # El input fue manejado

    # --- Actualizar el estado y datos SI el input actual fue manejado (válido o inválido) ---
    # Y si el estado actual no era 'recolectando' (ese maneja su propia transición)
    if input_handled and current_state != "recolectando":
         st.session_state.chat_state = next_state
         st.session_state.user_data_collector = user_data # Guardar datos actualizados
    # Si el estado actual *era* 'recolectando', su bloque de try/except ya estableció el next_state.


    # --- Mostrar la respuesta del bot generada (si existe) ---
    # Esto muestra los mensajes de pregunta o error de validación, NO los mensajes de predicción/consejos.
    # Los mensajes de los estados 'recolectando' y el primer mensaje ('inicio') se manejan por separado.
    if bot_response and current_state != "recolectando" and current_state != "inicio":
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})


    # --- Manejar el estado final ---
    if st.session_state.chat_state == "fin":
        st.balloons() # Opcional: añadir algo visual al finalizar
        time.sleep(2) # Dar tiempo a ver los globos/último mensaje
        st.stop() # Detiene la ejecución del script Streamlit aquí.