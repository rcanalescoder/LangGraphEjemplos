# ============================================================
# grafo_nivel_2.py
# ============================================================
#
# Este fichero contiene la lógica del segundo ejemplo de LangGraph
# para el caso de la casa rural.
#
# En el nivel 1 teníamos un único nodo que hacía todo:
# - leer la pregunta
# - interpretar el contexto
# - redactar la respuesta
#
# En este nivel vamos a dar un paso más en madurez:
# separaremos el trabajo en dos nodos secuenciales.
#
# Nodo 1:
#     analizar_consulta
#
# Nodo 2:
#     redactar_respuesta
#
# De esta forma el flujo se parece un poco más a un proceso
# estructurado y no tanto a una única llamada al modelo.
#
# ============================================================


# ------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------

# TypedDict nos ayuda a definir con claridad la estructura
# del estado que circulará por el grafo.
from typing import TypedDict

# Importamos las piezas básicas de LangGraph para construir
# el flujo con inicio, fin y nodos intermedios.
from langgraph.graph import StateGraph, START, END

# Esta clase permite usar un modelo servido por Ollama en local.
from langchain_ollama import ChatOllama


# ------------------------------------------------------------
# DEFINICIÓN DEL ESTADO
# ------------------------------------------------------------
#
# En este nivel el estado crece un poco respecto al ejemplo
# anterior.
#
# Ahora no solo guardamos:
# - la pregunta del cliente
# - la respuesta final
#
# También vamos a guardar:
# - un análisis intermedio de la consulta
#
# Esto es importante porque muestra una de las ideas clave
# de LangGraph:
#
# el estado puede almacenar resultados intermedios entre nodos.
#
# ------------------------------------------------------------

class EstadoCasaRural(TypedDict):
    pregunta_cliente: str
    analisis_consulta: str
    respuesta_asistente: str


# ------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# ------------------------------------------------------------
#
# Seguimos utilizando Ollama en local.
# Esto permite ejecutar el ejemplo sin depender de una API externa.
#
# Antes de usar este código conviene tener descargado el modelo:
#
#     ollama pull llama3.1:8b
#
# ------------------------------------------------------------

def crear_modelo():
    return ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )


# ------------------------------------------------------------
# CONTEXTO DE NEGOCIO
# ------------------------------------------------------------
#
# Mantenemos un contexto fijo para que el ejemplo siga siendo
# fácil de entender.
#
# Más adelante este contexto podría venir de una base de datos,
# documentos, herramientas o sistemas externos.
#
# ------------------------------------------------------------

CONTEXTO_CASA_RURAL = """
Eres el asistente de atención al cliente de la casa rural La Encina Verde.

Información sobre la casa rural:

- Se admiten mascotas, pero es necesario avisar previamente.
- El check-in es a partir de las 15:00.
- El check-out es hasta las 12:00.
- La casa tiene jardín, barbacoa y chimenea.
- Hay piscina exterior durante la temporada de verano.
- Está cerca de rutas de senderismo y de varios pueblos con encanto.
- Es un alojamiento pensado para una estancia tranquila y cómoda.

Instrucciones para responder:
- Utiliza un tono cercano y profesional.
- Sé claro y útil.
- Si no conoces la respuesta, indícalo con honestidad.
"""


# ------------------------------------------------------------
# NODO 1: ANALIZAR LA CONSULTA
# ------------------------------------------------------------
#
# Este nodo no responde todavía al cliente.
#
# Su responsabilidad es comprender mejor la pregunta y generar
# un pequeño análisis intermedio.
#
# Ese análisis puede recoger cosas como:
# - qué parece necesitar el cliente
# - si está pidiendo una recomendación
# - si menciona niños, mascotas u otros detalles
# - qué información parece importante para responder
#
# El resultado se guarda en el estado para que el siguiente nodo
# lo utilice.
#
# ------------------------------------------------------------

def analizar_consulta(state: EstadoCasaRural) -> dict:
    """
    Analiza la consulta del cliente y genera una interpretación
    breve que se guardará en el estado.
    """

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]

    prompt = f"""
Eres un asistente que analiza consultas de clientes para una casa rural.

Tu tarea no es responder todavía.
Tu tarea es interpretar qué necesita el cliente.

Analiza esta consulta y devuelve un texto breve en castellano que explique:
- cuál es la necesidad principal del cliente
- qué elementos importantes menciona
- qué tipo de respuesta parece esperar

Consulta del cliente:
{pregunta}
"""

    respuesta = modelo.invoke(prompt)

    return {
        "analisis_consulta": respuesta.content
    }


# ------------------------------------------------------------
# NODO 2: REDACTAR LA RESPUESTA
# ------------------------------------------------------------
#
# Este nodo sí genera la respuesta final para el cliente.
#
# Para ello utiliza:
# - la pregunta original
# - el análisis generado por el nodo anterior
# - el contexto fijo del negocio
#
# Esta separación hace que la respuesta se construya con una
# comprensión previa de la necesidad del usuario.
#
# ------------------------------------------------------------

def redactar_respuesta(state: EstadoCasaRural) -> dict:
    """
    Genera la respuesta final del asistente utilizando la
    pregunta original y el análisis previo.
    """

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]
    analisis = state["analisis_consulta"]

    prompt = f"""
{CONTEXTO_CASA_RURAL}

Además, dispones de este análisis previo de la consulta del cliente:

{analisis}

Pregunta original del cliente:
{pregunta}

Redacta una respuesta final para el cliente.
La respuesta debe:
- sonar natural
- ser útil
- estar bien enfocada a lo que el cliente necesita
- basarse solo en la información disponible
"""

    respuesta = modelo.invoke(prompt)

    return {
        "respuesta_asistente": respuesta.content
    }


# ------------------------------------------------------------
# CONSTRUCCIÓN DEL GRAFO
# ------------------------------------------------------------
#
# El flujo de este nivel ya no tiene un solo paso.
#
# Ahora el recorrido es:
#
# START → analizar_consulta → redactar_respuesta → END
#
# Esto representa un pequeño proceso secuencial:
# primero comprender, después responder.
#
# ------------------------------------------------------------

def crear_aplicacion():
    """
    Construye y compila el grafo del nivel 2.
    """

    grafo = StateGraph(EstadoCasaRural)

    # Registramos los dos nodos del flujo.
    grafo.add_node("analizar_consulta", analizar_consulta)
    grafo.add_node("redactar_respuesta", redactar_respuesta)

    # Definimos el orden secuencial del proceso.
    grafo.add_edge(START, "analizar_consulta")
    grafo.add_edge("analizar_consulta", "redactar_respuesta")
    grafo.add_edge("redactar_respuesta", END)

    return grafo.compile()