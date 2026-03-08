# ============================================================
# grafo_nivel_1.py
# ============================================================
#
# Este fichero contiene toda la lógica del ejemplo de LangGraph.
#
# Su objetivo es definir:
# - el estado del sistema
# - el modelo que usaremos
# - el nodo del grafo
# - la construcción del flujo de ejecución
#
# Este fichero NO se ejecuta directamente.
# Su función es proporcionar la lógica para que otros programas
# (por ejemplo main.py) puedan utilizar el grafo.
#
# Esto es una buena práctica porque separa:
#
#     lógica del sistema  → grafo_nivel_1.py
#     ejecución del programa → main.py
#
# ============================================================


# ------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------

# TypedDict nos permite definir la estructura del estado que
# viajará por el grafo.
from typing import TypedDict

# Importamos las herramientas básicas de LangGraph para
# construir el flujo.
from langgraph.graph import StateGraph, START, END

# Esta clase nos permite utilizar un modelo LLM servido por Ollama.
from langchain_ollama import ChatOllama


# ------------------------------------------------------------
# DEFINICIÓN DEL ESTADO
# ------------------------------------------------------------
#
# El estado representa la información que el sistema transporta
# mientras el grafo se ejecuta.
#
# En este ejemplo el estado es muy simple y contiene solo dos
# elementos:
#
# - pregunta_cliente → la pregunta que hace el usuario
# - respuesta_asistente → la respuesta generada por el sistema
#
# Más adelante podremos ampliar el estado con más información,
# por ejemplo:
#
# - fechas de reserva
# - número de huéspedes
# - si viajan con mascota
# - precio calculado
# - resultados de herramientas externas
#
# ------------------------------------------------------------

class EstadoCasaRural(TypedDict):
    pregunta_cliente: str
    respuesta_asistente: str


# ------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# ------------------------------------------------------------
#
# Esta función crea el modelo que utilizará el sistema.
#
# Separarlo en una función facilita cambiar el modelo en el
# futuro sin modificar el resto del código.
#
# El modelo se ejecuta en local mediante Ollama.
#
# Antes de ejecutar el programa debes tener descargado el modelo:
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
# Este texto contiene información básica sobre la casa rural.
#
# En este primer nivel lo mantenemos fijo dentro del código
# para simplificar el ejemplo.
#
# En niveles posteriores este contexto podría provenir de:
#
# - una base de datos
# - documentos
# - un sistema RAG
# - APIs externas
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

Instrucciones para responder:
- Utiliza un tono cercano y profesional.
- Sé claro y útil.
- Si no conoces la respuesta, indícalo con honestidad.
"""


# ------------------------------------------------------------
# NODO DEL GRAFO
# ------------------------------------------------------------
#
# Un nodo representa un paso dentro del flujo del agente.
#
# En este nivel solo tenemos un nodo, cuya responsabilidad es:
#
# 1. Leer la pregunta del cliente
# 2. Crear un prompt con el contexto de negocio
# 3. Enviar ese prompt al modelo
# 4. Guardar la respuesta en el estado
#
# ------------------------------------------------------------

def responder_pregunta(state: EstadoCasaRural) -> dict:

    modelo = crear_modelo()

    pregunta = state["pregunta_cliente"]

    prompt = f"""
{CONTEXTO_CASA_RURAL}

Pregunta del cliente:
{pregunta}

Respuesta:
"""

    respuesta = modelo.invoke(prompt)

    return {
        "respuesta_asistente": respuesta.content
    }


# ------------------------------------------------------------
# CONSTRUCCIÓN DEL GRAFO
# ------------------------------------------------------------
#
# Esta función construye el grafo completo y devuelve
# la aplicación lista para ejecutarse.
#
# Flujo del ejemplo:
#
# START → responder_pregunta → END
#
# ------------------------------------------------------------

def crear_aplicacion():

    grafo = StateGraph(EstadoCasaRural)

    grafo.add_node("responder_pregunta", responder_pregunta)

    grafo.add_edge(START, "responder_pregunta")
    grafo.add_edge("responder_pregunta", END)

    return grafo.compile()