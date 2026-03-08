# ============================================================
# NIVEL 1 - GRAFO MÍNIMO DE ATENCIÓN AL CLIENTE PARA UNA CASA RURAL
# ============================================================
#
# Objetivo del ejemplo:
# Construir la versión más simple posible de un asistente con LangGraph.
#
# Qué hace este ejemplo:
# - Recibe una pregunta de un cliente.
# - La envía a un único nodo del grafo.
# - Ese nodo usa un LLM en local con Ollama.
# - Devuelve una respuesta final apoyándose en un contexto fijo.
#
# Qué queremos enseñar aquí:
# - Qué es el estado (State).
# - Qué es un nodo (Node).
# - Cómo conectar un flujo mínimo con START y END.
# - Cómo usar LangGraph con un modelo local servido por Ollama.
#
# Requisitos previos:
# - Tener Ollama instalado en local.
# - Tener descargado un modelo, por ejemplo:
#       ollama pull llama3.1:8b
# - Tener instaladas estas librerías:
#       pip install langgraph langchain langchain-ollama
#
# ============================================================


# ------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------

# TypedDict nos permite definir con claridad la estructura del estado
# que circulará por el grafo.
from typing import TypedDict

# Importamos las piezas fundamentales de LangGraph para crear un grafo
# con un inicio, un final y nodos intermedios.
from langgraph.graph import StateGraph, START, END

# Este wrapper nos permite usar un modelo servido por Ollama desde LangChain.
from langchain_ollama import ChatOllama


# ------------------------------------------------------------
# DEFINICIÓN DEL ESTADO
# ------------------------------------------------------------
#
# El estado es la información que viaja por el grafo.
# En este ejemplo será deliberadamente pequeño para que se entienda bien.
#
# Campos:
# - pregunta_cliente: texto que escribe el usuario
# - respuesta_asistente: respuesta generada por el sistema
#
# Más adelante podremos añadir más datos, por ejemplo:
# - intención detectada
# - fechas de estancia
# - número de huéspedes
# - si viajan con mascota
# - precio calculado
# ------------------------------------------------------------

class EstadoCasaRural(TypedDict):
    pregunta_cliente: str
    respuesta_asistente: str


# ------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO LOCAL CON OLLAMA
# ------------------------------------------------------------
#
# Aquí indicamos qué modelo queremos usar desde Ollama.
#
# Ejemplos habituales:
# - "llama3.1:8b"
# - "mistral:7b"
# - "qwen2.5:7b"
#
# Temperature = 0 para que la respuesta sea más estable y menos creativa,
# algo útil en un ejemplo de atención al cliente.
# ------------------------------------------------------------

modelo = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


# ------------------------------------------------------------
# CONTEXTO FIJO DE NEGOCIO
# ------------------------------------------------------------
#
# Este bloque simula el conocimiento básico de la casa rural.
# En este nivel lo dejamos "hardcodeado" dentro del código para que
# el ejemplo sea muy fácil de seguir.
#
# Más adelante este contexto podría salir de:
# - una base de datos
# - un CRM
# - un documento
# - una herramienta externa
# - una base vectorial
# ------------------------------------------------------------

CONTEXTO_CASA_RURAL = """
Eres el asistente de atención al cliente de la casa rural La Encina Verde.

Información disponible sobre la casa rural:
- Se admiten mascotas, pero se debe avisar con antelación.
- El check-in es a partir de las 15:00.
- El check-out es hasta las 12:00.
- La casa dispone de jardín, barbacoa y chimenea.
- Hay piscina exterior disponible durante la temporada de verano.
- La casa está cerca de rutas de senderismo y de varios pueblos con encanto.
- Responde siempre con un tono cercano, claro y profesional.
- Si la información no aparece en el contexto, indícalo con honestidad.
"""


# ------------------------------------------------------------
# NODO DEL GRAFO
# ------------------------------------------------------------
#
# Este será el único nodo del flujo.
#
# Su trabajo es:
# 1. Leer la pregunta del cliente desde el estado.
# 2. Construir un prompt con el contexto del negocio.
# 3. Enviar ese prompt al modelo local.
# 4. Guardar la respuesta en el estado.
#
# En LangGraph, un nodo recibe el estado completo y devuelve un
# diccionario con los cambios que quiere aplicar.
# ------------------------------------------------------------

def responder_pregunta(state: EstadoCasaRural) -> dict:
    """
    Nodo único del grafo.
    Toma la pregunta del cliente y genera una respuesta usando
    el contexto fijo de la casa rural.
    """

    # Recuperamos la pregunta del estado.
    pregunta = state["pregunta_cliente"]

    # Construimos el prompt que se enviará al modelo.
    #
    # La idea es muy simple:
    # - Le recordamos al modelo qué papel tiene.
    # - Le damos contexto del negocio.
    # - Le pasamos la pregunta concreta del cliente.
    # - Le pedimos una respuesta clara y breve.
    prompt = f"""
{CONTEXTO_CASA_RURAL}

Pregunta del cliente:
{pregunta}

Instrucciones:
- Responde de forma natural.
- Sé claro y útil.
- No inventes información que no aparezca en el contexto.
- Si no sabes la respuesta, dilo con honestidad.

Respuesta:
"""

    # Invocamos el modelo local a través de Ollama.
    respuesta_modelo = modelo.invoke(prompt)

    # Devolvemos únicamente el campo del estado que queremos actualizar.
    return {
        "respuesta_asistente": respuesta_modelo.content
    }


# ------------------------------------------------------------
# CONSTRUCCIÓN DEL GRAFO
# ------------------------------------------------------------
#
# Aquí definimos la estructura del flujo.
#
# En este primer nivel el recorrido es extremadamente simple:
#
# START -> responder_pregunta -> END
#
# Es decir:
# - empieza el proceso
# - se ejecuta el único nodo
# - termina el proceso
# ------------------------------------------------------------

# Creamos el constructor del grafo indicando qué tipo de estado usaremos.
grafo = StateGraph(EstadoCasaRural)

# Registramos el nodo con un nombre identificativo.
grafo.add_node("responder_pregunta", responder_pregunta)

# Conectamos el inicio del flujo con el nodo.
grafo.add_edge(START, "responder_pregunta")

# Conectamos el nodo con el final del flujo.
grafo.add_edge("responder_pregunta", END)

# Compilamos el grafo para poder ejecutarlo.
aplicacion = grafo.compile()


# ------------------------------------------------------------
# EJECUCIÓN DE PRUEBA
# ------------------------------------------------------------
#
# Simulamos una pregunta real de un cliente.
# Este diccionario representa el estado inicial con el que arranca
# la ejecución del grafo.
# ------------------------------------------------------------

estado_inicial = {
    "pregunta_cliente": "¿Admitís mascotas?",
    "respuesta_asistente": ""
}

# Ejecutamos el grafo completo.
resultado = aplicacion.invoke(estado_inicial)


# ------------------------------------------------------------
# MOSTRAR RESULTADO
# ------------------------------------------------------------
#
# Imprimimos la respuesta generada por el asistente.
# ------------------------------------------------------------

print("Pregunta del cliente:")
print(estado_inicial["pregunta_cliente"])
print("\nRespuesta del asistente:")
print(resultado["respuesta_asistente"])