# ============================================================
# main_nivel_4_bis.py
# ============================================================
#
# Punto de entrada del Nivel 4 bis.
#
# Este programa:
# - genera el diagrama del grafo si no existe
# - pide una pregunta al usuario
# - ejecuta el flujo
# - muestra la respuesta final
# - muestra métricas de llamadas al LLM
# ============================================================

from langchain_core.messages import HumanMessage

from grafo_nivel_4_bis import crear_aplicacion, asegurar_diagrama_grafo


def imprimir_metricas(metricas_llm: dict):
    """
    Imprime métricas por llamada al modelo.
    """

    print("\nMétricas por llamada al LLM:\n")

    for nombre_nodo, metricas in metricas_llm.items():
        print(f"--- {nombre_nodo} ---")
        for clave, valor in metricas.items():
            print(f"{clave}: {valor}")
        print()


def main():
    """
    Punto de entrada principal.
    """

    print("\nAsistente Casa Rural - Nivel 4 bis\n")

    # --------------------------------------------------------
    # Generar diagrama automáticamente si no existe
    # --------------------------------------------------------
    asegurar_diagrama_grafo()

    # --------------------------------------------------------
    # Entrada del usuario
    # --------------------------------------------------------
    pregunta = input("Pregunta del cliente: ").strip()

    estado_inicial = {
        "messages": [HumanMessage(content=pregunta)],
        "llm_calls": 0,
        "metricas_llm": {}
    }

    app = crear_aplicacion()
    resultado = app.invoke(estado_inicial)

    # El último mensaje del flujo debería ser la respuesta final
    mensaje_final = resultado["messages"][-1]

    print("\nRespuesta del asistente:\n")
    print(mensaje_final.content)

    print(f"\nNúmero total de llamadas al LLM: {resultado.get('llm_calls', 0)}")

    metricas = resultado.get("metricas_llm", {})
    if metricas:
        imprimir_metricas(metricas)


if __name__ == "__main__":
    main()