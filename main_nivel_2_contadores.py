# ============================================================
# main_nivel_2_contadores.py
# ============================================================
#
# Este fichero ejecuta el grafo del nivel 2 desde línea de comandos.
#
# Además de mostrar:
# - el análisis intermedio
# - la respuesta final
#
# también mostrará:
# - métricas por nodo
# - métricas totales del flujo
#
# Esto es útil para entender el coste técnico del sistema
# cuando una misma petición pasa por varios pasos.
#
# ============================================================

from grafo_nivel_2_contadores import crear_aplicacion


def imprimir_metricas_por_nodo(metricas_llm: dict):
    """
    Muestra de forma legible las métricas recogidas en cada nodo.
    """

    print("\nMétricas por llamada al LLM:\n")

    for nombre_nodo, metricas in metricas_llm.items():
        print(f"--- Nodo: {nombre_nodo} ---")
        print(f"Modelo: {metricas.get('modelo')}")
        print(f"Tokens de entrada: {metricas.get('tokens_entrada')}")
        print(f"Tokens de salida: {metricas.get('tokens_salida')}")
        print(f"Tokens totales: {metricas.get('tokens_totales')}")
        print(f"Latencia total: {metricas.get('latencia_total_ms')} ms")
        print(f"Tiempo de prompt: {metricas.get('latencia_prompt_ms')} ms")
        print(f"Tiempo de generación: {metricas.get('latencia_generacion_ms')} ms")
        print(f"Tiempo de carga del modelo: {metricas.get('latencia_carga_modelo_ms')} ms")
        print(f"Motivo de finalización: {metricas.get('done_reason')}")
        print()


def imprimir_metricas_totales(metricas_llm: dict):
    """
    Calcula y muestra un pequeño resumen acumulado del flujo completo.
    """

    total_tokens_entrada = sum(
        m.get("tokens_entrada", 0) for m in metricas_llm.values()
    )
    total_tokens_salida = sum(
        m.get("tokens_salida", 0) for m in metricas_llm.values()
    )
    total_tokens = sum(
        m.get("tokens_totales", 0) for m in metricas_llm.values()
    )
    total_latencia = sum(
        m.get("latencia_total_ms", 0) for m in metricas_llm.values()
    )

    print("\nResumen total del flujo:\n")
    print(f"Total tokens de entrada: {total_tokens_entrada}")
    print(f"Total tokens de salida: {total_tokens_salida}")
    print(f"Total tokens: {total_tokens}")
    print(f"Latencia total acumulada: {round(total_latencia, 2)} ms")
    print()


def main():
    """
    Punto de entrada principal del programa.
    """

    print("\nAsistente de atención al cliente - Casa Rural - Nivel 2\n")

    pregunta = input("Escribe la pregunta del cliente: ").strip()

    estado_inicial = {
        "pregunta_cliente": pregunta,
        "analisis_consulta": "",
        "respuesta_asistente": "",
        "metricas_llm": {}
    }

    aplicacion = crear_aplicacion()
    resultado = aplicacion.invoke(estado_inicial)

    print("\nAnálisis de la consulta:\n")
    print(resultado["analisis_consulta"])

    print("\nRespuesta del asistente:\n")
    print(resultado["respuesta_asistente"])

    metricas_llm = resultado.get("metricas_llm", {})

    if metricas_llm:
        imprimir_metricas_por_nodo(metricas_llm)
        imprimir_metricas_totales(metricas_llm)
    else:
        print("\nNo se pudieron recuperar métricas del modelo.\n")


if __name__ == "__main__":
    main()