# ============================================================
# main_nivel_2.py
# ============================================================
#
# Este fichero actúa como punto de entrada para ejecutar
# el segundo ejemplo desde línea de comandos.
#
# Su responsabilidad es:
# - pedir una pregunta al usuario
# - construir el estado inicial
# - ejecutar el grafo
# - mostrar tanto el análisis intermedio como la respuesta final
#
# Mostrar el análisis intermedio ayuda mucho en este nivel
# porque deja ver que ahora el flujo tiene dos pasos y que el
# primer nodo realmente aporta información al segundo.
#
# ============================================================


# Importamos la función que construye el grafo del nivel 2.
from grafo_nivel_2 import crear_aplicacion


def main():
    """
    Punto de entrada del programa.
    """

    print("\nAsistente de atención al cliente - Casa Rural - Nivel 2\n")

    # Pedimos una pregunta al usuario.
    pregunta = input("Escribe la pregunta del cliente: ").strip()

    # Construimos el estado inicial.
    #
    # En este nivel añadimos también el campo analisis_consulta,
    # que al principio estará vacío y se rellenará en el primer nodo.
    estado_inicial = {
        "pregunta_cliente": pregunta,
        "analisis_consulta": "",
        "respuesta_asistente": ""
    }

    # Creamos la aplicación de LangGraph.
    aplicacion = crear_aplicacion()

    # Ejecutamos el grafo completo.
    resultado = aplicacion.invoke(estado_inicial)

    # Mostramos el análisis intermedio para entender mejor el flujo.
    print("\nAnálisis de la consulta:\n")
    print(resultado["analisis_consulta"])

    # Mostramos la respuesta final del asistente.
    print("\nRespuesta del asistente:\n")
    print(resultado["respuesta_asistente"])
    print()


# Permite ejecutar el script directamente desde terminal.
if __name__ == "__main__":
    main()