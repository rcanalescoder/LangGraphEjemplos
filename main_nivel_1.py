# ============================================================
# main_nivel_1.py
# ============================================================
#
# Este fichero es el punto de entrada del programa.
#
# Su única responsabilidad es:
#
# - pedir una pregunta al usuario
# - construir el estado inicial
# - ejecutar el grafo
# - mostrar la respuesta
#
# Separar esto de la lógica del grafo permite reutilizar el
# sistema en otros entornos, por ejemplo:
#
# - una API con FastAPI
# - una interfaz web
# - un chatbot
# - un backend de reservas
#
# ============================================================


# Importamos la función que construye nuestro grafo.
from grafo_nivel_1 import crear_aplicacion


def main():

    print("\nAsistente de atención al cliente - Casa Rural\n")

    # Pedimos una pregunta al usuario
    pregunta = input("Escribe la pregunta del cliente: ").strip()

    # Creamos el estado inicial del sistema
    estado_inicial = {
        "pregunta_cliente": pregunta,
        "respuesta_asistente": ""
    }

    # Construimos la aplicación LangGraph
    aplicacion = crear_aplicacion()

    # Ejecutamos el grafo
    resultado = aplicacion.invoke(estado_inicial)

    # Mostramos la respuesta final
    print("\nRespuesta del asistente:\n")
    print(resultado["respuesta_asistente"])
    print()


# Este bloque permite ejecutar el script desde terminal
if __name__ == "__main__":
    main()