import json
import numpy as np
import matplotlib.pyplot as plt
from connect4.connect_state import ConnectState

# 1. Leer el historial completo del archivo JSON generado por el torneo
archivo_path = "versus/match_Group C_vs_Group A.json"

with open(archivo_path, "r") as f:
    match_data = json.load(f)

# Guardamos los nombres base registrados en los metadatos globales
nombre_a = match_data["player_a"]  
nombre_b = match_data["player_b"]  

print(f"Marcador Global de la Serie: {nombre_a} ({match_data['player_a_wins']}) vs {nombre_b} ({match_data['player_b_wins']})")

# Recreamos el generador de números aleatorios exacto que usó la función `play`
# Asumiendo que usaste los valores por defecto del código (seed=911)
torneo_rng = np.random.default_rng(911)

# 2. Recorrer secuencialmente cada una de las partidas jugadas
for numero_partida, partida_actual in enumerate(match_data["games"]):
    print(f"\n========================================================")
    print(f"       REPRODUCIENDO REPLAY DE LA PARTIDA Nº: {numero_partida + 1}")
    print(f"========================================================\n")
    
    # --- DETECCIÓN EXACTA REPLICANDO EL TORNEO ---
    # Lanzamos la misma moneda que lanzó el framework justo antes de iniciar esta partida.
    # Como los generadores de NumPy son deterministas con la misma semilla, 
    # esta llamada a .random() devolverá exactamente el mismo valor que devolvió en el torneo.
    if torneo_rng.random() < 0.5:
        # Salió Cara: El player_a recibe la variable 'first' (Fichas Rojas)
        agente_rojo_minus_1 = nombre_a
        agente_amarillo_1 = nombre_b
    else:
        # Salió Cruz: El player_b recibe la variable 'first' (Fichas Rojas)
        agente_rojo_minus_1 = nombre_b
        agente_amarillo_1 = nombre_a

    print(f" [ROLES ASIGNADOS POR LA MONEDA EN ESTE JUEGO]:")
    print(f" -> Fichas Rojas (-1, Empieza): {agente_rojo_minus_1}")
    print(f" -> Fichas Amarillas (1, Segundo): {agente_amarillo_1}\n")
    print("--------------------------------------------------------")

    # 3. Recorrer cada turno dentro de la partida actual
    for turno, (board_list, action) in enumerate(partida_actual):
        matriz_tablero = np.array(board_list)
        state = ConnectState(board=matriz_tablero)
        
        # Contamos cuántas fichas hay en el tablero antes de este movimiento
        num_fichas = np.count_nonzero(matriz_tablero)
        
        # Si hay un número par de fichas (ej. 0 al inicio), le toca al jugador inicial (-1)
        if num_fichas % 2 == 0:
            agente_actual = agente_rojo_minus_1
            color_ficha = "Rojo (-1)"
        else:
            agente_actual = agente_amarillo_1
            color_ficha = "Amarillo (1)"
        
        # Imprimimos en la consola con el nombre exacto del modelo que pensó la jugada
        print(f"Turno {turno:02d} | [{agente_actual}] ({color_ficha}) pensó y tiró en la columna: {action}")
        
        # Muestra el tablero de forma gráfica usando Matplotlib
        state.show()

    print(f"\n--- Fin de la partida {numero_partida + 1} ---")
    if numero_partida < len(match_data["games"]) - 1:
        input("Presiona la tecla ENTER en la consola para comenzar a ver la siguiente partida...")

print("\n========================================================")
print("¡Has terminado de ver todos los replays del torneo!")
print("========================================================")
