import json
import numpy as np
import matplotlib.pyplot as plt
from connect4.connect_state import ConnectState

# Leer el historial guardado por el torneo
with open("versus/match_Group B_vs_Group A.json", "r") as f:
    match_data = json.load(f)

# Tomamos la primera partida guardada en la lista 'games'
primera_partida = match_data["games"][0]

# Recreamos cada turno guardado
for turno, (board_list, action) in enumerate(primera_partida):
    matriz_tablero = np.array(board_list)
    state = ConnectState(board=matriz_tablero)
    
    print(f"Turno {turno} - El jugador decidió la columna: {action}")
    state.show()