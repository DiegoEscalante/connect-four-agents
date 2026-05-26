# Agente Combinado — MCTS (Rojo) + Minimax+Allis (Amarillo)

Agente híbrido que selecciona el motor de búsqueda según el color asignado:

| Color | Motor | Origen |
|---|---|---|
| **Rojo** (primer jugador) | MCTS con UCB1 + tree reuse | Grupo A (Escalante) |
| **Amarillo** (segundo jugador) | Minimax + Alfa-Beta + Heurística Allis | Julián Romero |

## Justificación del diseño

Connect-4 está resuelto: Rojo gana con juego perfecto. MCTS explora el árbol de forma amplia sin conocimiento explícito, lo que lo hace robusto como primer jugador con presupuesto de tiempo generoso. Minimax+Allis incorpora conocimiento de dominio específico (teoría odd/even threats de Victor Allis), que es especialmente valioso para Amarillo, donde la paridad de filas determina si una amenaza se materializa antes de que el oponente la bloquee.

## Uso

```python
from groups.Combinado.policy import Combinado

agent = Combinado()
agent.mount()                  # inicializa ambos motores y el cronómetro
col = agent.act(board)         # board: np.ndarray (6×7)
```

El agente detecta automáticamente su color en el primer `act()`: si el tablero está vacío es Rojo (usa MCTS), si ya tiene fichas es Amarillo (usa Minimax).

## Archivos

| Archivo | Descripción |
|---|---|
| `policy.py` | Implementación completa del agente `Combinado` |
| `readme.md` | Este archivo |

No se requieren archivos externos ni entrenamiento previo.

## Requisitos

```
numpy
```

Ejecutar desde la raíz del proyecto (donde está la carpeta `connect4/`).

## Parámetros internos

| Motor | Parámetro | Valor | Descripción |
|---|---|---|---|
| MCTS | `total_game_time` | 60.0 s | Presupuesto total de tiempo por partida |
| MCTS | `exploration_constant` | √2 | Constante UCB1 |
| Minimax | `depth` | 10 | Profundidad máxima (Iterative Deepening) |
| Minimax | `time_limit` | 55.0 s | Presupuesto de tiempo por partida |
| Minimax | `w_odd_even` | 20 | Peso del bonus de paridad de Allis |
