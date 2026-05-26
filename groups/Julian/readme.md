# Agente Julian — Minimax + Alfa-Beta + Heurística Allis

Agente de Connect-4 basado en **búsqueda Minimax con poda Alfa-Beta**, heurística inspirada en la teoría de **Victor Allis (1988)**, e **Iterative Deepening con límite de tiempo**.

Paradigma **knowledge-based** (búsqueda determinista + conocimiento explícito del dominio), a diferencia del enfoque MCTS (estocástico, sin conocimiento del juego) usado por los otros agentes del grupo.

## Archivos

| Archivo | Descripción |
|---|---|
| `policy.py` | Implementación completa del agente `Julian` |
| `entrega.ipynb` | Notebook con 7 experimentos de análisis y validación |
| `readme.md` | Este archivo |

No se requieren archivos de datos externos: el agente es puramente algorítmico (sin entrenamiento offline).

## Requisitos

```
numpy
```

El agente se importa desde la raíz del proyecto (donde está la carpeta `connect4/`).

## Uso

```python
from groups.Julian.policy import Julian

agent = Julian()       # configuración por defecto recomendada
agent.mount()          # inicializa el cronómetro y limpia la tabla de transposiciones
col = agent.act(board) # board: np.ndarray (6×7), devuelve columna 0-6
```

## Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `depth` | 10 | Profundidad máxima de búsqueda (el Iterative Deepening se detiene antes si se acaba el tiempo). |
| `time_limit` | 55.0 | Presupuesto total de tiempo en segundos por partida. Controla en la práctica la profundidad real alcanzada. |
| `w_center` | 6 | Peso del sesgo hacia la columna central (más ventanas disponibles desde el centro). |
| `w_odd_even` | 20 | Peso del bonus de paridad de Allis (teoría odd/even threats). 0 = desactiva Allis. |
| `use_allis` | True | Activa/desactiva la teoría de amenazas pares/impares de Allis. |

## Concepto clave — Teoría de Allis

Victor Allis (1988) demostró que Connect-4 está resuelto y el primer jugador (Rojo) gana con juego perfecto. Su teoría de **odd/even threats** establece que:

- **Rojo (juega primero)** prefiere amenazas en **filas impares** (1, 3, 5 desde abajo)
- **Amarillo (juega segundo)** prefiere amenazas en **filas pares** (2, 4, 6)

Esto se debe a la paridad de turnos en el endgame (Zugzwang). La función de evaluación añade un bonus `w_odd_even` a las amenazas propias en la paridad correcta y penaliza las del oponente.

## Optimizaciones implementadas

1. **Tabla de transposiciones con Zobrist hashing**: cachea posiciones ya evaluadas con un hash de 64 bits (XOR de valores aleatorios por ficha). Actualización O(1) al hacer/deshacer un movimiento. Permite reutilizar resultados de iteraciones anteriores del Iterative Deepening.
2. **Iterative Deepening**: busca a depth=1, 2, 3, … hasta agotar el presupuesto de tiempo. El resultado de la profundidad anterior queda en la TT y acelera la siguiente. Garantiza el mejor movimiento disponible en cualquier momento.
3. **Atajo táctico inmediato**: antes de llamar a Minimax, revisa si hay un movimiento ganador o un bloqueo urgente en O(cols). Evita desperdiciar búsqueda en situaciones obvias.
4. **Chequeo incremental de victoria** (`winner_at`): O(28) en vez de O(168) al verificar solo las líneas que pasan por la última ficha colocada.
5. **Ventanas precalculadas**: las 69 ventanas de 4 celdas se generan al importar el módulo, no en cada evaluación.
6. **Center-first move ordering**: ordena las columnas [3,2,4,1,5,0,6] para mejorar la poda alfa-beta hasta ~O(b^{d/2}).

## Rendimiento

| Config | vs Aleatorio (Rojo) | vs Aleatorio (Amarillo) | vs MCTS (500 sims) |
|---|---|---|---|
| depth≤10, time_limit=55s | 100% | 100% | ~85% (Rojo+Amarillo) |

Benchmarks aproximados: depth=3 → 0.17 s/mov, depth=4 → 0.6 s/mov, depth=5 → 0.78 s/mov. Con Iterative Deepening y `time_limit=55s`, el agente típicamente alcanza depth=7–8 en apertura y depth=10+ en el endgame (tablero más lleno = menos ramas).
