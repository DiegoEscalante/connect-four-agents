# Agente Julian — Minimax + Alfa-Beta + Heurística Allis

## Idea principal

Agente de Connect-4 basado en **búsqueda Minimax con poda Alfa-Beta** y una función de evaluación heurística inspirada en la teoría de **Victor Allis (1988)**.

Paradigma **knowledge-based** (búsqueda determinista + conocimiento del dominio), a diferencia de los agentes MCTS del grupo que son estocásticos y no usan conocimiento explícito del juego.

## Archivos

| Archivo | Descripción |
|---|---|
| `policy.py` | Implementación del agente `Julian` |
| `entrega.ipynb` | Notebook de análisis con 6 experimentos |

## Uso

```python
from groups.Julian.policy import Julian

agent = Julian(depth=5, w_center=6, w_odd_even=20, use_allis=True)
agent.mount()
col = agent.act(board)   # board: np.ndarray (6×7)
```

## Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `depth` | 5 | Profundidad de búsqueda Minimax. Variable numérica principal. |
| `w_center` | 6 | Peso del sesgo a la columna central. |
| `w_odd_even` | 20 | Peso del bonus de paridad de Allis (teoría odd/even threats). |
| `use_allis` | True | Activa/desactiva la teoría de amenazas pares/impares. |

## Concepto clave — Teoría de Allis

Victor Allis (1988) demostró que Connect-4 está resuelto y el primer jugador (Rojo) gana con juego perfecto. Su teoría de **odd/even threats** establece que:

- **Red (juega primero)** prefiere amenazas en **filas impares** (1, 3, 5 desde abajo)
- **Yellow (juega segundo)** prefiere amenazas en **filas pares** (2, 4, 6)

Esto se debe a la paridad de turnos en el endgame (Zugzwang). La función de evaluación añade un bonus a las amenazas propias con la paridad correcta y penaliza las del oponente.

## Rendimiento

| Config | vs Aleatorio Rojo | vs Aleatorio Amarillo |
|---|---|---|
| depth=5, Allis=True | 100% | 100% |
| depth=3, Allis=True | 100% | 100% |

Benchmarks de tiempo: depth=3 → 0.17s/mov, depth=4 → 0.6s/mov, depth=5 → 0.78s/mov.
Con depth=5 y una partida de ~21 movimientos propios: ~16 segundos total (margen amplio respecto al límite de 60s).

## Optimizaciones implementadas

1. **Chequeo incremental de victoria** (`winner_at`): O(28) en vez de O(168)
2. **Ventanas precalculadas** al importar el módulo (69 ventanas, no se recalculan)
3. **Allis fusionado** al scan de ventanas (un solo pase en vez de dos)
4. **Center-first move ordering**: mejora la poda alfa-beta hasta ~O(b^{d/2})
