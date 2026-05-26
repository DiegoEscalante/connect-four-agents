# Agente Connect-4 – Cristian Manuel Castañeda Gutiérrez

**Fundamentos de Inteligencia Artificial | Universidad de La Sabana**

---

## Código del agente

| Versión | Archivo | Enlace |
|---------|---------|--------|
| V1 | `policy.py` | [Ver en GitHub](https://github.com/DiegoEscalante/connect-four-agents/blob/Castaneda-agent/groups/Cristian/policy.py) |
| V2 | `policy_qValues.py` | [Ver en GitHub](https://github.com/DiegoEscalante/connect-four-agents/blob/Castaneda-agent/groups/Cristian/policy_qValues.py) |

---

## Idea principal y diferenciación

El agente implementa **MCTS/UCT adversarial** (Monte Carlo Tree Search con Upper Confidence Bound) para Connect-4. Existen dos versiones con distinto nivel de sofisticación.

### Lo que distingue este agente de los demás del grupo

1. **Backpropagación adversarial (negamax-style):** los nodos acumulan recompensas desde la perspectiva del jugador que los creó, no de un único jugador fijo. Esto hace que UCB1 sea correcto para ambos colores simultáneamente — la selección optimiza al jugador actual en cada nivel del árbol.

2. **Q-table persistente entre partidas:** la experiencia acumulada en partidas anteriores se guarda en disco (`q_table.pkl`) y se carga al inicio de cada nueva partida. El agente mejora con el tiempo sin reentrenamiento explícito.

3. **Rollouts guiados por Q-table:** durante la fase de simulación, en lugar de jugar completamente al azar, se selecciona la acción con probabilidad proporcional al Q-value acumulado (equivalente a un softmax). Se activa cuando la Q-table tiene más de 100 entradas.

---

## Descripción de las versiones

### V1 — `policy.py` (MCTS puro)

- MCTS/UCT con rollouts aleatorios
- Heurística táctica de 1 paso: detecta victoria inmediata o bloqueo necesario **antes** de lanzar MCTS (evita pérdidas triviales)
- Expansión con orden centrado: columna central primero
- Sin memoria entre partidas

**Parámetros clave:**
```python
Cristian(simulations=500, C=1.414)
```

### V2 — `policy_qValues.py` (MCTS + Q-table)

Extiende V1 con tres mejoras:

| Mejora | Detalle |
|--------|---------|
| Backpropagación adversarial | Recompensas alternantes por nivel; UCB1 correcto para ambos jugadores |
| Q-table persistente | Archivo `q_table.pkl` en disco; se actualiza en cada simulación MCTS |
| Rollouts guiados | Selección por Q-values (softmax) cuando hay suficientes datos |

**Parámetros clave:**
```python
Cristian(simulations=500, C=1.414, q_guided=True, save_every=10)
```

**Datos necesarios para ejecución:** el archivo `q_table.pkl` (si existe) se carga automáticamente desde la misma carpeta. Si no existe, el agente empieza con tabla vacía y la construye durante las partidas.

---

## Análisis de rendimiento

El estudio completo está en [`entrega.ipynb`](entrega.ipynb). Resumen de resultados clave:

### Configuración del agente

| Variable | Efecto observado |
|----------|-----------------|
| `simulations` | Mayor salto entre 50 → 200 sims; rendimientos decrecientes a partir de 500 |
| `q_guided=True` | Mejora victorias especialmente como Amarillo (segunda ventaja posicional) |
| Q-table vacía vs. cargada | Primeras 20-30 partidas son noisier; luego la tasa de victoria se estabiliza hacia arriba |

### Tipo de oponente

| Oponente | V1 | V2 |
|----------|----|----|
| Aleatorio | >90% victorias | >90% victorias |
| Sí mismo (self-play) | ~50% (simétrico) | ~50% (simétrico) |
| V1 (directo) | — | ~73% victorias |
| Grupo A (MCTS + reuse + tiempo) | — | ~25% victorias |

### Debilidad principal identificada

V2 pierde el 75% de las partidas contra el Grupo A. La causa no es la calidad del rollout sino el **presupuesto de búsqueda**: el Grupo A usa MCTS basado en tiempo (60 s totales) con reutilización del árbol entre turnos, logrando 6-10× más simulaciones efectivas por turno que las 500 fijas de V2.

---

## Propuestas de mejora para una versión futura

### 1. Reutilización del árbol entre turnos
Conservar el subárbol del movimiento elegido (y el del rival) como raíz del siguiente turno. Esto multiplica las simulaciones efectivas 3-5× sin aumentar el tiempo de cómputo. Es la mejora con mayor impacto según el Exp. 9.

### 2. Presupuesto de tiempo adaptativo
Reemplazar `for _ in range(simulations)` por `while (time.time() - start) < budget`. Permite usar todo el tiempo disponible en posiciones complejas y terminar antes en posiciones triviales.

### 3. Tablas de transposición (Zobrist hashing)
La Q-table actual usa `board.tobytes()` como clave — estados similares no comparten información. Con hash de Zobrist, tableros alcanzados por distintas secuencias de movimientos comparten sus Q-values, acelerando la convergencia.

---

## Guía de uso

```bash
# Desde la raíz del repositorio
cd connect-four-agents

# Validar V1 y V2 vs aleatorio y entre sí
python validate.py

# Validar V2 vs Grupo A
python validate_vs_groupA.py

# Abrir el notebook de análisis
jupyter notebook groups/Cristian/entrega.ipynb
```

**Requisitos:** `numpy`, `matplotlib`, `jupyter`. El archivo `q_table.pkl` se genera automáticamente al jugar partidas con V2.
