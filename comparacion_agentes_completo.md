# Comparación de Agentes — Julian vs Cristian vs Group A

**Fundamentos de Inteligencia Artificial 2026.1**

---

## 1. Paradigma

| Aspecto | Julian (Minimax + Allis) | Cristian V2 (MCTS + Q-table) | Group A (MCTS + Tree Reuse) |
|---|---|---|---|
| **Tipo** | Búsqueda determinista + conocimiento | Simulación estocástica + aprendizaje persistente | Simulación estocástica |
| **Fuente de valor** | Función heurística hecha a mano | Rollouts guiados por Q-table acumulada entre partidas | Rollouts aleatorios |
| **Conocimiento del dominio** | Alto (ventanas, centro, paridad Allis) | Medio (Q-table aprende con experiencia, center-first) | Bajo (solo sesgo al centro) |
| **Parámetro principal** | `depth` (profundidad de búsqueda) | `simulations` (rollouts por movimiento) | Tiempo por turno (simulaciones) |
| **Reproducibilidad** | Determinista — mismo tablero, mismo movimiento | Estocástico — varía por seed y estado de Q-table | Estocástico — varía por seed |
| **Aprende entre partidas** | No | **Sí** — Q-table persiste en disco | No |

---

## 2. Arquitectura interna

### Julian — Minimax + Alfa-Beta + Iterative Deepening

```
act(board)
  │
  ├── immediate_tactic()        # O(cols): ganar o bloquear en 1 movimiento
  │
  └── Iterative Deepening       # depth = 1, 2, 3, ... hasta agotar tiempo
        └── minimax(depth d)
              ├── TT lookup      # Zobrist hash → resultado cacheado O(1)
              ├── winner_at()    # chequeo incremental O(28) vs O(168)
              ├── poda α-β       # corta ramas que no afectan el resultado
              └── evaluate()    # ventanas (69) + centro + Allis odd/even
```

**Componentes clave:**
- **69 ventanas precalculadas** al importar el módulo (horizontales, verticales, diagonales)
- **Zobrist hashing**: hash de 64 bits con XOR incremental — actualizar el hash al hacer un movimiento es O(1)
- **Tabla de transposiciones (_TT)**: evita reevaluar posiciones ya vistas, permite mayor profundidad efectiva
- **Teoría de Allis**: bonus/penalización por amenazas en la paridad de fila correcta (odd para Rojo, even para Amarillo)

---

### Cristian V2 — MCTS/UCT adversarial + Q-table persistente

```
act(board)
  │
  ├── _immediate_action()       # O(7): ganar o bloquear en 1 movimiento
  │     ├── Prioridad 1: ¿puedo ganar ya?  → jugar col ganadora
  │     └── Prioridad 2: ¿oponente gana?   → bloquear col
  │
  └── MCTS x500 simulaciones
        ├── Selección   (UCB1 adversarial — backprop alternante)
        ├── Expansión   (center-first: col 3 primero)
        ├── Simulación  (rollout guiado por Q-table si >100 entradas)
        └── Retropropagación (invierte result en cada nivel: result = 1 - result)
              └── _q_update()  → actualiza Q-table en disco por cada nodo
```

**Componentes clave:**

- **Q-table persistente (`q_table.pkl`)**: diccionario `{(board.tobytes(), col) → [suma_rewards, visitas]}` guardado en disco. Acumula experiencia de **todas las partidas jugadas**, no solo la actual.
- **Rollouts Q-guiados**: en `_simulate()`, cuando la Q-table supera 100 entradas, selecciona acciones con probabilidad proporcional al Q-value del jugador que mueve en ese estado (softmax simplificado). Los rollouts mejoran con cada partida.
- **Backpropagación alternante (corrección adversarial)**: en lugar de propagar siempre desde la perspectiva de `me`, invierte `result = 1.0 - result` en cada nivel. Efecto: `node.wins/visits` refleja siempre el win-rate del **jugador que creó ese nodo**, haciendo que UCB1 sea correcto para ambos jugadores durante la selección. Esto elimina el sesgo Rojo/Amarillo que tenía V1.
- **Heurística táctica O(7)**: antes de gastar simulaciones, detecta victorias y bloqueos en 1 paso. Garantiza que el agente nunca pierda por no ver algo obvio.
- **Q-update en backpropagación**: cada una de las 500 simulaciones por movimiento actualiza la Q-table, que crece con cada turno y cada partida.

---

### Group A — MCTS + UCB1 + Tree Reuse

```
act(board)
  │
  ├── Recuperar árbol del turno anterior    # reutiliza nodos ya explorados
  │     └── buscar tablero actual como "nieto" del root previo
  │
  └── Bucle MCTS hasta agotar tiempo
        ├── Selección (UCB1 + sesgo progresivo al centro)
        ├── Expansión (center-first)
        ├── Simulación (rollout aleatorio)
        └── Retropropagación
```

**Componentes clave:**
- **Tree reuse**: conserva el árbol entre turnos — los nodos explorados en el turno anterior siguen siendo válidos
- **Sesgo progresivo**: pondera UCB1 con `3 - |col - 3|` dividido por visitas, favorece el centro al inicio
- **Presupuesto de tiempo dinámico**: distribuye el tiempo restante entre los turnos estimados que quedan

---

## 3. Diferencia fundamental: ¿cómo "piensan"?

```
Minimax (Julian)              Cristian V2                   MCTS (Group A)
─────────────────             ───────────                   ──────────────
Explora árbol completo        Explora árbol de forma        Explora árbol de forma
hasta profundidad d           probabilística + Q-table      probabilística con UCB1

Sabe exactamente qué pasa     Estima valor por promedio     Estima valor por promedio
hasta d movimientos adelante  de rollouts guiados por       de miles de partidas
                              experiencia acumulada         simuladas aleatorias

Cada movimiento: árbol nuevo  Cada movimiento: árbol nuevo  Cada movimiento: reutiliza
(TT ayuda con pos. vistas)    pero Q-table persiste         árbol anterior (acumula)

Fuerte en táctica             Fuerte en táctica (heurística)Fuerte en estrategia
Débil como agresor proactivo  + Q-table mejora con partidas (exploración amplia)

No aprende entre partidas     Aprende entre partidas        No aprende entre partidas
                              (Q-table en disco)
```

---

## 4. Componente diferenciador de cada agente

| | Julian | Cristian V2 | Group A |
|---|---|---|---|
| **Componente único** | Heurística Allis + 69 ventanas precalculadas | Q-table persistente entre partidas + backprop alternante | Tree reuse entre turnos |
| **¿Qué lo hace mejor en el largo plazo?** | Mayor profundidad (depth) | Más partidas jugadas (Q-table crece) | Mayor tiempo por turno |
| **Memoria entre movimientos** | Tabla de transposiciones (en partida) | Q-table en disco (entre partidas) | Árbol reutilizado (en partida) |
| **vs aleatorio** | 100% | 100% | 100% |

---

## 5. Resultados empíricos

### Cristian V2 vs Cristian V1 (MCTS básico, sin Q-table)

| Configuración | WR Rojo (V2) | WR Amarillo (V2) | WR Total |
|---|---|---|---|
| V2 vs V1 · 20 partidas/color | **80%** | **75%** | **77.5%** |

> La Q-table acumulada aporta una ventaja medible del ~25% sobre el mismo algoritmo sin aprendizaje persistente.

### Cristian V2 vs Agente Aleatorio

| Rol | WR |
|---|---|
| Como Rojo | **100%** (20/20) |
| Como Amarillo | **100%** (20/20) |

### Cristian V2 vs Group A (Exp 9)

| Rol de V2 | W | D | L | WR |
|---|---|---|---|---|
| Como Rojo | 6 | 0 | 14 | 30% |
| Como Amarillo | 4 | 0 | 16 | 20% |
| **Total** | **10** | **0** | **30** | **25%** |

### Julian vs Cristian V2 (MCTS básico, referencia)

| Configuración | WR Rojo (Julian) | WR Amarillo (Julian) | WR Total |
|---|---|---|---|
| Julian d=4 vs Cristian 200 sims | 75% | 75% | 75% |
| Julian d=4 vs Cristian 500 sims | 75% | 75% | 75% |
| Julian d=5 vs Cristian 200 sims | 80% | 90% | 85% |
| Julian d=5 vs Cristian 500 sims | 75% | 95% | 85% |

### Julian vs Group A

| Rol de Julian | Resultado |
|---|---|
| Como Rojo (primer jugador) | Pierde ~90% |
| Como Amarillo (segundo jugador) | Gana ~90% |

---

## 6. Análisis del patrón Rojo vs Amarillo

### Por qué Julian gana como Amarillo

- **Minimax es reactivo por naturaleza**: cada nodo del árbol empieza desde el movimiento del oponente — "dado que jugaron X, ¿cuál es mi mejor respuesta?" es exactamente lo que Minimax optimiza
- **`immediate_tactic()` bloquea al instante**: si Group A crea una amenaza como Rojo, Julian la detecta antes de gastar búsqueda
- **MCTS como agresor es menos predecible**: los rollouts aleatorios no generan planes de ataque consistentes — Julian los anticipa con búsqueda profunda

### Por qué Julian pierde como Rojo

- **Tiene que ser proactivo**: crear amenazas de la nada requiere evaluar posiciones donde el oponente aún no respondió — el heurístico de ventanas no captura bien los planes a largo plazo
- **Tree reuse de Group A es acumulativo**: cuando Julian mueve como Rojo, Group A ya tiene miles de simulaciones del turno anterior. Julian reinicia la búsqueda en cada movimiento
- **El early game tiene el árbol más ancho** (b≈7, pocas fichas) — la poda alfa-beta es menos efectiva y el presupuesto de tiempo por movimiento es el mismo que en el endgame

### Cristian V2: comportamiento simétrico

- **Backpropagación alternante** elimina el sesgo Rojo/Amarillo que tenía V1: V2 obtiene ~50% como Rojo y ~47% como Amarillo en self-play (14-1-15), confirmando corrección adversarial.
- **La heurística táctica** garantiza que V2 no pierda por amenazas obvias en ningún rol.
- **La Q-table** no diferencia entre colores: almacena recompensas desde la perspectiva del **jugador que movió** en cada estado, haciendo el aprendizaje independiente del color asignado.

---

## 7. Fortalezas y debilidades

| | Julian | Cristian V2 | Group A |
|---|---|---|---|
| **Fortaleza** | Táctica profunda, bloqueo preciso, determinista | Aprendizaje persistente, simétrico, táctica inmediata | Exploración amplia, tree reuse, mejora con tiempo |
| **Debilidad** | Lento como agresor, early game limitado | Sin tree reuse: 500 sims fijas vs miles de Group A | Débil ante amenazas tácticas profundas |
| **Escala con** | Mayor profundidad (depth) | Más partidas jugadas + mayor presupuesto de sims | Mayor tiempo (más simulaciones por turno) |
| **Falla cuando** | El oponente crea planes que la heurística no valora | Q-table vacía vs oponente muy fuerte (partidas iniciales) | El oponente tiene táctica que los rollouts no detectan |
| **Memoria** | Intraturn (TT) | Inter-partida (Q-table pkl) | Intraturn (tree reuse) |

---

## 8. Gap de rendimiento: ¿por qué Cristian V2 pierde vs Group A?

El Exp 9 muestra que V2 obtiene solo 25% vs Group A. El análisis de factores:

| Factor | Group A | Cristian V2 | Impacto |
|---|---|---|---|
| Sims/turno efectivas | ~3000-5000 (tiempo × tree reuse) | 500 fijas | **6-10× más búsqueda** |
| Reutilización del árbol | Sí (nieto) | **No** | Trabajo previo no se pierde |
| Sesgo progresivo UCB | Sí | No (solo center-first) | Exploración inicial más informada |
| Q-table persistente | No | **Sí** | Único diferenciador a favor de V2 |

**Conclusión**: el cuello de botella de V2 es el presupuesto de búsqueda, no la calidad del rollout. La Q-table da ventaja real (+25% vs V1) pero no compensa la diferencia de 6-10× en simulaciones efectivas que tiene Group A gracias al tree reuse.

---

## 9. Conclusión general

Los tres agentes son competitivos pero con perfiles radicalmente distintos:

- **Julian domina en posiciones donde la táctica es decisiva** — amenazas en 2-3 movimientos, bloqueos, endgame. Es el más fuerte como defensor.
- **Cristian V2 es el único que aprende entre partidas** — con suficientes partidas previas contra oponentes fuertes, los rollouts se vuelven más informativos. Su ventaja crece con el tiempo.
- **Group A domina en el juego de apertura** — la exploración estocástica y el tree reuse acumulan ventaja estratégica antes de que Julian llegue a profundidad suficiente o que V2 tenga Q-table entrenada.

La mejora que más igualaría a V2 con Group A es la **reutilización del árbol**: con tree reuse, V2 pasaría de 500 sims fijas a ~3000-5000 efectivas, reduciendo el gap computacional de 6-10× a prácticamente 1×, manteniendo además el único diferenciador que Group A no tiene: la Q-table persistente entre partidas.
