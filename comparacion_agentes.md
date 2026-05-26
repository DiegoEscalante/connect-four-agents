# Comparación de Agentes — Julian vs Group A

**Fundamentos de Inteligencia Artificial 2026.1**

---

## 1. Paradigma

| Aspecto | Julian (Minimax + Allis) | Group A (MCTS) |
|---|---|---|
| **Tipo** | Búsqueda determinista + conocimiento | Simulación estocástica |
| **Fuente de valor** | Función heurística hecha a mano | Rollouts aleatorios |
| **Conocimiento del dominio** | Alto (ventanas, centro, paridad Allis) | Bajo (solo sesgo al centro) |
| **Parámetro principal** | `depth` (profundidad de búsqueda) | Tiempo por turno (simulaciones) |
| **Reproducibilidad** | Determinista — mismo tablero, mismo movimiento | Estocástico — varía por seed |

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
Minimax (Julian)                     MCTS (Group A)
─────────────────                    ──────────────
Explora árbol completo               Explora árbol de forma probabilística
hasta profundidad d                  guiado por UCB1

Sabe exactamente qué pasa            Estima valor por promedio
hasta d movimientos adelante         de miles de partidas simuladas

Cada movimiento: árbol nuevo         Cada movimiento: reutiliza árbol anterior
(TT ayuda con posiciones vistas)     (acumula conocimiento entre turnos)

Fuerte en táctica (ve amenazas)      Fuerte en estrategia (exploración amplia)
Débil como agresor proactivo         Débil bloqueando amenazas profundas
```

---

## 4. Resultados empíricos

### Julian vs Cristian (MCTS básico, sin tree reuse)

| Configuración | WR Rojo | WR Amarillo | WR Total |
|---|---|---|---|
| Julian d=4 vs Cristian 200 sims | 75% | 75% | 75% |
| Julian d=4 vs Cristian 500 sims | 75% | 75% | 75% |
| Julian d=5 vs Cristian 200 sims | 80% | 90% | 85% |
| Julian d=5 vs Cristian 500 sims | 75% | 95% | 85% |

### Julian vs Group A (MCTS con tree reuse + sesgo progresivo)

| Rol de Julian | Resultado |
|---|---|
| Como Rojo (primer jugador) | Pierde ~90% |
| Como Amarillo (segundo jugador) | Gana ~90% |

---

## 5. Análisis del patrón Rojo vs Amarillo

### Por qué Julian gana como Amarillo

- **Minimax es reactivo por naturaleza**: cada nodo del árbol empieza desde el movimiento del oponente — "dado que jugaron X, ¿cuál es mi mejor respuesta?" es exactamente lo que Minimax optimiza
- **`immediate_tactic()` bloquea al instante**: si Group A crea una amenaza como Rojo, Julian la detecta antes de gastar búsqueda
- **MCTS como agresor es menos predecible**: los rollouts aleatorios no generan planes de ataque consistentes — Julian los anticipa con búsqueda profunda

### Por qué Julian pierde como Rojo

- **Tiene que ser proactivo**: crear amenazas de la nada requiere evaluar posiciones donde el oponente aún no respondió — el heurístico de ventanas no captura bien los planes a largo plazo
- **Tree reuse de Group A es acumulativo**: cuando Julian mueve como Rojo, Group A ya tiene miles de simulaciones del turno anterior. Julian reinicia la búsqueda en cada movimiento (la TT ayuda con posiciones repetidas pero no con posiciones futuras del árbol)
- **El early game tiene el árbol más ancho** (b≈7, pocas fichas) — la poda alfa-beta es menos efectiva y el presupuesto de tiempo por movimiento es el mismo que en el endgame

---

## 6. Fortalezas y debilidades

| | Julian | Group A |
|---|---|---|
| **Fortaleza** | Táctica profunda, bloqueo preciso, determinista | Exploración amplia, mejora con tiempo, tree reuse |
| **Debilidad** | Lento como agresor, early game limitado | Débil ante amenazas tácticas profundas |
| **Escala con** | Mayor profundidad (depth) | Mayor tiempo (más simulaciones) |
| **Falla cuando** | El oponente crea planes que la heurística no valora | El oponente tiene táctica que los rollouts no detectan |

---

## 7. Conclusión

Ambos agentes son competitivos pero con perfiles opuestos:

- **Julian domina en posiciones donde la táctica es decisiva** — amenazas en 2-3 movimientos, bloqueos, endgame
- **Group A domina en el juego de apertura** donde la exploración estocástica y el tree reuse acumulan ventaja estratégica antes de que Julian llegue a profundidad suficiente

Esta asimetría Rojo/Amarillo confirma algo conocido en teoría de juegos: **Minimax con buena heurística es un defensor excelente, pero atacar bien requiere ya sea mayor profundidad o una función de evaluación que capture planes ofensivos de largo plazo**.

La diferencia entre los resultados contra Cristian (75-85% WR) y contra Group A (~50% WR global) se explica casi enteramente por el **tree reuse** — Group A lleva ventaja de exploración acumulada desde el turno 1, mientras que Cristian descarta su árbol en cada movimiento.
