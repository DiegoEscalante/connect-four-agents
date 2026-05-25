import math
import pickle
import os
import numpy as np
from connect4.policy import Policy

ROWS = 6
COLS = 7

Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# Nodo del árbol MCTS
# ═══════════════════════════════════════════════════════════════════════════════

class _MCTSNode:
    """
    Nodo del árbol MCTS.

    wins/visits : acumulan desde la perspectiva del JUGADOR QUE MOVIÓ para
                  crear este nodo (node.parent.to_move). Esto hace que UCB1
                  sea correcto para ambos jugadores durante la selección.
    """

    __slots__ = ["board", "to_move", "parent", "action",
                 "children", "wins", "visits", "untried"]

    def __init__(self, board, to_move, parent=None, action=None):
        self.board    = board
        self.to_move  = to_move
        self.parent   = parent
        self.action   = action
        self.children = []
        self.wins     = 0.0
        self.visits   = 0
        valid = [c for c in range(COLS) if board[0, c] == 0]
        self.untried  = sorted(valid, key=lambda c: abs(c - COLS // 2), reverse=True)

    def ucb1(self, C):
        if self.visits == 0:
            return float("inf")
        return (self.wins / self.visits
                + C * math.sqrt(math.log(self.parent.visits) / self.visits))

    def is_fully_expanded(self):
        return len(self.untried) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Agente MCTS adversarial + Q-table persistente
# ═══════════════════════════════════════════════════════════════════════════════

class Cristian(Policy):
    """
    Agente Connect-4: MCTS/UCT adversarial con Q-table persistente entre partidas.

    Diferencias clave respecto a policy.py (V1 - MCTS puro)
    --------------------------------------------------------
    1. Q-table en disco (q_table.pkl): acumula experiencia de TODAS las
       partidas jugadas. A más partidas, los rollouts son más inteligentes.

    2. Rollouts guiados por Q-table: en lugar de jugar completamente aleatorio,
       selecciona acciones con probabilidad proporcional al Q-value acumulado
       (softmax simplificado). Se activa cuando la Q-table tiene > 100 entradas.

    3. Backpropagación alternante (corrección adversarial): cada nodo acumula
       la recompensa desde la perspectiva del JUGADOR QUE MOVIÓ para crear ese
       nodo. Así UCB1 es correcto para ambos jugadores: cuando es el turno del
       oponente, la selección elige el movimiento mejor para el oponente (peor
       para nosotros), modelando correctamente la adversarialidad.

    4. Q-update integrado en backpropagación: cada simulación MCTS actualiza
       la Q-table, que crece con cada partida jugada.

    Parámetros
    ----------
    simulations : int   → rollouts por movimiento (default 500)
    C           : float → constante de exploración UCB1 (default √2)
    q_guided    : bool  → activar rollouts guiados por Q-table (default True)
    save_every  : int   → guardar Q-table cada N movimientos (default 10)
    """

    def __init__(self, simulations=500, C=1.414, q_guided=True, save_every=10):
        self.simulations = simulations
        self.C           = C
        self.q_guided    = q_guided
        self.save_every  = save_every
        self._rng        = np.random.default_rng()
        self._move_count = 0
        # Clave : (board.tobytes(), col)
        # Valor : [suma_recompensas_del_mover, num_visitas]
        # La recompensa siempre se guarda desde la perspectiva del jugador
        # que movió en ese estado → válida para rojo y amarillo.
        self.q_table = {}

    # ── Persistencia ────────────────────────────────────────────────────────── #

    def mount(self, timeout=None):
        self._rng        = np.random.default_rng()
        self._move_count = 0
        self._load_q_table()

    def _load_q_table(self):
        if os.path.exists(Q_TABLE_PATH):
            try:
                with open(Q_TABLE_PATH, "rb") as f:
                    self.q_table = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                # Archivo corrupto (escritura interrumpida) → empezar vacío
                self.q_table = {}
                os.remove(Q_TABLE_PATH)
        else:
            self.q_table = {}

    def _save_q_table(self):
        try:
            with open(Q_TABLE_PATH, "wb") as f:
                pickle.dump(self.q_table, f)
        except OSError:
            pass  # fallo de I/O (ej. OneDrive lock) — no interrumpe la partida

    def _maybe_save(self):
        if self.save_every > 0 and self._move_count % self.save_every == 0:
            self._save_q_table()

    # ── Operaciones sobre la Q-table ─────────────────────────────────────────── #

    def _q_update(self, board, col, mover_reward):
        """Actualiza Q(board, col) con la recompensa del jugador que movió."""
        key = (board.tobytes(), col)
        if key not in self.q_table:
            self.q_table[key] = [0.0, 0]
        self.q_table[key][0] += mover_reward
        self.q_table[key][1] += 1

    def _q_value(self, board, col):
        """Retorna Q̂(board, col). Si es desconocido, retorna 0.5 (neutro)."""
        key = (board.tobytes(), col)
        if key not in self.q_table or self.q_table[key][1] == 0:
            return 0.5
        wins, visits = self.q_table[key]
        return wins / visits

    # ── Utilidades del tablero ───────────────────────────────────────────────── #

    def _infer_player(self, board):
        return -1 if np.sum(board == -1) == np.sum(board == 1) else 1

    def _valid_cols(self, board):
        return [c for c in range(COLS) if board[0, c] == 0]

    def _drop(self, board, col, player):
        b = board.copy()
        for r in range(ROWS - 1, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                break
        return b

    def _winner(self, board):
        for r in range(ROWS):
            for c in range(COLS):
                p = board[r, c]
                if p == 0:
                    continue
                if c + 3 < COLS and all(board[r, c + i] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and all(board[r + i, c] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and c + 3 < COLS and all(board[r + i, c + i] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and c - 3 >= 0 and all(board[r + i, c - i] == p for i in range(4)):
                    return p
        return 0

    def _is_terminal(self, board):
        return self._winner(board) != 0 or not self._valid_cols(board)

    # ── Fases MCTS ───────────────────────────────────────────────────────────── #

    def _select(self, node):
        """Selección UCB1. Con backpropagación alternante, UCB1 es correcto
        para ambos jugadores: cada nodo maximiza la ganancia del mover que lo creó."""
        while not self._is_terminal(node.board) and node.is_fully_expanded():
            node = max(node.children, key=lambda ch: ch.ucb1(self.C))
        return node

    def _expand(self, node):
        if self._is_terminal(node.board) or not node.untried:
            return node
        col       = node.untried.pop()
        new_board = self._drop(node.board, col, node.to_move)
        child     = _MCTSNode(new_board, -node.to_move, parent=node, action=col)
        node.children.append(child)
        return child

    def _simulate(self, board, to_move, me):
        """
        Rollout hasta fin del juego.
        Si q_guided=True y la Q-table tiene datos, usa Q-values para guiar la
        selección de acciones (probabilidad proporcional al Q-value del mover).
        Retorna: 1.0 si me gana, 0.0 si pierde, 0.5 si empate.
        """
        b = board.copy()
        p = to_move
        for _ in range(ROWS * COLS):
            valid = np.where(b[0] == 0)[0]
            if len(valid) == 0:
                return 0.5
            if self.q_guided and len(self.q_table) > 100:
                # Q-value desde la perspectiva de p (el jugador que mueve)
                q_vals = np.array([self._q_value(b, int(c)) for c in valid], dtype=float)
                q_vals = q_vals - q_vals.min() + 1e-6
                probs  = q_vals / q_vals.sum()
                col    = int(self._rng.choice(valid, p=probs))
            else:
                col = int(self._rng.choice(valid))
            for r in range(ROWS - 1, -1, -1):
                if b[r, col] == 0:
                    b[r, col] = p
                    break
            w = self._winner(b)
            if w != 0:
                return 1.0 if w == me else 0.0
            p = -p
        return 0.5

    def _backpropagate(self, node, result):
        """
        Backpropagación con recompensas alternantes (corrección adversarial).

        result : win rate desde la perspectiva del JUGADOR QUE MOVIÓ para crear
                 este nodo (node.parent.to_move). Se invierte en cada nivel
                 para que cada padre acumule desde su propia perspectiva.

        Efecto: node.wins/visits siempre refleja el win rate del mover que
        creó ese nodo → UCB1 es correcto para ambos jugadores en _select.
        """
        while node is not None:
            node.visits += 1
            node.wins   += result
            if node.action is not None and node.parent is not None:
                self._q_update(node.parent.board, node.action, result)
            result = 1.0 - result  # perspectiva opuesta para el padre
            node = node.parent

    # ── Interfaz pública ─────────────────────────────────────────────────────── #

    def act(self, s):
        me  = self._infer_player(s)
        opp = -me
        self._move_count += 1
        valid = self._valid_cols(s)

        # Prioridad 1 — ganar ahora (recompensa 1.0; guardar porque el juego termina)
        for col in valid:
            if self._winner(self._drop(s, col, me)) == me:
                self._q_update(s, col, 1.0)
                self._save_q_table()
                return col

        # Prioridad 2 — bloquear al oponente (recompensa 0.75: crítico pero no es victoria)
        for col in valid:
            if self._winner(self._drop(s, col, opp)) == opp:
                self._q_update(s, col, 0.75)
                self._maybe_save()
                return col

        # MCTS con backpropagación alternante
        root = _MCTSNode(s.copy(), me)
        for _ in range(self.simulations):
            leaf   = self._select(root)
            child  = self._expand(leaf)
            reward = self._simulate(child.board, child.to_move, me)
            # Convertir reward (perspectiva de 'me') a result (perspectiva del mover
            # que creó 'child', es decir child.parent.to_move = -child.to_move).
            result = reward if child.to_move != me else 1.0 - reward
            self._backpropagate(child, result)

        if not root.children:
            return valid[0]

        best = max(root.children, key=lambda ch: ch.visits)
        self._maybe_save()
        return best.action