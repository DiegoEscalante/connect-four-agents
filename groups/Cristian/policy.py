import math
import numpy as np
from connect4.policy import Policy

ROWS = 6
COLS = 7


# ═══════════════════════════════════════════════════════════════════════════════
# Nodo del árbol MCTS
# ═══════════════════════════════════════════════════════════════════════════════

class _MCTSNode:
    """
    Nodo del árbol de búsqueda Monte Carlo.

    Atributos
    ---------
    board    : estado del tablero en este nodo
    to_move  : jugador que debe mover (-1=Red, 1=Yellow)
    parent   : nodo padre (None para la raíz)
    action   : columna jugada para llegar a este nodo
    children : nodos hijo expandidos
    wins     : recompensa acumulada (victorias del agente)
    visits   : veces que este nodo fue visitado
    untried  : columnas aún no expandidas (orden center-first)
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
        """UCB1 = explotación + C·exploración (UCT estándar)."""
        if self.visits == 0:
            return float("inf")
        return (self.wins / self.visits
                + C * math.sqrt(math.log(self.parent.visits) / self.visits))

    def is_fully_expanded(self):
        return len(self.untried) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Agente MCTS
# ═══════════════════════════════════════════════════════════════════════════════

class Cristian(Policy):
    """
    Agente de Connect-4 basado en Monte Carlo Tree Search (MCTS / UCT).

    Algoritmo (4 fases por simulación)
    ------------------------------------
    1. Selección   : recorre el árbol con UCB1 hasta un nodo no completamente
                     expandido o terminal.
    2. Expansión   : agrega un hijo por la primera columna sin explorar
                     (orden center-first para mejor poda).
    3. Simulación  : juega aleatoriamente hasta el final del juego (rollout).
    4. Retropropag.: actualiza wins/visits en todos los ancestros.

    Tras las N simulaciones, retorna la columna del hijo más visitado
    (criterio más robusto que el de mayor win-rate).

    Conexión con el curso
    ---------------------
    * UCB1 (UCT) = la misma fórmula de exploración-explotación vista en
      bandits de múltiples brazos (Hoja 8-9).
    * El árbol de estados alternantes es el MDP competitivo de la Hoja 12.

    Parámetros
    ----------
    simulations : int
        Número de rollouts por movimiento. Más = mejor juego, más lento.
        500 equilibra calidad y velocidad (~0.05 s/movimiento).
    C : float
        Constante de exploración de UCB1. 1.414 (sqrt(2)) es el valor estándar.
    """

    def __init__(self, simulations=500, C=1.414):
        self.simulations = simulations
        self.C = C
        self._rng = np.random.default_rng()

    def mount(self, timeout=None):
        # timeout: límite de tiempo por movimiento inyectado por Gradescope; no se usa aquí
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Utilidades del tablero
    # ------------------------------------------------------------------ #

    def _infer_player(self, board):
        """
        Infiere el color propio a partir del conteo de fichas.
        Red (-1) siempre mueve primero, por lo que si los conteos
        son iguales es el turno de Red (y eso somos nosotros).
        """
        return -1 if np.sum(board == -1) == np.sum(board == 1) else 1

    def _valid_cols(self, board):
        return [c for c in range(COLS) if board[0, c] == 0]

    def _drop(self, board, col, player):
        """Devuelve un nuevo tablero con la ficha de player en col."""
        b = board.copy()
        for r in range(ROWS - 1, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                break
        return b

    def _winner(self, board):
        """Retorna el jugador ganador (-1, 1) o 0 si no hay ganador."""
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

    # ------------------------------------------------------------------ #
    # Fases MCTS
    # ------------------------------------------------------------------ #

    def _select(self, node):
        """Fase 1: baja por el árbol con UCB1 hasta un nodo expansible."""
        while not self._is_terminal(node.board) and node.is_fully_expanded():
            node = max(node.children, key=lambda ch: ch.ucb1(self.C))
        return node

    def _expand(self, node):
        """Fase 2: agrega un hijo para la siguiente columna sin explorar."""
        if self._is_terminal(node.board) or not node.untried:
            return node
        col = node.untried.pop()
        new_board = self._drop(node.board, col, node.to_move)
        child = _MCTSNode(new_board, -node.to_move, parent=node, action=col)
        node.children.append(child)
        return child

    def _simulate(self, board, to_move, me):
        """
        Fase 3: rollout aleatorio hasta el fin del juego.
        Retorna 1.0 si gana `me`, 0.0 si pierde, 0.5 si empata.
        """
        b = board.copy()
        p = to_move
        for _ in range(ROWS * COLS):
            valid = np.where(b[0] == 0)[0]
            if len(valid) == 0:
                return 0.5
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

    def _backpropagate(self, node, reward):
        """Fase 4: actualiza wins/visits desde el nodo hasta la raíz."""
        while node is not None:
            node.visits += 1
            node.wins   += reward
            node = node.parent

    def _immediate_action(self, board, me):
        """
        Detección táctica de un paso: ganar ahora o bloquear al oponente.
        Prioridad 1 — victoria propia: si existe columna que me da 4 en línea, jugarla.
        Prioridad 2 — bloqueo: si el oponente ganaría en su próximo turno, bloquearlo.
        Retorna la columna urgente o None si no hay acción inmediata.
        """
        opp   = -me
        valid = self._valid_cols(board)

        # Prioridad 1: ¿puedo ganar ya?
        for col in valid:
            if self._winner(self._drop(board, col, me)) == me:
                return col

        # Prioridad 2: ¿el oponente gana en su siguiente turno?
        for col in valid:
            if self._winner(self._drop(board, col, opp)) == opp:
                return col

        return None

    # ------------------------------------------------------------------ #
    # Interfaz publica
    # ------------------------------------------------------------------ #

    def act(self, s):
        me = self._infer_player(s)

        # Acción táctica inmediata (O(cols) — antes de gastar simulaciones)
        tactical = self._immediate_action(s, me)
        if tactical is not None:
            return tactical

        root = _MCTSNode(s.copy(), me)

        for _ in range(self.simulations):
            leaf   = self._select(root)
            child  = self._expand(leaf)
            reward = self._simulate(child.board, child.to_move, me)
            self._backpropagate(child, reward)

        if not root.children:
            return self._valid_cols(s)[0]

        # Hijo más visitado = elección más robusta
        best = max(root.children, key=lambda ch: ch.visits)
        return best.action