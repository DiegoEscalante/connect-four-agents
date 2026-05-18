import math
import numpy as np
from connect4.policy import Policy

ROWS = 6
COLS = 7
CENTER_WEIGHTS = [1, 2, 3, 4, 3, 2, 1]


def drop(board, col, player):
    """Devuelve un nuevo tablero con la ficha de player puesta en col."""
    new_board = board.copy()
    for r in range(ROWS - 1, -1, -1):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            break
    return new_board


def winner(board):
    """Devuelve -1, 1 o 0 si nadie ha ganado todavía."""
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r, c]
            if p == 0:
                continue
            if c + 3 < COLS and all(board[r, c+i] == p for i in range(4)):
                return p
            if r + 3 < ROWS and all(board[r+i, c] == p for i in range(4)):
                return p
            if r + 3 < ROWS and c + 3 < COLS and all(board[r+i, c+i] == p for i in range(4)):
                return p
            if r + 3 < ROWS and c - 3 >= 0 and all(board[r+i, c-i] == p for i in range(4)):
                return p
    return 0


def valid_cols(board):
    return [c for c in range(COLS) if board[0, c] == 0]


def is_terminal(board):
    return winner(board) != 0 or len(valid_cols(board)) == 0


def infer_player(board):
    """Red (-1) mueve primero. Si hay igual número de fichas, le toca a Red."""
    reds = np.sum(board == -1)
    yellows = np.sum(board == 1)
    if reds == yellows:
        return -1
    else:
        return 1


def immediate_tactic(board, me):
    """Si hay victoria o bloqueo en 1 movimiento, retorna la columna. Si no, None."""
    opp = -me
    # Si puedo ganar ya, gano
    for col in valid_cols(board):
        if winner(drop(board, col, me)) == me:
            return col
    # Si el oponente gana en su siguiente turno, lo bloqueo
    for col in valid_cols(board):
        if winner(drop(board, col, opp)) == opp:
            return col
    return None


def guided_action(board, player, rng):
    """
    Default policy guiada (slide 26 deck 13 + slide 21 deck 12).
    Reglas:
      1. Si puedo ganar en este movimiento, juego ahí.
      2. Si el oponente gana en su siguiente turno, lo bloqueo.
      3. Si no, muestreo proporcional a los pesos del centro.
    """
    valid = valid_cols(board)
    opp = -player

    # Regla 1: ganar
    for col in valid:
        if winner(drop(board, col, player)) == player:
            return col

    # Regla 2: bloquear
    for col in valid:
        if winner(drop(board, col, opp)) == opp:
            return col

    # Regla 3: muestreo ponderado por centro
    weights = np.array([CENTER_WEIGHTS[c] for c in valid], dtype=float)
    probs = weights / weights.sum()
    return int(rng.choice(valid, p=probs))


def random_action(board, rng):
    """Default policy aleatoria (para la versión guided=False, comparación)."""
    valid = valid_cols(board)
    return int(rng.choice(valid))


class Node:
    def __init__(self, board, to_move, parent=None, action=None):
        self.board = board
        self.to_move = to_move      # jugador que debe mover desde aquí (-1 o 1)
        self.parent = parent
        self.action = action         # columna que llevó a este nodo
        self.children = []
        self.value_sum = 0.0         # suma de rewards desde perspectiva de to_move
        self.visits = 0
        self.untried = valid_cols(board)

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def ucb_score(self, C):
        """
        UCB1 desde la perspectiva del padre.
        El padre ya movió; ahora le toca al jugador to_move de este hijo.
        El valor promedio del hijo está desde la perspectiva de to_move,
        pero el padre quiere maximizar SU valor, que es el opuesto.
        """
        if self.visits == 0:
            return float('inf')
        avg_value = self.value_sum / self.visits
        exploitation = -avg_value
        exploration = C * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class Julian(Policy):
    def __init__(self, simulations=500, C=1.414, guided=True):
        self.simulations = simulations
        self.C = C
        self.guided = guided
        self.rng = np.random.default_rng()

    def mount(self):
        self.rng = np.random.default_rng()

    def select(self, node):
        while not is_terminal(node.board) and node.is_fully_expanded():
            node = max(node.children, key=lambda ch: ch.ucb_score(self.C))
        return node

    def expand(self, node):
        if is_terminal(node.board) or not node.untried:
            return node
        col = node.untried.pop()
        new_board = drop(node.board, col, node.to_move)
        child = Node(new_board, -node.to_move, parent=node, action=col)
        node.children.append(child)
        return child

    def simulate(self, board, to_move):
        b = board.copy()
        p = to_move
        while True:
            w = winner(b)
            if w != 0:
                return w
            if len(valid_cols(b)) == 0:
                return 0
            if self.guided:
                col = guided_action(b, p, self.rng)
            else:
                col = random_action(b, self.rng)
            b = drop(b, col, p)
            p = -p

    def backpropagate(self, node, game_winner):
        """
        Sube por el árbol guardando el valor desde la perspectiva del
        jugador que mueve en cada nodo (slide 17 deck 12).
        """
        cur = node
        while cur is not None:
            cur.visits += 1
            if game_winner == 0:
                reward = 0.0
            elif cur.to_move == game_winner:
                # El jugador que iba a mover desde aquí terminó ganando
                reward = 1.0
            else:
                reward = -1.0
            cur.value_sum += reward
            cur = cur.parent

    def act(self, s):
        me = infer_player(s)

        # Atajo táctico (no es la diferencia conceptual, solo ahorra simulaciones)
        tac = immediate_tactic(s, me)
        if tac is not None:
            return tac

        root = Node(board=s.copy(), to_move=me)

        for _ in range(self.simulations):
            leaf = self.select(root)
            child = self.expand(leaf)
            game_winner = self.simulate(child.board, child.to_move)
            self.backpropagate(child, game_winner)

        # Hijo más visitado = decisión más robusta
        if not root.children:
            return valid_cols(s)[0]
        best = max(root.children, key=lambda ch: ch.visits)
        return best.action
