import time
import random as _rnd
import numpy as np
from connect4.policy import Policy

ROWS = 6
COLS = 7
CENTER_COL = 3
AB_INF = 10_000_000

WIN_SCORE   = 100_000
THREE_SCORE = 50
TWO_SCORE   = 10
OPP_THREE   = -60   # bloquear amenaza del oponente vale más que atacar
OPP_TWO     = -10

# ─────────────────────────────────────────────────────────────
# Tabla de transposiciones con Zobrist hashing.
# _ZOBRIST[(fila, col, jugador)] es un entero de 64 bits aleatorio y fijo.
# El hash de un tablero es el XOR de los valores de cada ficha colocada.
# XOR es invertible: agregar/quitar una ficha = XOR de su valor (O(1)).
# _TT mapea hash → (profundidad, score); se limpia al inicio de cada partida.
# ─────────────────────────────────────────────────────────────
_rnd.seed(42)
_ZOBRIST: dict = {
    (r, c, p): _rnd.getrandbits(64)
    for r in range(ROWS)
    for c in range(COLS)
    for p in (1, -1)
}
_TT: dict = {}


# ─────────────────────────────────────────────────────────────
# Todas las ventanas de 4 celdas precalculadas (69 en total).
# Cada ventana es una tupla de 4 pares (row, col).
# Se calculan una sola vez al importar el módulo.
# ─────────────────────────────────────────────────────────────
_WINDOWS: list[tuple] = []

for _r in range(ROWS):
    for _c in range(COLS - 3):
        _WINDOWS.append(((_r, _c), (_r, _c+1), (_r, _c+2), (_r, _c+3)))

for _c in range(COLS):
    for _r in range(ROWS - 3):
        _WINDOWS.append(((_r, _c), (_r+1, _c), (_r+2, _c), (_r+3, _c)))

for _r in range(ROWS - 3):
    for _c in range(COLS - 3):
        _WINDOWS.append(((_r, _c), (_r+1, _c+1), (_r+2, _c+2), (_r+3, _c+3)))

for _r in range(3, ROWS):
    for _c in range(COLS - 3):
        _WINDOWS.append(((_r, _c), (_r-1, _c+1), (_r-2, _c+2), (_r-3, _c+3)))


# ─────────────────────────────────────────────────────────────
# Utilidades del tablero
# ─────────────────────────────────────────────────────────────

def drop(board, col, player):
    b = board.copy()
    for r in range(ROWS - 1, -1, -1):
        if b[r, col] == 0:
            b[r, col] = player
            return b, r        # devuelve tablero Y fila donde aterrizó la ficha
    return b, -1


def valid_cols(board):
    return [c for c in range(COLS) if board[0, c] == 0]


def infer_player(board):
    reds = int(np.sum(board == -1))
    yellows = int(np.sum(board == 1))
    return -1 if reds == yellows else 1


def winner_at(board, r, c, player):
    """Chequeo incremental: ¿player tiene 4 en línea pasando por (r, c)?
    O(28) en vez de O(168) del escaneo completo."""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            nr, nc = r + sign * dr, c + sign * dc
            while 0 <= nr < ROWS and 0 <= nc < COLS and board[nr, nc] == player:
                count += 1
                nr += sign * dr
                nc += sign * dc
        if count >= 4:
            return True
    return False


def winner(board):
    """Escaneo completo — solo para el atajo táctico inicial."""
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


def immediate_tactic(board, me):
    opp = -me
    for col in valid_cols(board):
        b, r = drop(board, col, me)
        if r >= 0 and winner_at(b, r, col, me):
            return col
    for col in valid_cols(board):
        b, r = drop(board, col, opp)
        if r >= 0 and winner_at(b, r, col, opp):
            return col
    return None


def board_hash(board):
    """Hash Zobrist del tablero: XOR de _ZOBRIST[(r,c,player)] por cada ficha."""
    h = 0
    for r in range(ROWS):
        for c in range(COLS):
            p = int(board[r, c])
            if p != 0:
                h ^= _ZOBRIST[(r, c, p)]
    return h


# ─────────────────────────────────────────────────────────────
# Heurística de evaluación (ventanas + centro + Allis fusionados)
# ─────────────────────────────────────────────────────────────

def is_playable(board, r, c):
    """Casilla vacía y soportada (fila inferior o celda debajo ocupada)."""
    return board[r, c] == 0 and (r == ROWS - 1 or board[r + 1, c] != 0)


def evaluate(board, me, w_center, w_odd_even, use_allis):
    """
    Función de evaluación heurística con tres componentes:

    1. Ventanas de 4: puntúa las 69 ventanas del tablero según fichas propias/oponente.
    2. Sesgo al centro: las fichas en la columna central valen más (más ventanas disponibles).
    3. Teoría de Allis (odd/even threats): bonus por amenazas propias en la paridad correcta.
       - Red (me=-1) prefiere amenazas en filas IMPARES (fila_allis = ROWS-r_np es impar ↔ r_np impar).
       - Yellow (me=+1) prefiere amenazas en filas PARES (r_np par).
    """
    opp = -me
    score = w_center * int(np.sum(board[:, CENTER_COL] == me))

    # Parity preferred row-index parity for each player
    # fila_allis = ROWS - r_np  →  fila_allis impar ↔ r_np impar
    me_good_parity  = 1 if me  == -1 else 0   # r_np % 2 == me_good_parity → preferred
    opp_good_parity = 1 if opp == -1 else 0

    for cells in _WINDOWS:
        r0, c0 = cells[0]
        r1, c1 = cells[1]
        r2, c2 = cells[2]
        r3, c3 = cells[3]

        v0 = board[r0, c0]
        v1 = board[r1, c1]
        v2 = board[r2, c2]
        v3 = board[r3, c3]

        mc = (v0 == me) + (v1 == me) + (v2 == me) + (v3 == me)
        oc = (v0 == opp) + (v1 == opp) + (v2 == opp) + (v3 == opp)
        ec = 4 - mc - oc

        if mc == 4:
            score += WIN_SCORE
        elif oc == 4:
            score -= WIN_SCORE
        elif mc == 3 and ec == 1:
            score += THREE_SCORE
            if use_allis:
                # Encuentra la casilla vacía y aplica bonus de paridad
                if v0 == 0:
                    er, ec_col = r0, c0
                elif v1 == 0:
                    er, ec_col = r1, c1
                elif v2 == 0:
                    er, ec_col = r2, c2
                else:
                    er, ec_col = r3, c3
                if is_playable(board, er, ec_col) and (er % 2 == me_good_parity):
                    score += w_odd_even
        elif mc == 2 and ec == 2:
            score += TWO_SCORE
        elif oc == 3 and ec == 1:
            score += OPP_THREE
            if use_allis:
                if v0 == 0:
                    er, ec_col = r0, c0
                elif v1 == 0:
                    er, ec_col = r1, c1
                elif v2 == 0:
                    er, ec_col = r2, c2
                else:
                    er, ec_col = r3, c3
                if is_playable(board, er, ec_col) and (er % 2 == opp_good_parity):
                    score -= w_odd_even
        elif oc == 2 and ec == 2:
            score += OPP_TWO

    return score


# ─────────────────────────────────────────────────────────────
# Búsqueda Minimax con poda Alfa-Beta + Iterative Deepening
# ─────────────────────────────────────────────────────────────

# Orden center-first fijo: [3,2,4,1,5,0,6]
_COL_ORDER = sorted(range(COLS), key=lambda c: abs(c - CENTER_COL))


class _Timeout(Exception):
    pass


def ordered_cols(board):
    return [c for c in _COL_ORDER if board[0, c] == 0]


def minimax(board, depth, alpha, beta, maximizing, me,
            w_center, w_odd_even, use_allis,
            last_row=-1, last_col=-1, last_player=0, deadline=None, z_hash=0):
    """
    Minimax con poda alfa-beta, chequeo incremental de victoria y TT.

    z_hash: hash Zobrist del tablero actual (para TT lookup/store en O(1)).
    """
    if deadline is not None and time.time() > deadline:
        raise _Timeout()

    if last_row >= 0 and winner_at(board, last_row, last_col, last_player):
        gain = WIN_SCORE + depth
        return gain if last_player == me else -gain

    cols = valid_cols(board)
    if not cols:
        return 0

    # Transposition table: si ya evaluamos esta posición a igual o mayor depth, reusar.
    if z_hash in _TT:
        tt_d, tt_s = _TT[z_hash]
        if tt_d >= depth:
            return tt_s

    if depth == 0:
        score = evaluate(board, me, w_center, w_odd_even, use_allis)
        _TT[z_hash] = (0, score)
        return score

    if maximizing:
        value = -AB_INF
        for col in ordered_cols(board):
            child, lr = drop(board, col, me)
            ch = z_hash ^ _ZOBRIST[(lr, col, int(me))]
            value = max(value, minimax(
                child, depth - 1, alpha, beta, False, me,
                w_center, w_odd_even, use_allis, lr, col, me, deadline, ch
            ))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        _TT[z_hash] = (depth, value)
        return value
    else:
        opp = -me
        value = AB_INF
        for col in ordered_cols(board):
            child, lr = drop(board, col, opp)
            ch = z_hash ^ _ZOBRIST[(lr, col, int(opp))]
            value = min(value, minimax(
                child, depth - 1, alpha, beta, True, me,
                w_center, w_odd_even, use_allis, lr, col, opp, deadline, ch
            ))
            beta = min(beta, value)
            if alpha >= beta:
                break
        _TT[z_hash] = (depth, value)
        return value


# ─────────────────────────────────────────────────────────────
# Agente
# ─────────────────────────────────────────────────────────────

class Julian(Policy):
    """
    Agente de Connect-4 basado en Minimax con poda Alfa-Beta, heurística
    inspirada en la teoría de Victor Allis (1988), e Iterative Deepening
    con límite de tiempo por partida.

    Paradigma: knowledge-based (búsqueda determinista + evaluación hecha a mano).
    Contrasta con MCTS, que es estocástico y no usa conocimiento del dominio.

    Parámetros
    ----------
    depth      : profundidad máxima de búsqueda (fallback si no hay tiempo).
    w_center   : peso del sesgo al centro (columna 3 vale más).
    w_odd_even : peso del bonus de paridad de Allis.
    use_allis  : activar/desactivar la teoría de amenazas pares/impares.
    time_limit : presupuesto total de tiempo por partida en segundos.
    """

    def __init__(self, depth=10, w_center=6, w_odd_even=20, use_allis=True,
                 time_limit=55.0):
        self.depth      = depth
        self.w_center   = w_center
        self.w_odd_even = w_odd_even
        self.use_allis  = use_allis
        self.time_limit = time_limit
        self._start_time = None

    def mount(self, timeout=None):
        self._start_time = time.time()
        global _TT
        _TT = {}   # nueva partida → limpiar cache

    def act(self, s):
        me = infer_player(s)

        # Atajo táctico O(cols): ganar o bloquear antes de gastar búsqueda
        tac = immediate_tactic(s, me)
        if tac is not None:
            return tac

        # Calcular presupuesto de tiempo para este movimiento
        elapsed = time.time() - self._start_time
        remaining = max(0.5, self.time_limit - elapsed)
        pieces = int(np.sum(s != 0))
        my_moves_left = max(1, (42 - pieces + 1) // 2)
        move_budget = (remaining / my_moves_left) * 0.85   # margen de seguridad 15%

        deadline = time.time() + move_budget

        # Iterative deepening: depth=1,2,3,... hasta agotar tiempo o depth máximo
        zh = board_hash(s)   # hash Zobrist del tablero actual (se pasa a minimax)
        best_col = ordered_cols(s)[0]

        for d in range(1, self.depth + 1):
            try:
                candidate_score = -AB_INF
                candidate_col   = ordered_cols(s)[0]
                for col in ordered_cols(s):
                    child, lr = drop(s, col, me)
                    ch = zh ^ _ZOBRIST[(lr, col, int(me))]
                    score = minimax(
                        child, d - 1, -AB_INF, AB_INF,
                        False, me, self.w_center, self.w_odd_even, self.use_allis,
                        lr, col, me, deadline, ch
                    )
                    if score > candidate_score:
                        candidate_score = score
                        candidate_col   = col
                best_col = candidate_col   # depth d completado: actualizar
            except _Timeout:
                break   # usar el mejor de la profundidad anterior

        return best_col
