import time
import math
import random as _rnd
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Motor Minimax+Allis (de Julian) — privado, sin heredar Policy
# ─────────────────────────────────────────────────────────────

ROWS = 6
COLS = 7
CENTER_COL = 3
AB_INF = 10_000_000

WIN_SCORE   = 100_000
THREE_SCORE = 50
TWO_SCORE   = 10
OPP_THREE   = -60
OPP_TWO     = -10

_rnd.seed(42)
_ZOBRIST: dict = {
    (r, c, p): _rnd.getrandbits(64)
    for r in range(ROWS)
    for c in range(COLS)
    for p in (1, -1)
}
_TT: dict = {}

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


def _drop(board, col, player):
    b = board.copy()
    for r in range(ROWS - 1, -1, -1):
        if b[r, col] == 0:
            b[r, col] = player
            return b, r
    return b, -1

def _valid_cols(board):
    return [c for c in range(COLS) if board[0, c] == 0]

def _infer_player(board):
    reds = int(np.sum(board == -1))
    yellows = int(np.sum(board == 1))
    return -1 if reds == yellows else 1

def _winner_at(board, r, c, player):
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

def _immediate_tactic(board, me):
    opp = -me
    for col in _valid_cols(board):
        b, r = _drop(board, col, me)
        if r >= 0 and _winner_at(b, r, col, me):
            return col
    for col in _valid_cols(board):
        b, r = _drop(board, col, opp)
        if r >= 0 and _winner_at(b, r, col, opp):
            return col
    return None

def _board_hash(board):
    h = 0
    for r in range(ROWS):
        for c in range(COLS):
            p = int(board[r, c])
            if p != 0:
                h ^= _ZOBRIST[(r, c, p)]
    return h

def _is_playable(board, r, c):
    return board[r, c] == 0 and (r == ROWS - 1 or board[r + 1, c] != 0)

def _evaluate(board, me, w_center, w_odd_even, use_allis):
    opp = -me
    score = w_center * int(np.sum(board[:, CENTER_COL] == me))
    me_good_parity  = 1 if me  == -1 else 0
    opp_good_parity = 1 if opp == -1 else 0
    for cells in _WINDOWS:
        r0, c0 = cells[0]; r1, c1 = cells[1]; r2, c2 = cells[2]; r3, c3 = cells[3]
        v0 = board[r0, c0]; v1 = board[r1, c1]; v2 = board[r2, c2]; v3 = board[r3, c3]
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
                er, ec_col = (r0,c0) if v0==0 else (r1,c1) if v1==0 else (r2,c2) if v2==0 else (r3,c3)
                if _is_playable(board, er, ec_col) and (er % 2 == me_good_parity):
                    score += w_odd_even
        elif mc == 2 and ec == 2:
            score += TWO_SCORE
        elif oc == 3 and ec == 1:
            score += OPP_THREE
            if use_allis:
                er, ec_col = (r0,c0) if v0==0 else (r1,c1) if v1==0 else (r2,c2) if v2==0 else (r3,c3)
                if _is_playable(board, er, ec_col) and (er % 2 == opp_good_parity):
                    score -= w_odd_even
        elif oc == 2 and ec == 2:
            score += OPP_TWO
    return score

_COL_ORDER = sorted(range(COLS), key=lambda c: abs(c - CENTER_COL))

class _Timeout(Exception):
    pass

def _ordered_cols(board):
    return [c for c in _COL_ORDER if board[0, c] == 0]

def _minimax(board, depth, alpha, beta, maximizing, me,
             w_center, w_odd_even, use_allis,
             last_row=-1, last_col=-1, last_player=0, deadline=None, z_hash=0):
    if deadline is not None and time.time() > deadline:
        raise _Timeout()
    if last_row >= 0 and _winner_at(board, last_row, last_col, last_player):
        gain = WIN_SCORE + depth
        return gain if last_player == me else -gain
    cols = _valid_cols(board)
    if not cols:
        return 0
    if z_hash in _TT:
        tt_d, tt_s = _TT[z_hash]
        if tt_d >= depth:
            return tt_s
    if depth == 0:
        score = _evaluate(board, me, w_center, w_odd_even, use_allis)
        _TT[z_hash] = (0, score)
        return score
    if maximizing:
        value = -AB_INF
        for col in _ordered_cols(board):
            child, lr = _drop(board, col, me)
            ch = z_hash ^ _ZOBRIST[(lr, col, int(me))]
            value = max(value, _minimax(child, depth-1, alpha, beta, False, me,
                                        w_center, w_odd_even, use_allis, lr, col, me, deadline, ch))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        _TT[z_hash] = (depth, value)
        return value
    else:
        opp = -me
        value = AB_INF
        for col in _ordered_cols(board):
            child, lr = _drop(board, col, opp)
            ch = z_hash ^ _ZOBRIST[(lr, col, int(opp))]
            value = min(value, _minimax(child, depth-1, alpha, beta, True, me,
                                        w_center, w_odd_even, use_allis, lr, col, opp, deadline, ch))
            beta = min(beta, value)
            if alpha >= beta:
                break
        _TT[z_hash] = (depth, value)
        return value


class _MinimaxEngine:
    def __init__(self, depth=10, w_center=6, w_odd_even=20, use_allis=True, time_limit=55.0):
        self.depth      = depth
        self.w_center   = w_center
        self.w_odd_even = w_odd_even
        self.use_allis  = use_allis
        self.time_limit = time_limit
        self._start_time = None

    def mount(self, timeout=None):
        self._start_time = time.time()
        global _TT
        _TT = {}

    def act(self, s):
        me = _infer_player(s)
        tac = _immediate_tactic(s, me)
        if tac is not None:
            return tac
        elapsed = time.time() - self._start_time
        remaining = max(0.5, self.time_limit - elapsed)
        pieces = int(np.sum(s != 0))
        my_moves_left = max(1, (42 - pieces + 1) // 2)
        move_budget = (remaining / my_moves_left) * 0.85
        deadline = time.time() + move_budget
        zh = _board_hash(s)
        best_col = _ordered_cols(s)[0]
        for d in range(1, self.depth + 1):
            try:
                candidate_score = -AB_INF
                candidate_col   = _ordered_cols(s)[0]
                for col in _ordered_cols(s):
                    child, lr = _drop(s, col, me)
                    ch = zh ^ _ZOBRIST[(lr, col, int(me))]
                    score = _minimax(child, d-1, -AB_INF, AB_INF, False, me,
                                     self.w_center, self.w_odd_even, self.use_allis,
                                     lr, col, me, deadline, ch)
                    if score > candidate_score:
                        candidate_score = score
                        candidate_col   = col
                best_col = candidate_col
            except _Timeout:
                break
        return best_col


# ─────────────────────────────────────────────────────────────
# Motor MCTS (de Grupo A) — privado, sin heredar Policy
# ─────────────────────────────────────────────────────────────

class _MCTSNode:
    def __init__(self, state: ConnectState, parent: Optional['_MCTSNode'] = None,
                 action_taken: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: list['_MCTSNode'] = []
        self.untried_actions: list[int] = sorted(state.get_free_cols(), key=lambda col: abs(3 - col))
        self.visits: int = 0
        self.wins: float = 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_final(self) -> bool:
        return self.state.is_final()


class _MCTSEngine:
    def __init__(self, total_game_time: float = 60.0,
                 exploration_constant: float = math.sqrt(2)):
        self.total_game_time = total_game_time
        self.exploration_constant = exploration_constant
        self.max_total_turns = 21
        self.total_time_remaining = total_game_time
        self.agent_rng = np.random.default_rng()
        self.action_timeout = 10.0
        self.persisted_root: Optional[_MCTSNode] = None

    def mount(self, timeout: Optional[float] = None) -> None:
        if timeout is not None:
            self.action_timeout = timeout
        self.total_time_remaining = self.total_game_time
        self.persisted_root = None

    def _ucb_select(self, node: _MCTSNode) -> _MCTSNode:
        best_score = -float('inf')
        best_children: list[_MCTSNode] = []
        for child in node.children:
            exploitation = child.wins / child.visits
            exploration = self.exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
            peso_centro = 3 - abs(3 - child.action_taken)
            heuristic_bias = (peso_centro * 0.5) / (child.visits + 1)
            ucb_score = exploitation + exploration + heuristic_bias
            if ucb_score > best_score:
                best_score = ucb_score
                best_children = [child]
            elif math.isclose(ucb_score, best_score):
                best_children.append(child)
        return self.agent_rng.choice(best_children)

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        action = node.untried_actions.pop(0)
        next_state = node.state.transition(action)
        child_node = _MCTSNode(state=next_state, parent=node, action_taken=action)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state: ConnectState) -> int:
        current_state = state
        while not current_state.is_final():
            actions = current_state.get_free_cols()
            if not actions:
                break
            current_state = current_state.transition(int(self.agent_rng.choice(actions)))
        return current_state.get_winner()

    def _backpropagate(self, node: _MCTSNode, winner_id: int) -> None:
        curr = node
        while curr is not None:
            curr.visits += 1
            if winner_id == 0:
                curr.wins += 0.5
            else:
                if curr.parent is not None:
                    if curr.parent.state.player == winner_id:
                        curr.wins += 1.0
                else:
                    if curr.state.player == winner_id:
                        curr.wins += 1.0
            curr = curr.parent

    def _find_matching_child(self, parent_node: _MCTSNode, target_board: np.ndarray) -> Optional[_MCTSNode]:
        for child in parent_node.children:
            if np.array_equal(child.state.board, target_board):
                return child
        return None

    def act(self, s: np.ndarray) -> int:
        start = time.time()
        num_chips = int(np.sum(s != 0))

        if num_chips <= 1:
            self.total_time_remaining = self.total_game_time
            self.persisted_root = None

        my_player_id = -1 if num_chips % 2 == 0 else 1
        my_turns_played = num_chips // 2
        turns_remaining = max(1, self.max_total_turns - my_turns_played)

        safe_pool = max(0.2, self.total_time_remaining - 1.5)
        turn_time = min(max(0.2, self.action_timeout - 1.5), safe_pool / turns_remaining)

        root_state = ConnectState(board=s.copy(), player=my_player_id)
        root_node = None

        if self.persisted_root is not None:
            for prev_move in self.persisted_root.children:
                found = self._find_matching_child(prev_move, s)
                if found:
                    root_node = found
                    root_node.parent = None
                    break

        if root_node is None:
            root_node = _MCTSNode(state=root_state)

        legal_actions = root_state.get_free_cols()
        if not legal_actions:
            return 0
        if len(legal_actions) == 1:
            return legal_actions[0]

        while (time.time() - start) < turn_time:
            node = root_node
            while not node.is_final and node.is_fully_expanded:
                node = self._ucb_select(node)
            if not node.is_final and not node.is_fully_expanded:
                node = self._expand(node)
            winner_id = self._simulate(node.state)
            self._backpropagate(node, winner_id)

        self.total_time_remaining -= time.time() - start
        self.persisted_root = root_node

        if not root_node.children:
            return legal_actions[0]
        return int(max(root_node.children, key=lambda c: c.visits).action_taken)


# ─────────────────────────────────────────────────────────────
# Agente híbrido público
# MCTS (Grupo A) de primero — Minimax+Allis (Julian) de segundo
# ─────────────────────────────────────────────────────────────

class Combinado(Policy):
    """
    Agente híbrido: usa MCTS cuando juega primero y Minimax+Allis cuando juega segundo.
    """

    def __init__(self):
        self._first  = _MCTSEngine(total_game_time=60.0)
        self._second = _MinimaxEngine(depth=10, w_center=6, w_odd_even=20,
                                      use_allis=True, time_limit=55.0)
        self._active = None

    def mount(self, timeout=None):
        self._first.mount(timeout)
        self._second.mount(timeout)
        self._active = None

    def act(self, s: np.ndarray) -> int:
        if self._active is None:
            # 0 fichas → tablero vacío → somos el primer jugador
            self._active = self._first if int(np.sum(s != 0)) == 0 else self._second
        return self._active.act(s)
