"""
Cristian V2 (MCTS + Q-table)  vs  Grupo A (MCTS + tree-reuse + prog-bias)

Uso:
    cd connect-four-agents
    python validate_vs_groupA.py
"""

import contextlib
import importlib.util
import io
import os
import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy
from groups.Cristian.policy_qValues import Cristian as CristianV2

# Carga dinámica porque la carpeta "Group A" tiene espacio en el nombre
_spec = importlib.util.spec_from_file_location(
    "group_a_policy",
    os.path.join(os.path.dirname(__file__), "groups", "Group A", "policy.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GroupA = _mod.MCTSAgent


# ─────────────────────────────────────────────────────────────────────────────
# Motor de partida  (silencia los prints internos del Grupo A)
# ─────────────────────────────────────────────────────────────────────────────

def play_game(red, yellow):
    """Juega una partida completa; retorna ganador (-1, 1) o 0 si empate."""
    rng   = np.random.default_rng()
    state = ConnectState()
    while not state.is_final():
        with contextlib.redirect_stdout(io.StringIO()):   # silencia prints internos
            if state.player == -1:
                col = red.act(state.board)
            else:
                col = yellow.act(state.board)
        if not state.is_applicable(col):
            col = int(rng.choice(state.get_free_cols()))
        state = state.transition(col)
    return state.get_winner()


# ─────────────────────────────────────────────────────────────────────────────
# Bloque de enfrentamiento
# ─────────────────────────────────────────────────────────────────────────────

def run_matchup(agent_a, agent_b, name_a, name_b, n_games=20, progress=5):
    """
    Juega n_games con agent_a como Rojo y n_games como Amarillo.
    Imprime resultados y retorna dict con estadísticas.
    """
    results = {}
    for role, my_player in [("ROJO", -1), ("AMARILLO", 1)]:
        w = d = l = 0
        for i in range(n_games):
            agent_a.mount()
            agent_b.mount()
            if my_player == -1:
                winner = play_game(red=agent_a, yellow=agent_b)
            else:
                winner = play_game(red=agent_b, yellow=agent_a)

            if winner == my_player:   w += 1
            elif winner == 0:         d += 1
            else:                     l += 1

            if (i + 1) % progress == 0:
                print(f"    {role}: {i+1}/{n_games}...")

        pct = w / n_games * 100
        print(f"\n  {name_a} como {role}:")
        print(f"    Victorias : {w:>3}/{n_games}  ({pct:.1f}%)")
        print(f"    Empates   : {d:>3}/{n_games}  ({d/n_games*100:.1f}%)")
        print(f"    Derrotas  : {l:>3}/{n_games}  ({l/n_games*100:.1f}%)")
        results[role] = {"wins": w, "draws": d, "losses": l}

    total_w = results["ROJO"]["wins"] + results["AMARILLO"]["wins"]
    total   = n_games * 2
    print(f"\n  TOTAL {name_a}: {total_w}/{total} victorias ({total_w/total*100:.1f}%)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N    = 20     # partidas por color (sube a 30 si tienes tiempo)
    SEP  = "═" * 58

    cristian = CristianV2(simulations=500)
    grupo_a  = GroupA()

    print(f"\n{SEP}")
    print(f"  Cristian V2  vs  Grupo A  [{N} partidas/color]")
    print(SEP)
    run_matchup(cristian, grupo_a, "Cristian V2", "Grupo A", n_games=N)

    print(f"\n{SEP}")
    print("  (invertido) Grupo A  vs  Cristian V2")
    print(SEP)
    run_matchup(grupo_a, cristian, "Grupo A", "Cristian V2", n_games=N)

    print(f"\n{SEP}")
    print("  Validación completa.")
    print(SEP)
