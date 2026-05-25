"""
Validación comparativa de los agentes de Cristian.

Secciones
---------
1. V1 (MCTS puro)       vs Aleatorio
2. V2 (MCTS + Q-table)  vs Aleatorio
3. V1 self-play          (V1 Rojo vs V1 Amarillo)
4. V2 self-play          (V2 Rojo vs V2 Amarillo)
5. V1 vs V2             (cross-play)

Uso:
    cd connect-four-agents
    python validate.py
"""

import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy
from groups.Cristian.policy import Cristian as CristianV1
from groups.Cristian.policy_qValues import Cristian as CristianV2


# ─────────────────────────────────────────────────────────────────────────────
# Agente de referencia
# ─────────────────────────────────────────────────────────────────────────────

class RandomAgent(Policy):
    """Elige columna válida al azar."""
    def mount(self, **_): pass
    def act(self, s: np.ndarray) -> int:
        cols = [c for c in range(7) if s[0, c] == 0]
        return int(np.random.default_rng().choice(cols))


# ─────────────────────────────────────────────────────────────────────────────
# Motor de partida
# ─────────────────────────────────────────────────────────────────────────────

def play_game(red_policy, yellow_policy):
    """Juega una partida completa. Retorna el ganador (-1, 1) o 0 si empate."""
    rng = np.random.default_rng()
    state = ConnectState()
    while not state.is_final():
        if state.player == -1:
            col = red_policy.act(state.board)
        else:
            col = yellow_policy.act(state.board)
        if not state.is_applicable(col):
            col = int(rng.choice(state.get_free_cols()))
        state = state.transition(col)
    return state.get_winner()


# ─────────────────────────────────────────────────────────────────────────────
# Bloque reutilizable: agente A vs agente B, N partidas por color
# ─────────────────────────────────────────────────────────────────────────────

def run_matchup(agent_a, agent_b, name_a, n_games=50, progress_every=10):
    """
    Juega n_games partidas donde agent_a es Rojo y luego n_games donde es Amarillo.
    Imprime tabla de resultados y retorna dict con estadísticas.
    """
    results = {}

    for role, my_player in [("ROJO", -1), ("AMARILLO", 1)]:
        wins = draws = losses = 0
        for i in range(n_games):
            agent_a.mount()
            agent_b.mount()
            if my_player == -1:
                w = play_game(red_policy=agent_a, yellow_policy=agent_b)
            else:
                w = play_game(red_policy=agent_b, yellow_policy=agent_a)

            if w == my_player:
                wins += 1
            elif w == 0:
                draws += 1
            else:
                losses += 1

            if (i + 1) % progress_every == 0:
                print(f"    {role}: {i+1}/{n_games} partidas...")

        wr = wins / n_games * 100
        print(f"\n  {name_a} como {role}:")
        print(f"    Victorias : {wins:>3}/{n_games}  ({wr:.1f}%)")
        print(f"    Empates   : {draws:>3}/{n_games}  ({draws/n_games*100:.1f}%)")
        print(f"    Derrotas  : {losses:>3}/{n_games}  ({losses/n_games*100:.1f}%)")
        results[role] = {"wins": wins, "draws": draws, "losses": losses, "n": n_games}

    return results


def print_section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def print_summary(results, n_games):
    r, y = results["ROJO"], results["AMARILLO"]
    total_w = r["wins"] + y["wins"]
    total   = n_games * 2
    no_loss = r["losses"] == 0 and y["losses"] == 0
    ok_wr   = (r["wins"] / n_games) >= 0.5 and (y["wins"] / n_games) >= 0.5
    print(f"\n  RESUMEN TOTAL: {total_w}/{total} victorias ({total_w/total*100:.1f}%)")
    print(f"  Nunca pierde:     {'SI' if no_loss else 'NO'}")
    print(f"  >= 50% victorias: {'SI' if ok_wr   else 'NO'}")


# ─────────────────────────────────────────────────────────────────────────────
# Secciones de validación
# ─────────────────────────────────────────────────────────────────────────────

def section_vs_random(n_games=50, simulations=500):
    """V1 y V2 contra el jugador aleatorio."""
    random_agent = RandomAgent()

    # V1 vs Aleatorio
    print_section(f"1 · V1 (MCTS puro, sims={simulations}) vs Aleatorio  [{n_games} partidas/color]")
    v1 = CristianV1(simulations=simulations)
    r1 = run_matchup(v1, random_agent, "V1", n_games)
    print_summary(r1, n_games)

    # V2 vs Aleatorio
    print_section(f"2 · V2 (MCTS+Q-table, sims={simulations}) vs Aleatorio  [{n_games} partidas/color]")
    v2 = CristianV2(simulations=simulations)
    r2 = run_matchup(v2, random_agent, "V2", n_games)
    print_summary(r2, n_games)


def section_self_play(n_games=30, simulations=500):
    """Cada versión juega contra sí misma."""

    # V1 self-play
    print_section(f"3 · V1 self-play  [{n_games} partidas/color]")
    v1a = CristianV1(simulations=simulations)
    v1b = CristianV1(simulations=simulations)
    wins_r = draws_r = losses_r = 0
    wins_y = draws_y = losses_y = 0
    for i in range(n_games):
        v1a.mount(); v1b.mount()
        w = play_game(red_policy=v1a, yellow_policy=v1b)
        if w == -1: wins_r += 1
        elif w == 0: draws_r += 1
        else: losses_r += 1
        if (i + 1) % 10 == 0:
            print(f"    ROJO: {i+1}/{n_games}...")
    for i in range(n_games):
        v1a.mount(); v1b.mount()
        w = play_game(red_policy=v1b, yellow_policy=v1a)
        if w == 1: wins_y += 1
        elif w == 0: draws_y += 1
        else: losses_y += 1
        if (i + 1) % 10 == 0:
            print(f"    AMARILLO: {i+1}/{n_games}...")
    print(f"\n  V1 como ROJO    — W:{wins_r} D:{draws_r} L:{losses_r}  ({wins_r/n_games*100:.1f}%)")
    print(f"  V1 como AMARILLO— W:{wins_y} D:{draws_y} L:{losses_y}  ({wins_y/n_games*100:.1f}%)")
    print(f"  (En self-play simétrico esperamos ~50% por color)")

    # V2 self-play
    print_section(f"4 · V2 self-play  [{n_games} partidas/color]")
    v2a = CristianV2(simulations=simulations)
    v2b = CristianV2(simulations=simulations)
    wins_r = draws_r = losses_r = 0
    wins_y = draws_y = losses_y = 0
    for i in range(n_games):
        v2a.mount(); v2b.mount()
        w = play_game(red_policy=v2a, yellow_policy=v2b)
        if w == -1: wins_r += 1
        elif w == 0: draws_r += 1
        else: losses_r += 1
        if (i + 1) % 10 == 0:
            print(f"    ROJO: {i+1}/{n_games}...")
    for i in range(n_games):
        v2a.mount(); v2b.mount()
        w = play_game(red_policy=v2b, yellow_policy=v2a)
        if w == 1: wins_y += 1
        elif w == 0: draws_y += 1
        else: losses_y += 1
        if (i + 1) % 10 == 0:
            print(f"    AMARILLO: {i+1}/{n_games}...")
    print(f"\n  V2 como ROJO    — W:{wins_r} D:{draws_r} L:{losses_r}  ({wins_r/n_games*100:.1f}%)")
    print(f"  V2 como AMARILLO— W:{wins_y} D:{draws_y} L:{losses_y}  ({wins_y/n_games*100:.1f}%)")
    print(f"  (En self-play simétrico esperamos ~50% por color)")


def section_v1_vs_v2(n_games=30, simulations=500):
    """V1 (MCTS puro) vs V2 (MCTS + Q-table)."""
    print_section(f"5 · V1 vs V2  [{n_games} partidas/color]")
    v1 = CristianV1(simulations=simulations)
    v2 = CristianV2(simulations=simulations)
    r = run_matchup(v2, v1, "V2", n_games)
    total_w = r["ROJO"]["wins"] + r["AMARILLO"]["wins"]
    total   = n_games * 2
    print(f"\n  V2 gana {total_w}/{total} partidas contra V1 ({total_w/total*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N   = 50   # partidas por color para vs-random y cross-play
    N_S = 30   # partidas por color para self-play (más rápido)
    SIMS = 500

    section_vs_random(n_games=N,   simulations=SIMS)
    section_self_play(n_games=N_S, simulations=SIMS)
    section_v1_vs_v2( n_games=N_S, simulations=SIMS)

    print(f"\n{'═'*60}")
    print("  Validación completa.")
    print(f"{'═'*60}\n")