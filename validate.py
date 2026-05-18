"""
Validación rápida del agente Cristian vs jugador aleatorio.
Corre desde tournament/:  python validate.py

Reporta:
  - Win-rate como Rojo (mueve primero, player=-1)
  - Win-rate como Amarillo (mueve segundo, player=1)
"""

import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy
from groups.Cristian.policy import Cristian


class RandomAgent(Policy):
    """Agente de referencia: elige columna válida al azar."""
    def mount(self): pass
    def act(self, s: np.ndarray) -> int:
        cols = [c for c in range(7) if s[0, c] == 0]
        return int(np.random.default_rng().choice(cols))


def play_game(red_policy, yellow_policy, seed=None):
    """Juega una partida y retorna el ganador (-1, 1 o 0=empate)."""
    rng = np.random.default_rng(seed)
    # red_policy y yellow_policy ya deben estar montados
    state = ConnectState()
    while not state.is_final():
        if state.player == -1:
            col = red_policy.act(state.board)
        else:
            col = yellow_policy.act(state.board)
        # Si la columna no es válida, elegir aleatoriamente (seguridad)
        if not state.is_applicable(col):
            col = int(rng.choice(state.get_free_cols()))
        state = state.transition(col)
    return state.get_winner()


def run_validation(n_games=100, simulations=500):
    print(f"{'='*55}")
    print(f"  Validación: Cristian (MCTS, sims={simulations}) vs Aleatorio")
    print(f"  {n_games} partidas por color")
    print(f"{'='*55}\n")

    cristian = Cristian(simulations=simulations)
    random_agent = RandomAgent()

    # ── Cristian como Rojo (mueve primero, player=-1) ──────────────────
    wins_red = draws_red = losses_red = 0
    for i in range(n_games):
        cristian.mount()
        random_agent.mount()
        result = play_game(red_policy=cristian, yellow_policy=random_agent)
        if result == -1:
            wins_red += 1
        elif result == 0:
            draws_red += 1
        else:
            losses_red += 1
        if (i + 1) % 20 == 0:
            print(f"  Rojo:     {i+1}/{n_games} partidas jugadas...")

    print(f"\n  Cristian como ROJO   (mueve primero):")
    print(f"    Victorias : {wins_red:>3} / {n_games}  ({wins_red/n_games*100:.1f}%)")
    print(f"    Empates   : {draws_red:>3} / {n_games}  ({draws_red/n_games*100:.1f}%)")
    print(f"    Derrotas  : {losses_red:>3} / {n_games}  ({losses_red/n_games*100:.1f}%)")

    # ── Cristian como Amarillo (mueve segundo, player=1) ───────────────
    wins_yel = draws_yel = losses_yel = 0
    print()
    for i in range(n_games):
        cristian.mount()
        random_agent.mount()
        result = play_game(red_policy=random_agent, yellow_policy=cristian)
        if result == 1:
            wins_yel += 1
        elif result == 0:
            draws_yel += 1
        else:
            losses_yel += 1
        if (i + 1) % 20 == 0:
            print(f"  Amarillo: {i+1}/{n_games} partidas jugadas...")

    print(f"\n  Cristian como AMARILLO (mueve segundo):")
    print(f"    Victorias : {wins_yel:>3} / {n_games}  ({wins_yel/n_games*100:.1f}%)")
    print(f"    Empates   : {draws_yel:>3} / {n_games}  ({draws_yel/n_games*100:.1f}%)")
    print(f"    Derrotas  : {losses_yel:>3} / {n_games}  ({losses_yel/n_games*100:.1f}%)")

    # ── Resumen ────────────────────────────────────────────────────────
    total_wins = wins_red + wins_yel
    total = n_games * 2
    print(f"\n{'='*55}")
    print(f"  TOTAL: {total_wins}/{total} victorias ({total_wins/total*100:.1f}%)")
    ok_no_losses = losses_red == 0 and losses_yel == 0
    ok_winrate = (wins_red / n_games) >= 0.5 and (wins_yel / n_games) >= 0.5
    print(f"  Requisito 'nunca pierde':  {'CUMPLIDO' if ok_no_losses else 'NO cumplido'}")
    print(f"  Requisito '>= 50% victorias': {'CUMPLIDO' if ok_winrate else 'NO cumplido'}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_validation(n_games=100, simulations=500)
