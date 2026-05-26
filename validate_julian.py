"""
Validación rápida de Julian vs jugador aleatorio.
Corre desde la raíz del proyecto:
    python validate_julian.py
"""
import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy
from groups.Julian.policy import Julian


class RandomAgent(Policy):
    def mount(self): pass
    def act(self, s):
        cols = [c for c in range(7) if s[0, c] == 0]
        return int(np.random.default_rng().choice(cols))


def play_game(red, yellow):
    state = ConnectState()
    while not state.is_final():
        if state.player == -1:
            col = red.act(state.board)
        else:
            col = yellow.act(state.board)
        if not state.is_applicable(col):
            col = int(np.random.default_rng().choice(state.get_free_cols()))
        state = state.transition(col)
    return state.get_winner()


def run_validation(n_games=100, simulations=500, guided=True):
    julian = Julian(simulations=simulations, guided=guided)
    random_agent = RandomAgent()

    print(f"Julian (sims={simulations}, guided={guided}) vs Random")
    print(f"{n_games} partidas por color\n")

    # Como Red (mueve primero, player=-1)
    w_r = l_r = d_r = 0
    for i in range(n_games):
        julian.mount()
        random_agent.mount()
        r = play_game(julian, random_agent)
        if r == -1:
            w_r += 1
        elif r == 1:
            l_r += 1
        else:
            d_r += 1
        if (i + 1) % 20 == 0:
            print(f"  Rojo:     {i+1}/{n_games} partidas jugadas...")

    print(f"\n  Julian como ROJO   (mueve primero):")
    print(f"    Victorias : {w_r:>3} / {n_games}  ({w_r/n_games*100:.1f}%)")
    print(f"    Empates   : {d_r:>3} / {n_games}  ({d_r/n_games*100:.1f}%)")
    print(f"    Derrotas  : {l_r:>3} / {n_games}  ({l_r/n_games*100:.1f}%)")

    # Como Amarillo (mueve segundo, player=1)
    w_y = l_y = d_y = 0
    print()
    for i in range(n_games):
        julian.mount()
        random_agent.mount()
        r = play_game(random_agent, julian)
        if r == 1:
            w_y += 1
        elif r == -1:
            l_y += 1
        else:
            d_y += 1
        if (i + 1) % 20 == 0:
            print(f"  Amarillo: {i+1}/{n_games} partidas jugadas...")

    print(f"\n  Julian como AMARILLO (mueve segundo):")
    print(f"    Victorias : {w_y:>3} / {n_games}  ({w_y/n_games*100:.1f}%)")
    print(f"    Empates   : {d_y:>3} / {n_games}  ({d_y/n_games*100:.1f}%)")
    print(f"    Derrotas  : {l_y:>3} / {n_games}  ({l_y/n_games*100:.1f}%)")

    total_wins = w_r + w_y
    total = n_games * 2
    print(f"\n{'='*55}")
    print(f"  TOTAL: {total_wins}/{total} victorias ({total_wins/total*100:.1f}%)")
    ok_no_loss = l_r == 0 and l_y == 0
    ok_winrate = (w_r / n_games) >= 0.5 and (w_y / n_games) >= 0.5
    print(f"  Requisito 'nunca pierde':     {'CUMPLIDO' if ok_no_loss else 'NO cumplido'}")
    print(f"  Requisito '>=50% victorias':  {'CUMPLIDO' if ok_winrate else 'NO cumplido'}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_validation(n_games=100, simulations=500, guided=True)
