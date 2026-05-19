import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
import math
import time
from typing import Optional

class MCTSNode:
    def __init__(self, state: ConnectState, parent: Optional['MCTSNode'] = None, action_taken: Optional[int] = None):
        self.state: ConnectState = state
        self.parent: Optional['MCTSNode'] = parent
        self.action_taken: Optional[int] = action_taken
        self.children: list['MCTSNode'] = []
        self.untried_actions: list[int] = state.get_free_cols()
        self.visits: int = 0
        self.wins: int = 0
    
    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    @property
    def is_final(self) -> bool:
        return self.state.is_final()

class MCTSAgent(Policy):
    def __init__(self, total_game_time: float = 60.0, exploration_constant: float = math.sqrt(2)):
        self.total_game_time = total_game_time
        self.exploration_constant = exploration_constant
        self.max_total_turns = 21 
        self.total_time_remaining = total_game_time
        self.agent_rng = np.random.default_rng()
        self.action_timeout = 10.0 
        
    def mount(self, timeout: Optional[float] = None) -> None:
        if timeout is not None:
            self.action_timeout = timeout
        self.total_time_remaining = self.total_game_time    

    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        '''Fase 1: Selección de nodo a expandir usando UCB'''
        best_score = -float('inf')
        best_children: list[MCTSNode] = []

        for child in node.children:
            exploitation = child.wins / child.visits
            exploration = self.exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
            ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_children = [child]
            elif math.isclose(ucb_score, best_score):
                best_children.append(child)

        return self.agent_rng.choice(best_children)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        '''Fase 2: Expansión de un nodo no completamente expandido'''
        action = node.untried_actions.pop(self.agent_rng.integers(len(node.untried_actions)))
        next_state = node.state.transition(action)
        child_node = MCTSNode(state=next_state, parent=node, action_taken=action)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state: ConnectState) -> int:
        '''Fase 3: Simulación de una partida aleatoria desde el nodo expandido'''
        current_state = state
        while not current_state.is_final():
            actions = current_state.get_free_cols()
            if not actions:
                break
            action = self.agent_rng.choice(actions)           
            current_state = current_state.transition(int(action))
        return current_state.get_winner()

    def _backpropagate(self, node: MCTSNode, winner: int) -> None:
        '''Fase 4: Retropropagación de los resultados de la simulación a lo largo del camino recorrido'''
        curr_node: Optional[MCTSNode] = node
        while curr_node is not None:
            curr_node.visits += 1
            
            if winner == 0:
                curr_node.wins += 0.5
            else:
                if curr_node.parent is not None:
                    if curr_node.parent.state.player == winner:
                        curr_node.wins += 1.0
                    else:
                        curr_node.wins += 0.0
                else:
                    if curr_node.state.player == winner:
                        curr_node.wins += 1.0

            curr_node = curr_node.parent

    def act(self, s: np.ndarray) -> int:
        start_turn_time = time.time()

        # 1. Identificar fichas y detectar si es un juego nuevo para resetear el reloj
        num_chips_on_board = np.sum(s != 0)
        
        # Si el tablero está vacío (0) o tiene 1 ficha (juego nuevo siendo amarillo),
        # reiniciamos el pool de tiempo para esta partida específica.
        if num_chips_on_board <= 1:
            self.total_time_remaining = self.total_game_time

        my_player_id = -1 if num_chips_on_board % 2 == 0 else 1

        # Calcular cuántos turnos quedan estimadamente
        my_turns_played = num_chips_on_board // 2
        turns_remaining_estimated = max(1, self.max_total_turns - my_turns_played)

        # 2. Presupuestar el tiempo para este turno
        safe_time_pool = max(0.2, self.total_time_remaining - 1.5)
        time_allocated_for_this_turn = safe_time_pool / turns_remaining_estimated

        # Ajuste estricto contra el timeout por turno del autograder
        max_safe_turn_time = max(0.2, self.action_timeout - 1.5)
        time_allocated_for_this_turn = min(max_safe_turn_time, time_allocated_for_this_turn)

        # Inicialización del árbol
        root_state = ConnectState(board=s.copy(), player=my_player_id)
        root_node = MCTSNode(state=root_state)

        legal_actions = root_state.get_free_cols()
        if not legal_actions:
            return 0
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        simulations_count = 0

        # 3. Ejecutar simulaciones
        while (time.time() - start_turn_time) < time_allocated_for_this_turn:
            node = root_node

            # Fase 1: Selección
            while not node.is_final and node.is_fully_expanded:
                node = self._ucb_select(node)

            # Fase 2: Expansión
            if not node.is_final and not node.is_fully_expanded:
                node = self._expand(node)

            # Fase 3: Simulación
            winner = self._simulate(node.state)

            # Fase 4: Retropropagación
            self._backpropagate(node, winner)
            simulations_count += 1
        
        # 4. Descontar el tiempo real consumido
        elapsed_time = time.time() - start_turn_time
        self.total_time_remaining -= elapsed_time

        print(f"Turno: {my_turns_played} | Usado: {elapsed_time:.3f}s | Simulaciones: {simulations_count} | Banco Restante: {self.total_time_remaining:.2f}s")

        # Si por alguna razón el árbol no generó hijos (ej. el tablero ya era un estado final),
        # evitamos que max() falle devolviendo una acción legal por defecto.
        if not root_node.children:
            return legal_actions[0]

        # Seleccionar el movimiento más robusto
        best_child = max(root_node.children, key=lambda child: child.visits)
        return int(best_child.action_taken)