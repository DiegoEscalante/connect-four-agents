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
        # Pre-ordenamos las acciones para intentar siempre las del centro primero (Mejora Pasiva)
        free_cols = state.get_free_cols()
        self.untried_actions: list[int] = sorted(free_cols, key=lambda col: abs(3 - col))
        self.visits: int = 0
        self.wins: float = 0.0
    
    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    @property
    def is_final(self) -> bool:
        return self.state.is_final()

class MCTSAgent(Policy):
    def __init__(self, total_game_time: float = 60.0, exploration_constant: float = math.sqrt(2), config: dict = None):
        self.total_game_time = total_game_time
        self.exploration_constant = exploration_constant
        self.max_total_turns = 21 
        self.total_time_remaining = total_game_time
        self.agent_rng = np.random.default_rng()
        self.action_timeout = 10.0 
        
        # --- SISTEMA DE INTERRUPTORES (TOGGLES) ---
        if config is None:
            config = {}
        self.use_heuristic = config.get('heuristic', False)
        self.use_reuse_tree = config.get('reuse_tree', True)
        self.use_prog_bias = config.get('prog_bias', True)
        
        # Memoria persistente para reutilización del árbol
        self.persisted_root: Optional[MCTSNode] = None
        self.last_simulations_count = 0

    def mount(self, timeout: Optional[float] = None) -> None:
        if timeout is not None:
            self.action_timeout = timeout
        self.total_time_remaining = self.total_game_time    
        self.persisted_root = None # Limpiamos la memoria al montar un nuevo torneo

    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        '''Fase 1: Selección con Sesgo Progresivo opcional'''
        best_score = -float('inf')
        best_children: list[MCTSNode] = []

        for child in node.children:
            exploitation = child.wins / child.visits
            exploration = self.exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
            
            # --- INTERRUPTOR: Sesgo Progresivo ---
            heuristic_bias = 0.0
            if self.use_prog_bias:
                # Premiar jugar en el centro (columna 3), dividiendo por las visitas para que 
                # el sesgo desaparezca conforme el MCTS tiene datos reales.
                peso_centro = 3 - abs(3 - child.action_taken) # Da 3 para el centro, 0 para los bordes
                heuristic_bias = (peso_centro * 0.5) / (child.visits + 1)
                
            ucb_score = exploitation + exploration + heuristic_bias

            if ucb_score > best_score:
                best_score = ucb_score
                best_children = [child]
            elif math.isclose(ucb_score, best_score):
                best_children.append(child)

        return self.agent_rng.choice(best_children)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        '''Fase 2: Expansión (sacando siempre la mejor columna disponible primero gracias al sort)'''
        action = node.untried_actions.pop(0) # Siempre saca la más cercana al centro
        next_state = node.state.transition(action)
        child_node = MCTSNode(state=next_state, parent=node, action_taken=action)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state: ConnectState) -> int:
        '''Fase 3: Simulación con opción Heurística o Aleatoria Pura'''
        current_state = state
        
        while not current_state.is_final():
            actions = current_state.get_free_cols()
            if not actions:
                break
            
            chosen_action = None
            
            # --- INTERRUPTOR: Simulación Heurística ---
            if self.use_heuristic:
                current_player = current_state.player
                opponent = -current_player
                
                # 1. ¿Puedo ganar ya?
                for action in actions:
                    next_s = current_state.transition(action)
                    if next_s.is_final() and next_s.get_winner() == current_player:
                        chosen_action = action
                        break
                
                # 2. ¿Debo bloquear?
                if chosen_action is None:
                    opponent_state = ConnectState(board=current_state.board.copy(), player=opponent)
                    for action in actions:
                        next_s = opponent_state.transition(action)
                        if next_s.is_final() and next_s.get_winner() == opponent:
                            chosen_action = action
                            break
            
            # Si la heurística está apagada, o no encontró nada crítico, juega aleatorio
            if chosen_action is None:
                chosen_action = self.agent_rng.choice(actions)
                
            current_state = current_state.transition(int(chosen_action))
            
        return current_state.get_winner()

    def _backpropagate(self, node: MCTSNode, winner: int) -> None:
        '''Fase 4: Retropropagación'''
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
                    if curr_node.state.player == winner:
                        curr_node.wins += 1.0

            curr_node = curr_node.parent
            
    def _find_matching_child(self, parent_node: MCTSNode, target_board: np.ndarray) -> Optional[MCTSNode]:
        """Busca en los hijos inmediatos un tablero que coincida con el actual."""
        for child in parent_node.children:
            if np.array_equal(child.state.board, target_board):
                return child
        return None

    def act(self, s: np.ndarray) -> int:
        start_turn_time = time.time()
        num_chips_on_board = np.sum(s != 0)
        
        # Reinicio de partida
        if num_chips_on_board <= 1:
            self.total_time_remaining = self.total_game_time
            self.persisted_root = None

        my_player_id = -1 if num_chips_on_board % 2 == 0 else 1
        my_turns_played = num_chips_on_board // 2
        turns_remaining_estimated = max(1, self.max_total_turns - my_turns_played)

        # Cálculo de tiempo
        safe_time_pool = max(0.2, self.total_time_remaining - 1.5)
        time_allocated_for_this_turn = safe_time_pool / turns_remaining_estimated
        max_safe_turn_time = max(0.2, self.action_timeout - 1.5)
        time_allocated_for_this_turn = min(max_safe_turn_time, time_allocated_for_this_turn)

        root_state = ConnectState(board=s.copy(), player=my_player_id)
        root_node = None

        # --- INTERRUPTOR: Reutilización de Árbol ---
        if self.use_reuse_tree and self.persisted_root is not None:
            # Dado que ha pasado nuestro turno anterior y el del rival, el estado actual 
            # debe ser un "nieto" de nuestro root anterior. Lo buscamos.
            for mi_movimiento_anterior in self.persisted_root.children:
                hijo_encontrado = self._find_matching_child(mi_movimiento_anterior, s)
                if hijo_encontrado:
                    root_node = hijo_encontrado
                    root_node.parent = None # Cortamos la relación hacia arriba para liberar memoria RAM
                    break

        # Si no usamos reutilización, o el estado no se encontró, creamos raíz nueva
        if root_node is None:
            root_node = MCTSNode(state=root_state)

        legal_actions = root_state.get_free_cols()
        if not legal_actions:
            return 0
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        simulations_count = 0

        # MCTS Bucle Principal
        while (time.time() - start_turn_time) < time_allocated_for_this_turn:
            node = root_node
            while not node.is_final and node.is_fully_expanded:
                node = self._ucb_select(node)
            if not node.is_final and not node.is_fully_expanded:
                node = self._expand(node)
            winner = self._simulate(node.state)
            self._backpropagate(node, winner)
            simulations_count += 1
        self.last_simulations_count = simulations_count
        elapsed_time = time.time() - start_turn_time
        self.total_time_remaining -= elapsed_time

        # Guardar estado para el próximo turno si está activa la reutilización
        if self.use_reuse_tree:
            self.persisted_root = root_node

        # print(f"Turno: {my_turns_played} | Usado: {elapsed_time:.3f}s | Simul: {simulations_count} | Nodos reciclados: {'Sí' if self.use_reuse_tree and root_node.visits > simulations_count else 'No'}")

        if not root_node.children:
            return legal_actions[0]

        best_child = max(root_node.children, key=lambda child: child.visits)
        return int(best_child.action_taken)
    
