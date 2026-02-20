"""
Stone Age AlphaZero — MCTS + Training Pipeline

This module contains the complete MCTS search, self-play, replay buffer,
and training loop. It depends on two interfaces that are NOT implemented here:

  1. GameInterface  — wraps the C game engine (or a Python fallback)
  2. NeuralNetwork  — wraps the PyTorch model

Both are abstract base classes with thorough docstrings explaining exactly
what each method must do, its inputs, outputs, and invariants.

Usage:
    game_factory = ...   # something that returns GameInterface instances
    nn = ...             # a NeuralNetwork instance
    config = AlphaZeroConfig()
    trainer = AlphaZeroTrainer(config, game_factory, nn)
    trainer.run()
"""

from __future__ import annotations

import math
import time
import random
import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ════════════════════════════════════════════════════════════════════

@dataclass
class AlphaZeroConfig:
    """All hyperparameters in one place."""

    # --- Game ---
    num_players: int = 2
    action_space_size: int = 409        # fixed flat action space
    observation_size: int = 424         # int32 elements per observation
    max_chance_outcomes: int = 40       # upper bound on chance branching

    # --- MCTS ---
    num_simulations: int = 400          # simulations per move
    c_puct: float = 1.5                 # exploration constant in PUCT
    dirichlet_alpha: float = 0.3        # noise alpha at root
    dirichlet_fraction: float = 0.25    # fraction of noise mixed into prior
    temperature_move_threshold: int = 30 # after this many moves, temp → 0

    # --- Self-Play ---
    num_self_play_games: int = 100      # games per iteration
    max_game_moves: int = 1000          # safety cap on game length (kills stragglers)

    # --- Training ---
    num_iterations: int = 200           # total AlphaZero iterations
    num_epochs: int = 10                # training epochs per iteration
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    replay_buffer_size: int = 200_000   # max training examples stored
    min_replay_size: int = 2000         # don't train until this many examples

    # --- Evaluation ---
    eval_games: int = 40                # games to evaluate new model
    eval_threshold: float = 0.55        # win rate needed to replace best

    # --- Infrastructure ---
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10              # log every N self-play games
    seed: int = 42


# ════════════════════════════════════════════════════════════════════
# §2  GAME INTERFACE  (to be implemented by C wrapper or Python fallback)
# ════════════════════════════════════════════════════════════════════

class GameInterface(ABC):
    """
    Abstract interface to the Stone Age game engine.

    Every method below must be implemented. The implementation will
    typically wrap a C library via ctypes, but could also wrap the
    existing Python game code for testing.

    LIFECYCLE
    ---------
    A GameInterface instance represents one mutable game state.
    It is created by a factory (see GameFactory), cloned for MCTS
    simulations, and freed when garbage collected.

    PLAYER INDEXING
    ---------------
    Players are always indexed 0..num_players-1. The "current player"
    is whoever must make the next decision (or whose perspective the
    observation is from).

    ACTION IDS
    ----------
    Actions are integers in [0, action_space_size). At any decision
    point, only a subset are legal. The engine must handle all legal
    action IDs correctly and raise/assert on illegal ones.

    CHANCE NODES
    ------------
    When the game reaches a point requiring randomness (dice rolls),
    it enters a "chance node" state. No player acts; instead, an
    outcome must be sampled and applied. The MCTS tree handles this
    differently from decision nodes (no PUCT, no NN eval — just
    sample according to known probabilities).
    """

    # ------------------------------------------------------------------
    # Cloning
    # ------------------------------------------------------------------

    @abstractmethod
    def clone(self) -> GameInterface:
        """
        Return a deep copy of this game state.

        PERFORMANCE: This is the single hottest call in MCTS. For the C
        backend, this should be a malloc + memcpy of the ~500-byte
        GameState struct. No Python object graph traversal.

        The clone is fully independent — mutating it must not affect
        the original, and vice versa.

        Returns:
            A new GameInterface with identical state.
        """
        ...

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """
        Return the game state as a 1-D int32 numpy array, from the
        perspective of the current player.

        The array layout matches stone_age_state.py's get_state():
          - Section 0: Game meta            [4]
          - Section 1: Phase state          [37]
          - Section 2: Board locations      [32]
          - Section 3: Board misc           [5]
          - Section 4: Civ card slots       [8]
          - Section 5: Building slots       [8]
          - Section 6: Player states        [4 × PLAYER_BLOCK]
          - Section 7: Civ deck remaining   [34]
          - Section 8: Building stacks      [4]
          Total: ~416 int32 elements

        ROTATION: Player data is rotated so that index 0 always
        corresponds to the current player. This ensures the NN sees
        a consistent "self vs opponents" layout regardless of which
        seat it's playing from.

        Returns:
            np.ndarray of shape (observation_size,), dtype int32.
        """
        ...

    # ------------------------------------------------------------------
    # Legal Actions
    # ------------------------------------------------------------------

    @abstractmethod
    def legal_actions(self) -> np.ndarray:
        """
        Return the set of legal action IDs at the current decision point.

        This is only valid when is_chance_node() is False and
        is_terminal() is False.

        CONTEXT-DEPENDENT: The legal actions depend on the current
        sub-phase. For example:
          - PLACEMENT phase → location+count combos
          - TOOL_DECISION → which tools to use, or "done"
          - BUILDING_PAYMENT → resource combos that match the building
          - CARD_PAYMENT → resource combos summing to the card cost
          - ANY_RESOURCE_CHOICE → wood/clay/stone/gold
          - POTPOURRI_PICK → which of the rolled bonuses to take
          - FREE_CARD_CHOICE → which card slot to take for free
          - FEEDING → resource-for-food combos or "accept penalty"

        Returns:
            np.ndarray of shape (N,), dtype int32, where N >= 1.
            Each element is in [0, action_space_size).
            No duplicates. Unordered.
        """
        ...

    @abstractmethod
    def legal_actions_mask(self) -> np.ndarray:
        """
        Return a boolean mask of size action_space_size.

        mask[a] = True  iff action a is legal right now.
        mask[a] = False otherwise.

        This is equivalent to:
            mask = np.zeros(action_space_size, dtype=bool)
            mask[self.legal_actions()] = True
        but the implementation may compute it more efficiently.

        Used by the NN to mask the policy head output before softmax.

        Returns:
            np.ndarray of shape (action_space_size,), dtype bool.
        """
        ...

    # ------------------------------------------------------------------
    # State Transitions
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_action(self, action_id: int) -> None:
        """
        Apply a player's action to the game state, mutating it in place.

        PRECONDITIONS:
          - is_terminal() is False
          - is_chance_node() is False
          - action_id is in legal_actions()

        After this call, the game may advance to:
          - Another decision point (possibly for a different player)
          - A chance node (dice roll needed)
          - A terminal state (game over)

        The engine handles all internal bookkeeping: phase transitions,
        resource transfers, figure placement, card effects, etc.

        Args:
            action_id: int in [0, action_space_size).
        """
        ...

    # ------------------------------------------------------------------
    # Chance Nodes (Dice)
    # ------------------------------------------------------------------

    @abstractmethod
    def is_chance_node(self) -> bool:
        """
        Return True if the game is at a point where randomness must
        be resolved before any player can act.

        WHEN THIS HAPPENS in Stone Age:
          1. Resource gathering: after a player's figures are at a
             resource location, dice are rolled. The chance node
             represents the possible sum totals.
          2. Dice-resource card effect: roll 2 dice for a resource.
          3. Potpourri card effect: roll N dice (one per player)
             producing categorical outcomes (wood/clay/stone/gold/
             tool/agriculture).

        When True, the caller must use apply_chance_outcome() instead
        of apply_action(). Calling legal_actions() or apply_action()
        when at a chance node is an error.

        Returns:
            bool
        """
        ...

    @abstractmethod
    def chance_outcomes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the possible chance outcomes and their probabilities.

        PRECONDITION: is_chance_node() is True.

        For dice sum rolls (e.g. 3 dice → sums 3..18):
          outcomes = [0, 1, 2, ..., 15]  (indices, not actual sums)
          probs    = [p(sum=3), p(sum=4), ..., p(sum=18)]

        For potpourri (categorical):
          outcomes = [0, 1, ..., K-1]  (indices into the list of
                     distinct multisets of {wood,clay,stone,gold,tool,agri})
          probs    = [probability of each multiset]

        The outcome indices are opaque — the caller passes them back
        to apply_chance_outcome() without interpretation.

        Returns:
            (outcomes, probs) where:
              outcomes: np.ndarray of shape (K,), dtype int32
              probs:    np.ndarray of shape (K,), dtype float32
              probs sums to 1.0 (within floating point tolerance)
              K >= 1
        """
        ...

    @abstractmethod
    def apply_chance_outcome(self, outcome_index: int) -> None:
        """
        Apply a specific chance outcome to the game state.

        PRECONDITION: is_chance_node() is True, and outcome_index
        is one of the values from chance_outcomes()[0].

        After this call, the game advances past the random event.
        It may reach another decision point, another chance node
        (e.g. a card effect triggers another dice roll), or a
        terminal state.

        For dice sums: outcome_index maps to a specific total.
        For potpourri: outcome_index maps to a specific set of
        bonus assignments.

        Args:
            outcome_index: int from chance_outcomes().
        """
        ...

    # ------------------------------------------------------------------
    # Terminal State
    # ------------------------------------------------------------------

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Return True if the game is over.

        The game ends when:
          - Not enough civilization cards to refill 4 slots, OR
          - Any building stack is empty (stack + display slot both empty)

        After the round that triggers this, end-game scoring happens
        (green card sets, sand card multipliers, remaining resources)
        and the game enters terminal state.

        When True:
          - legal_actions() is undefined (don't call it)
          - apply_action() is undefined (don't call it)
          - get_rewards() returns the final outcome

        Returns:
            bool
        """
        ...

    @abstractmethod
    def get_rewards(self) -> np.ndarray:
        """
        Return the terminal reward for each player.

        PRECONDITION: is_terminal() is True.

        REWARD SCHEME (configurable, but default is rank-based):
          2-player: winner +1, loser -1  (0 for draw, rare)
          3-player: 1st → +1, 2nd → 0, 3rd → -1
          4-player: 1st → +1, 2nd → +0.33, 3rd → -0.33, 4th → -1

        The rewards reflect final scores after end-game scoring,
        with tiebreakers applied per official rules (sum of
        agriculture + tools + extra figures).

        Returns:
            np.ndarray of shape (num_players,), dtype float32.
            Values in [-1, +1].
        """
        ...

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @abstractmethod
    def current_player(self) -> int:
        """
        Return the index of the player who must act next.

        Valid only when is_chance_node() is False and is_terminal()
        is False. During chance nodes, there is no "current player"
        (the universe is rolling dice).

        Returns:
            int in [0, num_players).
        """
        ...

    @abstractmethod
    def num_players(self) -> int:
        """
        Return the number of players (2, 3, or 4).

        This is fixed for the lifetime of the game instance.

        Returns:
            int in {2, 3, 4}.
        """
        ...

    @abstractmethod
    def move_number(self) -> int:
        """
        Return the total number of player actions taken so far.

        Used to control temperature scheduling in self-play
        (high temperature early for exploration, low temperature
        later for exploitation).

        Chance outcomes do NOT increment the move counter — only
        player decisions (apply_action calls) do.

        Returns:
            int >= 0.
        """
        ...


# Type alias: a callable that returns a fresh GameInterface
GameFactory = Callable[[], GameInterface]


# ════════════════════════════════════════════════════════════════════
# §3  NEURAL NETWORK INTERFACE  (to be implemented in PyTorch)
# ════════════════════════════════════════════════════════════════════

class NeuralNetwork(ABC):
    """
    Abstract interface to the policy-value neural network.

    The network takes a game observation and outputs:
      - policy: probability over all actions (pre-masking)
      - value:  expected outcome from this state for each player

    ARCHITECTURE (recommended, see plan §6):
      Input(416 floats) → FC/ResNet MLP (4-6 layers, 512 wide)
        ├─► Policy Head → softmax over action_space_size
        └─► Value Head  → tanh, size num_players

    The interface supports both single inference (for interactive
    play) and batched inference (for efficient MCTS).
    """

    @abstractmethod
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single observation through the network.

        PREPROCESSING (done inside this method):
          1. Convert int32 observation to float32
          2. Normalize each section appropriately:
             - Resource counts: divide by max (e.g. wood/28)
             - Booleans: keep as 0.0/1.0
             - Card/building IDs: embed or one-hot
          3. Feed through the network in eval mode (no dropout)

        INPUT:
          observation: np.ndarray of shape (observation_size,), dtype int32.
            This is the raw output of GameInterface.get_observation().

        OUTPUTS:
          policy: np.ndarray of shape (action_space_size,), dtype float32.
            Raw softmax output — probabilities over ALL actions,
            including illegal ones. The caller will mask to legal
            actions and renormalize.
            Sum ≈ 1.0 (before masking).

          value: np.ndarray of shape (num_players,), dtype float32.
            Expected outcome from this position for each player seat
            (seat 0 = current player due to observation rotation).
            Values in [-1, +1] (tanh output).

        Returns:
            (policy, value) tuple.
        """
        ...

    @abstractmethod
    def predict_batch(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a batch of observations through the network.

        This is the performance-critical path. During MCTS, we collect
        multiple leaf node observations and evaluate them in one GPU
        batch to maximize throughput.

        INPUT:
          observations: np.ndarray of shape (batch_size, observation_size),
            dtype int32. Each row is one observation.

        OUTPUTS:
          policies: np.ndarray of shape (batch_size, action_space_size),
            dtype float32. Each row is a policy vector (see predict()).

          values: np.ndarray of shape (batch_size, num_players),
            dtype float32. Each row is a value vector (see predict()).

        Returns:
            (policies, values) tuple.
        """
        ...

    @abstractmethod
    def train_batch(
        self,
        observations: np.ndarray,
        target_policies: np.ndarray,
        target_values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train the network on one batch of examples.

        LOSS FUNCTION:
          L = L_policy + L_value + c * L_reg

          L_policy = -sum(target_policy * log(pred_policy + eps))
            Cross-entropy between MCTS visit-count policy and
            network policy. target_policy is already masked to
            legal actions and normalized.

          L_value = MSE(pred_value, target_value)
            Mean squared error on the value prediction.
            target_value is the game outcome (reward) assigned
            after the game ends, from the perspective of the
            player whose observation this was.

          L_reg = weight_decay * sum(params^2)
            Standard L2 regularization (usually handled by the
            optimizer, not computed explicitly).

        INPUTS:
          observations:    (batch, observation_size), int32
          target_policies: (batch, action_space_size), float32
            Sparse — mostly zeros. Non-zero entries sum to 1.0.
            These are the MCTS visit count distributions.
          target_values:   (batch,), float32
            Terminal reward for the player whose obs this is.
            Range [-1, +1].

        RETURNS:
          Dict with at least:
            'loss':        total loss (float)
            'policy_loss': policy component (float)
            'value_loss':  value component (float)

        Side effect: updates network weights via optimizer.step().
        """
        ...

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save model weights and optimizer state to a file.

        Used for checkpointing during training and for saving
        the best model after evaluation.

        Args:
            filepath: path to save file (e.g. "checkpoints/model_042.pt")
        """
        ...

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load model weights and optimizer state from a file.

        Args:
            filepath: path to saved checkpoint.
        """
        ...

    @abstractmethod
    def copy_weights_to(self, other: NeuralNetwork) -> None:
        """
        Copy this network's weights into another network instance.

        Used to snapshot the current model as the "best model" for
        evaluation, without filesystem I/O.

        Args:
            other: target NeuralNetwork (same architecture).
        """
        ...


# ════════════════════════════════════════════════════════════════════
# §4  MCTS NODE
# ════════════════════════════════════════════════════════════════════

class MCTSNode:
    """
    A single node in the MCTS search tree.

    There are three kinds of nodes:
      1. DECISION node   — a player must choose an action.
         Has children keyed by action_id.
      2. CHANCE node     — the environment must roll dice.
         Has children keyed by outcome_index, weighted by probability.
      3. TERMINAL node   — game is over. Leaf with known reward.

    Attributes:
        parent:          Parent node (None for root).
        action_from_parent: Action (or outcome) that led here.
        player:          Player index at this node (-1 for chance/terminal).
        is_chance:       True if this is a chance node.
        prior:           Prior probability P(a) from NN (or chance prob).
        visit_count:     N — number of times this node was visited.
        value_sum:       W — sum of backed-up values (per player).
        children:        Dict[int, MCTSNode] — child nodes.
        is_expanded:     True after expand() has been called.
    """

    __slots__ = (
        'parent', 'action_from_parent', 'player', 'is_chance',
        'prior', 'visit_count', 'value_sum', 'children',
        'is_expanded', 'num_players',
    )

    def __init__(
        self,
        parent: Optional[MCTSNode] = None,
        action_from_parent: int = -1,
        player: int = -1,
        is_chance: bool = False,
        prior: float = 0.0,
        num_players: int = 2,
    ):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.player = player
        self.is_chance = is_chance
        self.prior = prior
        self.num_players = num_players

        self.visit_count: int = 0
        self.value_sum: np.ndarray = np.zeros(num_players, dtype=np.float64)
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded: bool = False

    @property
    def q_values(self) -> np.ndarray:
        """Mean value vector Q = W / N. Returns zeros if unvisited."""
        if self.visit_count == 0:
            return np.zeros(self.num_players, dtype=np.float64)
        return self.value_sum / self.visit_count

    def expand_decision(
        self,
        legal_actions: np.ndarray,
        policy_priors: np.ndarray,
        current_player: int,
    ) -> None:
        """
        Expand a decision node: create one child per legal action.

        Args:
            legal_actions:  array of legal action IDs.
            policy_priors:  full policy vector (action_space_size,).
                            Only entries at legal_actions indices are used.
            current_player: who is deciding.
        """
        self.is_expanded = True
        self.player = current_player

        # Extract priors for legal actions only, renormalize
        legal_priors = policy_priors[legal_actions]
        prior_sum = legal_priors.sum()
        if prior_sum > 0:
            legal_priors = legal_priors / prior_sum
        else:
            # Uniform fallback if NN gives zero mass to all legal moves
            legal_priors = np.ones(len(legal_actions), dtype=np.float32) / len(legal_actions)

        for i, action in enumerate(legal_actions):
            self.children[int(action)] = MCTSNode(
                parent=self,
                action_from_parent=int(action),
                prior=float(legal_priors[i]),
                num_players=self.num_players,
            )

    def expand_chance(
        self,
        outcomes: np.ndarray,
        probs: np.ndarray,
    ) -> None:
        """
        Expand a chance node: create one child per possible outcome.

        Args:
            outcomes: array of outcome indices.
            probs:    probability of each outcome (sums to ~1).
        """
        self.is_expanded = True
        self.is_chance = True
        self.player = -1  # no player at chance nodes

        for i, outcome in enumerate(outcomes):
            self.children[int(outcome)] = MCTSNode(
                parent=self,
                action_from_parent=int(outcome),
                is_chance=False,  # children of chance nodes are decision nodes
                prior=float(probs[i]),
                num_players=self.num_players,
            )

    def select_child_puct(self, c_puct: float) -> Tuple[int, MCTSNode]:
        """
        Select child with highest PUCT score.

        PUCT formula (from AlphaZero):
          score(a) = Q(a)[player] + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))

        Q(a)[player] is the mean value for the player who is deciding
        at this node. Higher Q = better for that player.

        Args:
            c_puct: exploration constant.

        Returns:
            (action_id, child_node) with highest PUCT score.
        """
        sqrt_parent = math.sqrt(self.visit_count)
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            q = child.q_values[self.player] if child.visit_count > 0 else 0.0
            u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def select_child_chance(self) -> Tuple[int, MCTSNode]:
        """
        Select a child of a chance node by sampling proportional to
        the known outcome probabilities (stored as priors).

        This is NOT UCB — chance nodes don't do exploration/exploitation.
        We sample from the true distribution every time.

        Returns:
            (outcome_index, child_node)
        """
        outcomes = list(self.children.keys())
        probs = np.array([self.children[o].prior for o in outcomes], dtype=np.float64)
        probs = probs / probs.sum()  # renormalize for safety
        idx = np.random.choice(len(outcomes), p=probs)
        chosen = outcomes[idx]
        return chosen, self.children[chosen]

    def add_dirichlet_noise(self, alpha: float, fraction: float) -> None:
        """
        Add Dirichlet noise to the root node priors for exploration.

        Applied only at the root of each MCTS search. The noise
        ensures the search doesn't collapse to a single move early
        in training when the NN is still weak.

        new_prior = (1 - fraction) * original_prior + fraction * noise

        Args:
            alpha:    Dirichlet concentration parameter.
                      Lower = more peaked, higher = more uniform.
                      0.3 is typical for board games.
            fraction: how much noise to mix in. 0.25 is standard.
        """
        if not self.children:
            return
        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for i, action in enumerate(actions):
            child = self.children[action]
            child.prior = (1 - fraction) * child.prior + fraction * noise[i]


# ════════════════════════════════════════════════════════════════════
# §5  MCTS SEARCH
# ════════════════════════════════════════════════════════════════════

class MCTSSearch:
    """
    Monte Carlo Tree Search with neural network guidance.

    One MCTSSearch instance is created per move decision. It builds
    a search tree rooted at the current game state and returns a
    policy (visit count distribution) over legal actions.

    Call flow for each simulation:
      1. SELECT:   Walk from root to a leaf using PUCT / chance sampling.
      2. EXPAND:   At the leaf, expand using the NN policy prior.
      3. EVALUATE: Get the value estimate from the NN.
      4. BACKUP:   Propagate the value back up the tree.

    After all simulations, the root's visit counts define the policy.
    """

    def __init__(self, config: AlphaZeroConfig, neural_net: NeuralNetwork):
        self.config = config
        self.nn = neural_net

    def search(
        self,
        game: GameInterface,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Run MCTS from the given game state.

        Args:
            game:      current game state (will be cloned internally;
                       the original is NOT modified).
            add_noise: if True, add Dirichlet noise at root for
                       exploration (used during self-play, not eval).

        Returns:
            policy: np.ndarray of shape (action_space_size,), float32.
              The visit-count distribution over actions. Only entries
              corresponding to legal actions are non-zero. Sums to 1.
        """
        root = MCTSNode(num_players=game.num_players())

        # Expand root
        self._expand_node(root, game)

        if add_noise and not root.is_chance:
            root.add_dirichlet_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_fraction,
            )

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            sim_game = game.clone()
            search_path: List[MCTSNode] = [node]

            # SELECT: traverse tree to a leaf
            while node.is_expanded and not sim_game.is_terminal():
                if node.is_chance:
                    outcome, node = node.select_child_chance()
                    sim_game.apply_chance_outcome(outcome)
                else:
                    action, node = node.select_child_puct(self.config.c_puct)
                    sim_game.apply_action(action)
                search_path.append(node)

            # EVALUATE
            if sim_game.is_terminal():
                value = sim_game.get_rewards()  # shape (num_players,)
            else:
                # EXPAND leaf
                self._expand_node(node, sim_game)
                # Get NN value estimate
                obs = sim_game.get_observation()
                _, value = self.nn.predict(obs)
                # value is from rotated perspective (seat 0 = current player)
                # un-rotate to absolute player indices
                value = self._unrotate_value(value, sim_game.current_player(), game.num_players())

            # BACKUP
            for path_node in reversed(search_path):
                path_node.visit_count += 1
                path_node.value_sum += value

        # Extract policy from root visit counts
        return self._root_policy(root, game)

    def _expand_node(self, node: MCTSNode, game: GameInterface) -> None:
        """Expand a node — either decision or chance."""
        if game.is_terminal():
            return

        if game.is_chance_node():
            outcomes, probs = game.chance_outcomes()
            node.expand_chance(outcomes, probs)
        else:
            legal = game.legal_actions()
            obs = game.get_observation()
            policy, _ = self.nn.predict(obs)
            node.expand_decision(legal, policy, game.current_player())

    def _unrotate_value(
        self,
        value: np.ndarray,
        current_player: int,
        num_players: int,
    ) -> np.ndarray:
        """
        The NN outputs values from the perspective of the current player
        (seat 0 = current player, seat 1 = next, etc). We need to map
        these back to absolute player indices for tree backup.

        If the NN says value = [v0, v1] with current_player=1 in a
        2-player game, that means:
          absolute_player_1 (current) → v0
          absolute_player_0           → v1

        So we reverse the rotation.
        """
        result = np.zeros(num_players, dtype=np.float64)
        for relative_seat in range(num_players):
            absolute_seat = (current_player + relative_seat) % num_players
            result[absolute_seat] = value[relative_seat]
        return result

    def _root_policy(
        self,
        root: MCTSNode,
        game: GameInterface,
    ) -> np.ndarray:
        """
        Convert root visit counts into a probability distribution.

        Returns:
            np.ndarray of shape (action_space_size,), sums to 1.0.
        """
        policy = np.zeros(self.config.action_space_size, dtype=np.float32)

        if root.is_chance:
            # Shouldn't normally search from a chance node, but handle it
            return policy

        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            # Fallback: uniform over legal actions
            legal = game.legal_actions()
            policy[legal] = 1.0 / len(legal)
            return policy

        for action, child in root.children.items():
            policy[action] = child.visit_count / total_visits

        return policy


# ════════════════════════════════════════════════════════════════════
# §6  BATCHED MCTS (multiple trees in lockstep for GPU efficiency)
# ════════════════════════════════════════════════════════════════════

class BatchedMCTS:
    """
    Run multiple independent MCTS searches in lockstep to batch
    NN evaluations for GPU throughput.

    Instead of evaluating one observation at a time (wasteful on GPU),
    we run N MCTS trees simultaneously. Each simulation step, we
    collect all the leaf observations, batch them into one NN call,
    then distribute the results back.

    This is the primary way to get good GPU utilization during
    self-play, which is the bottleneck of AlphaZero training.
    """

    def __init__(self, config: AlphaZeroConfig, neural_net: NeuralNetwork):
        self.config = config
        self.nn = neural_net

    def search_batch(
        self,
        games: List[GameInterface],
        add_noise: bool = True,
    ) -> List[np.ndarray]:
        """
        Run MCTS for a batch of game states simultaneously.

        Args:
            games:     list of game states (not modified).
            add_noise: add Dirichlet noise at roots.

        Returns:
            list of policy vectors, one per game.
        """
        n = len(games)
        num_players = games[0].num_players()

        # Initialize roots
        roots: List[MCTSNode] = []
        for game in games:
            root = MCTSNode(num_players=num_players)
            roots.append(root)

        # Expand all roots — batch the NN calls
        self._batch_expand(roots, games)

        if add_noise:
            for root in roots:
                if not root.is_chance:
                    root.add_dirichlet_noise(
                        self.config.dirichlet_alpha,
                        self.config.dirichlet_fraction,
                    )

        # Run simulations in lockstep
        for _ in range(self.config.num_simulations):
            # Each tree does SELECT independently, collecting leaves
            leaf_data = []  # (tree_idx, search_path, sim_game, node)

            for i in range(n):
                node = roots[i]
                sim_game = games[i].clone()
                search_path = [node]

                # SELECT
                while node.is_expanded and not sim_game.is_terminal():
                    if node.is_chance:
                        outcome, node = node.select_child_chance()
                        sim_game.apply_chance_outcome(outcome)
                    else:
                        action, node = node.select_child_puct(self.config.c_puct)
                        sim_game.apply_action(action)
                    search_path.append(node)

                leaf_data.append((i, search_path, sim_game, node))

            # Separate terminals from leaves needing NN eval
            needs_eval = []
            terminal_data = []

            for i, search_path, sim_game, node in leaf_data:
                if sim_game.is_terminal():
                    terminal_data.append((i, search_path, sim_game.get_rewards()))
                else:
                    needs_eval.append((i, search_path, sim_game, node))

            # BATCH EXPAND + EVALUATE
            if needs_eval:
                observations = []
                current_players = []

                for _, _, sim_game, node in needs_eval:
                    # Expand the leaf node
                    if sim_game.is_chance_node():
                        outcomes, probs = sim_game.chance_outcomes()
                        node.expand_chance(outcomes, probs)
                        # Still need a value estimate — use obs from before chance
                        # Actually, for chance nodes we can skip NN eval and
                        # just let future simulations handle it. But for
                        # consistency, we evaluate the state as-is.
                    observations.append(sim_game.get_observation())
                    current_players.append(
                        sim_game.current_player() if not sim_game.is_chance_node() else 0
                    )

                # One batched NN call for all leaves
                obs_batch = np.stack(observations)
                policies_batch, values_batch = self.nn.predict_batch(obs_batch)

                # Distribute results
                for j, (idx, search_path, sim_game, node) in enumerate(needs_eval):
                    policy = policies_batch[j]
                    value = values_batch[j]

                    # Expand decision nodes that weren't expanded yet
                    if not node.is_expanded and not sim_game.is_chance_node():
                        legal = sim_game.legal_actions()
                        node.expand_decision(legal, policy, sim_game.current_player())

                    # Un-rotate value
                    cp = current_players[j]
                    abs_value = np.zeros(num_players, dtype=np.float64)
                    for rel in range(num_players):
                        abs_value[(cp + rel) % num_players] = value[rel]

                    # BACKUP
                    for path_node in reversed(search_path):
                        path_node.visit_count += 1
                        path_node.value_sum += abs_value

            # BACKUP terminals
            for idx, search_path, rewards in terminal_data:
                for path_node in reversed(search_path):
                    path_node.visit_count += 1
                    path_node.value_sum += rewards

        # Extract policies
        policies = []
        for i, (root, game) in enumerate(zip(roots, games)):
            policy = np.zeros(self.config.action_space_size, dtype=np.float32)
            total = sum(c.visit_count for c in root.children.values())
            if total > 0:
                for action, child in root.children.items():
                    policy[action] = child.visit_count / total
            else:
                legal = game.legal_actions()
                policy[legal] = 1.0 / len(legal)
            policies.append(policy)

        return policies

    def _batch_expand(
        self,
        nodes: List[MCTSNode],
        games: List[GameInterface],
    ) -> None:
        """Expand all root nodes, batching the NN calls."""
        observations = []
        decision_indices = []  # which nodes are decision (not chance/terminal)

        for i, (node, game) in enumerate(zip(nodes, games)):
            if game.is_terminal():
                continue
            if game.is_chance_node():
                outcomes, probs = game.chance_outcomes()
                node.expand_chance(outcomes, probs)
            else:
                observations.append(game.get_observation())
                decision_indices.append(i)

        if observations:
            obs_batch = np.stack(observations)
            policies_batch, _ = self.nn.predict_batch(obs_batch)

            for j, idx in enumerate(decision_indices):
                game = games[idx]
                legal = game.legal_actions()
                nodes[idx].expand_decision(
                    legal, policies_batch[j], game.current_player()
                )


# ════════════════════════════════════════════════════════════════════
# §7  REPLAY BUFFER
# ════════════════════════════════════════════════════════════════════

@dataclass
class TrainingExample:
    """One training example = one decision point from one game."""
    observation: np.ndarray    # (observation_size,) int32
    policy: np.ndarray         # (action_space_size,) float32 — MCTS visit dist
    value: float               # terminal reward for this player, float32

    def __repr__(self):
        return f"TrainingExample(obs_sum={self.observation.sum()}, value={self.value:.3f})"


class ReplayBuffer:
    """
    Fixed-size circular buffer of training examples.

    Examples are added from self-play games. When the buffer is full,
    oldest examples are evicted. Training samples uniformly at random
    from the buffer.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, example: TrainingExample) -> None:
        """Add one example to the buffer."""
        self.buffer.append(example)

    def add_batch(self, examples: List[TrainingExample]) -> None:
        """Add multiple examples."""
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of training examples.

        Returns:
            observations:    (batch_size, observation_size), int32
            target_policies: (batch_size, action_space_size), float32
            target_values:   (batch_size,), float32
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        examples = [self.buffer[i] for i in indices]

        observations = np.stack([e.observation for e in examples])
        policies = np.stack([e.policy for e in examples])
        values = np.array([e.value for e in examples], dtype=np.float32)

        return observations, policies, values

    def save(self, filepath: str) -> None:
        """Save buffer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, filepath: str) -> None:
        """Load buffer from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.buffer = deque(data, maxlen=self.max_size)


# ════════════════════════════════════════════════════════════════════
# §8  SELF-PLAY
# ════════════════════════════════════════════════════════════════════

class SelfPlayManager:
    """
    Generates training data by playing games using MCTS + current NN.

    Each game produces a list of TrainingExamples. The policy targets
    come from MCTS visit counts; the value targets come from the
    terminal game outcome.

    Temperature schedule:
      - First N moves (temperature_move_threshold): sample from MCTS
        policy proportionally (temperature=1). This adds diversity.
      - After N moves: play the move with highest visit count
        (temperature→0). This plays stronger in the endgame.
    """

    def __init__(
        self,
        config: AlphaZeroConfig,
        game_factory: GameFactory,
        neural_net: NeuralNetwork,
    ):
        self.config = config
        self.game_factory = game_factory
        self.nn = neural_net
        self.mcts = MCTSSearch(config, neural_net)

    def play_one_game(self) -> List[TrainingExample]:
        """
        Play one complete game of self-play, returning training examples.

        Returns:
            List of TrainingExamples from every decision point in the game.
        """
        game = self.game_factory()
        trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []
        # Each entry: (observation, mcts_policy, current_player)

        move_count = 0

        while not game.is_terminal() and move_count < self.config.max_game_moves:
            # Skip chance nodes — no decision, just sample
            if game.is_chance_node():
                outcomes, probs = game.chance_outcomes()
                idx = np.random.choice(len(outcomes), p=probs)
                game.apply_chance_outcome(int(outcomes[idx]))
                continue

            # Get observation BEFORE the move
            obs = game.get_observation()
            current = game.current_player()

            # Run MCTS
            policy = self.mcts.search(game, add_noise=True)

            # Store this decision point
            trajectory.append((obs.copy(), policy.copy(), current))

            # Choose action with temperature
            action = self._sample_action(policy, move_count)
            game.apply_action(action)
            move_count += 1

        if move_count >= self.config.max_game_moves:
            logger.warning("Game hit max move limit (%d)", self.config.max_game_moves)

        # Get terminal rewards
        if game.is_terminal():
            rewards = game.get_rewards()
        else:
            # Game didn't finish (hit move limit) — use zeros
            rewards = np.zeros(game.num_players(), dtype=np.float32)

        # Build training examples: each obs gets the reward of the player
        # whose perspective it was from (seat 0 in the rotated obs)
        examples = []
        for obs, policy, player in trajectory:
            examples.append(TrainingExample(
                observation=obs,
                policy=policy,
                value=float(rewards[player]),
            ))

        return examples

    def play_games(self, num_games: int) -> List[TrainingExample]:
        """
        Play multiple games sequentially, collecting all examples.

        For parallel self-play, see ParallelSelfPlayManager below.
        """
        all_examples = []
        for i in range(num_games):
            examples = self.play_one_game()
            all_examples.extend(examples)
            if (i + 1) % self.config.log_interval == 0:
                logger.info(
                    "Self-play: %d/%d games, %d total examples",
                    i + 1, num_games, len(all_examples),
                )
        return all_examples

    def _sample_action(self, policy: np.ndarray, move_number: int) -> int:
        """
        Sample an action from the MCTS policy with temperature control.

        Early game (move < threshold): sample proportional to visit
          counts (temperature = 1). This introduces variety so the NN
          sees diverse positions during training.

        Late game (move >= threshold): pick the action with the most
          visits (temperature → 0). This plays strongest.

        Args:
            policy: visit count distribution over all actions.
            move_number: how many moves into the game.

        Returns:
            action_id: int
        """
        if move_number < self.config.temperature_move_threshold:
            # Temperature = 1: sample proportionally
            # (policy is already a probability distribution)
            nonzero = policy > 0
            if not nonzero.any():
                raise ValueError("Policy is all zeros — no legal actions?")
            action = np.random.choice(len(policy), p=policy)
        else:
            # Temperature → 0: pick the max
            action = int(np.argmax(policy))

        return action


class BatchedSelfPlayManager:
    """
    Self-play using BatchedMCTS for GPU-efficient NN evaluation.

    Runs `batch_size` games simultaneously. Each move decision, all
    active games' MCTS searches are batched into shared NN calls.
    This gives ~10-15× speedup over sequential self-play on GPU/MPS.

    Games that finish are replaced with new ones until the target
    number of games is reached.
    """

    def __init__(
        self,
        config: AlphaZeroConfig,
        game_factory: GameFactory,
        neural_net: NeuralNetwork,
        batch_size: int = 16,
    ):
        self.config = config
        self.game_factory = game_factory
        self.nn = neural_net
        self.batched_mcts = BatchedMCTS(config, neural_net)
        self.batch_size = batch_size

    @staticmethod
    def _advance_past_chance(game):
        """Resolve all consecutive chance nodes (dice rolls)."""
        while not game.is_terminal() and game.is_chance_node():
            outcomes, probs = game.chance_outcomes()
            idx = np.random.choice(len(outcomes), p=probs)
            game.apply_chance_outcome(int(outcomes[idx]))

    @staticmethod
    def _sample_action(policy, move_number, threshold):
        if move_number < threshold:
            nonzero = policy > 0
            if not nonzero.any():
                return int(np.argmax(policy))
            return int(np.random.choice(len(policy), p=policy))
        return int(np.argmax(policy))

    def play_games(self, num_games: int) -> List[TrainingExample]:
        """
        Play num_games of self-play using batched MCTS.

        Games run in parallel batches. When a game finishes, its
        training examples are collected and a new game starts in
        its slot (until we've completed enough games).
        """
        all_examples: List[TrainingExample] = []
        games_completed = 0
        games_started = 0

        # Per-slot state
        games = [None] * self.batch_size
        trajectories = [None] * self.batch_size  # list of (obs, policy, player)
        move_counts = [0] * self.batch_size
        active = [False] * self.batch_size

        def start_game(slot):
            nonlocal games_started
            games[slot] = self.game_factory()
            trajectories[slot] = []
            move_counts[slot] = 0
            active[slot] = True
            games_started += 1
            self._advance_past_chance(games[slot])

        def finish_game(slot):
            nonlocal games_completed
            game = games[slot]
            if game.is_terminal():
                rewards = game.get_rewards()
            else:
                rewards = np.zeros(game.num_players(), dtype=np.float32)

            for obs, policy, player in trajectories[slot]:
                all_examples.append(TrainingExample(
                    observation=obs,
                    policy=policy,
                    value=float(rewards[player]),
                ))

            games_completed += 1
            active[slot] = False

            if games_completed % self.config.log_interval == 0:
                logger.info(
                    "Self-play: %d/%d games, %d examples",
                    games_completed, num_games, len(all_examples),
                )

        # Start initial batch
        n_initial = min(self.batch_size, num_games)
        for i in range(n_initial):
            start_game(i)

        # Main loop
        while games_completed < num_games:
            # Collect active decision-point games
            decision_slots = []
            decision_games = []
            for i in range(self.batch_size):
                if not active[i]:
                    continue
                g = games[i]
                if g.is_terminal() or move_counts[i] >= self.config.max_game_moves:
                    finish_game(i)
                    if games_started < num_games:
                        start_game(i)
                        g = games[i]
                    else:
                        continue
                if active[i] and not g.is_terminal() and not g.is_chance_node():
                    decision_slots.append(i)
                    decision_games.append(g)

            if not decision_games:
                break

            # Record observations before search
            observations = [g.get_observation() for g in decision_games]
            players = [g.current_player() for g in decision_games]

            # BATCHED MCTS — one NN call per simulation for ALL games
            policies = self.batched_mcts.search_batch(decision_games, add_noise=True)

            # Apply actions
            for j, slot in enumerate(decision_slots):
                obs = observations[j]
                policy = policies[j]
                player = players[j]

                trajectories[slot].append((obs.copy(), policy.copy(), player))

                action = self._sample_action(
                    policy, move_counts[slot],
                    self.config.temperature_move_threshold,
                )
                games[slot].apply_action(action)
                move_counts[slot] += 1

                # Advance past chance nodes after the action
                self._advance_past_chance(games[slot])

        # Finish any remaining active games
        for i in range(self.batch_size):
            if active[i] and trajectories[i]:
                finish_game(i)

        return all_examples


# ════════════════════════════════════════════════════════════════════
# §9  EVALUATION
# ════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Evaluate a new model against the current best by playing games.

    Both models use MCTS (but with no noise and temperature=0)
    to play at full strength. The new model must win > eval_threshold
    of games to replace the best model.

    Players alternate who goes first to reduce first-move advantage.
    """

    def __init__(self, config: AlphaZeroConfig, game_factory: GameFactory):
        self.config = config
        self.game_factory = game_factory

    def evaluate(
        self,
        new_net: NeuralNetwork,
        best_net: NeuralNetwork,
    ) -> Tuple[float, Dict[str, int]]:
        """
        Play evaluation games between two models.

        Args:
            new_net:  the model being evaluated.
            best_net: the current best model.

        Returns:
            (win_rate, stats) where:
              win_rate: float in [0, 1], fraction of games won by new_net.
              stats: dict with 'wins', 'losses', 'draws'.
        """
        new_mcts = MCTSSearch(self.config, new_net)
        best_mcts = MCTSSearch(self.config, best_net)

        wins = 0
        losses = 0
        draws = 0

        for i in range(self.config.eval_games):
            # Alternate who plays as player 0
            if i % 2 == 0:
                # new_net is player 0, best_net is player 1
                result = self._play_eval_game(new_mcts, best_mcts, new_player=0)
            else:
                # new_net is player 1, best_net is player 0
                result = self._play_eval_game(new_mcts, best_mcts, new_player=1)

            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1

            if (i + 1) % 10 == 0:
                logger.info(
                    "Eval: %d/%d games, W:%d L:%d D:%d",
                    i + 1, self.config.eval_games, wins, losses, draws,
                )

        total = wins + losses + draws
        win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.0
        stats = {'wins': wins, 'losses': losses, 'draws': draws}
        return win_rate, stats

    def _play_eval_game(
        self,
        new_mcts: MCTSSearch,
        best_mcts: MCTSSearch,
        new_player: int,
    ) -> int:
        """
        Play one evaluation game. Returns +1 if new wins, -1 if best wins, 0 draw.
        """
        game = self.game_factory()

        while not game.is_terminal():
            if game.is_chance_node():
                outcomes, probs = game.chance_outcomes()
                idx = np.random.choice(len(outcomes), p=probs)
                game.apply_chance_outcome(int(outcomes[idx]))
                continue

            current = game.current_player()

            # Use the appropriate MCTS (no noise, greedy action selection)
            if current == new_player:
                policy = new_mcts.search(game, add_noise=False)
            else:
                policy = best_mcts.search(game, add_noise=False)

            # Greedy: pick action with highest visit count
            action = int(np.argmax(policy))
            game.apply_action(action)

        rewards = game.get_rewards()
        new_reward = rewards[new_player]

        if new_reward > 0:
            return 1
        elif new_reward < 0:
            return -1
        else:
            return 0


# ════════════════════════════════════════════════════════════════════
# §10  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════

class AlphaZeroTrainer:
    """
    Top-level training orchestrator.

    Each iteration:
      1. Self-play: generate games with the current best model.
      2. Train: update the model on the replay buffer.
      3. Evaluate: pit the new model against the current best.
      4. If the new model wins enough, promote it to best.

    Checkpoints are saved after each iteration.
    """

    def __init__(
        self,
        config: AlphaZeroConfig,
        game_factory: GameFactory,
        neural_net: NeuralNetwork,
    ):
        self.config = config
        self.game_factory = game_factory
        self.current_net = neural_net
        self.best_net: Optional[NeuralNetwork] = None  # set in run()
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.evaluator = Evaluator(config, game_factory)
        self.iteration = 0

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def run(self) -> None:
        """Run the full AlphaZero training loop."""
        logger.info("Starting AlphaZero training")
        logger.info("Config: %s", self.config)

        # Initialize best_net as a copy of the starting network
        # (the NeuralNetwork implementation must support copy_weights_to)
        # For now, best_net IS current_net until first eval
        self.best_net = self.current_net  # shallow ref; updated after first eval

        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            logger.info("=" * 60)
            logger.info("ITERATION %d / %d", iteration + 1, self.config.num_iterations)
            logger.info("=" * 60)

            t0 = time.time()

            # ── Phase 1: Self-Play ──
            logger.info("Phase 1: Self-play (%d games)", self.config.num_self_play_games)
            self_play = BatchedSelfPlayManager(
                self.config, self.game_factory, self.current_net,
                batch_size=min(256, self.config.num_self_play_games),
            )
            new_examples = self_play.play_games(self.config.num_self_play_games)
            self.replay_buffer.add_batch(new_examples)
            logger.info(
                "Self-play done: %d new examples, buffer size: %d",
                len(new_examples), len(self.replay_buffer),
            )

            t1 = time.time()
            logger.info("Self-play took %.1fs", t1 - t0)

            # ── Phase 2: Training ──
            if len(self.replay_buffer) < self.config.min_replay_size:
                logger.info(
                    "Buffer too small (%d < %d), skipping training",
                    len(self.replay_buffer), self.config.min_replay_size,
                )
                continue

            logger.info("Phase 2: Training (%d epochs)", self.config.num_epochs)
            train_stats = self._train()
            logger.info("Training done: %s", train_stats)

            t2 = time.time()
            logger.info("Training took %.1fs", t2 - t1)

            # ── Checkpoint ──
            self._save_checkpoint(iteration)

            logger.info("Total iteration time: %.1fs", t2 - t0)

    def _train(self) -> Dict[str, float]:
        """
        Train the current network on the replay buffer for num_epochs.

        Returns:
            Aggregate stats over all training steps.
        """
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_steps = 0

        for epoch in range(self.config.num_epochs):
            # Number of batches per epoch = buffer_size / batch_size
            num_batches = max(1, len(self.replay_buffer) // self.config.batch_size)

            for _ in range(num_batches):
                actual_batch = min(self.config.batch_size, len(self.replay_buffer))
                obs, target_pol, target_val = self.replay_buffer.sample(actual_batch)

                stats = self.current_net.train_batch(obs, target_pol, target_val)

                total_loss += stats['loss']
                total_policy_loss += stats['policy_loss']
                total_value_loss += stats['value_loss']
                num_steps += 1

        return {
            'avg_loss': total_loss / max(num_steps, 1),
            'avg_policy_loss': total_policy_loss / max(num_steps, 1),
            'avg_value_loss': total_value_loss / max(num_steps, 1),
            'num_steps': num_steps,
        }

    def _save_checkpoint(self, iteration: int) -> None:
        """Save model and replay buffer."""
        model_path = os.path.join(
            self.config.checkpoint_dir, f"model_{iteration:04d}.pt"
        )
        self.current_net.save(model_path)

        # Save replay buffer less frequently (it's large)
        if iteration % 10 == 0:
            buf_path = os.path.join(
                self.config.checkpoint_dir, f"buffer_{iteration:04d}.pkl"
            )
            self.replay_buffer.save(buf_path)

        logger.info("Checkpoint saved: %s", model_path)


# ════════════════════════════════════════════════════════════════════
# §11  UTILITIES
# ════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed) — do this in the NN implementation


def play_one_move_with_mcts(
    game: GameInterface,
    nn: NeuralNetwork,
    config: AlphaZeroConfig,
    add_noise: bool = False,
    temperature: float = 0.0,
) -> Tuple[int, np.ndarray]:
    """
    Convenience function: run MCTS and pick one action.

    Useful for interactive play or debugging.

    Args:
        game:        current game state (not modified).
        nn:          neural network.
        config:      MCTS config.
        add_noise:   Dirichlet noise at root.
        temperature: 0 = greedy, 1 = proportional sampling.

    Returns:
        (action_id, policy) tuple.
    """
    mcts = MCTSSearch(config, nn)
    policy = mcts.search(game, add_noise=add_noise)

    if temperature == 0.0:
        action = int(np.argmax(policy))
    else:
        # Apply temperature: raise visit counts to 1/temp, renormalize
        adjusted = policy ** (1.0 / temperature)
        adjusted = adjusted / adjusted.sum()
        action = np.random.choice(len(adjusted), p=adjusted)

    return action, policy
