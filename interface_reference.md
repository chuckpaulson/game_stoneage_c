# Interface Reference — Every Function Call Explained

This document describes every function in the `GameInterface` and `NeuralNetwork`
abstract classes: what it does, when MCTS calls it, what data flows through it,
and what the C engine / PyTorch model must do internally.

---

## Call Flow Overview

Here's what happens during one MCTS simulation (one of ~400 per move):

```
SELECT phase (walking down the existing tree):
│
│  At each node, check: is it terminal? chance? decision?
│  ├─ Terminal:  stop, use game.get_rewards()
│  ├─ Chance:    node.select_child_chance() → game.apply_chance_outcome()
│  └─ Decision:  node.select_child_puct()   → game.apply_action()
│
│  Loop until we reach an unexpanded leaf.
│
EXPAND + EVALUATE phase (at the leaf):
│
│  game.is_terminal()?
│  ├─ Yes: value = game.get_rewards()
│  └─ No:
│       game.is_chance_node()?
│       ├─ Yes: game.chance_outcomes() → node.expand_chance()
│       └─ No:  game.get_observation()     ← NN input
│               nn.predict(observation)     ← NN forward pass
│               game.legal_actions()        ← mask for policy
│               node.expand_decision()      ← create children
│               value = nn output
│
BACKUP phase (walking back up):
│
│  For each node in path: node.visit_count += 1, node.value_sum += value
```

And here's the full self-play game loop:

```
game = game_factory()                    ← new game

while not game.is_terminal():
│
│  game.is_chance_node()?
│  ├─ Yes: game.chance_outcomes()        ← get dice distribution
│  │       sample from probs
│  │       game.apply_chance_outcome()   ← apply dice result
│  │       continue (no MCTS needed)
│  │
│  └─ No:  game.get_observation()        ← save for training
│          game.current_player()         ← who's deciding
│          MCTS search (400 simulations):
│            each sim clones: game.clone()
│            runs SELECT/EXPAND/EVALUATE/BACKUP
│          → mcts_policy
│          sample action from mcts_policy
│          game.apply_action(action)     ← advance the real game

game.get_rewards()                       ← terminal outcome
assign rewards to training examples
```

---

## GameInterface Functions — Detailed Breakdown

### `clone() → GameInterface`

**When called:** At the START of every MCTS simulation. If you run 400
simulations per move and a game has ~150 moves, that's 60,000 clone calls
per game.

**What it must do:** Create a completely independent copy of the game state.
Mutating the clone must not affect the original.

**Performance requirement:** This is the #1 hottest function. For the C
backend: `malloc(sizeof(GameState))` + `memcpy`. The GameState struct is
~500 bytes, so this should take < 1 microsecond. No Python object creation,
no dict copying, no list allocation.

**What MCTS does with it:** The clone is the "simulation game" that gets
mutated as MCTS walks down the tree (applying actions and chance outcomes).
It's thrown away after the simulation.

**Common bug:** Shallow copies that share mutable sub-objects. Every array,
dict, and list inside the state must be independent.

---

### `get_observation() → np.ndarray`  (shape: `(416,)`, dtype: `int32`)

**When called:** Twice per MCTS simulation that reaches a non-terminal,
non-chance leaf node:
1. Once during root expansion (at the start of search)
2. Once at each leaf during EXPAND

Also called once per move in self-play to record the training example.

Total: ~400 calls per move during search + 1 for the training record.

**What it must do:** Serialize the full game state into a fixed-size int32
array, rotated so that the current player is always at index 0. This is
your existing `get_state()` function from `stone_age_state.py`.

**Layout (from your existing code):**
```
[0:4]     Game meta: num_players, round, first_player_relative, current_player_idx
[4:41]    Phase state: phase, active_player, sub_phase, resolution_player,
          resolution_completed[4], locations_resolved[16],
          dice_result, pending_resource_type, pending_slot, remaining_picks,
          potpourri_available[4], players_fed[4], food_still_needed
[41:73]   Board locations: 8 locations × 4 players (figure counts)
[73:78]   Board misc: blocked_village, wood_supply, clay_supply, stone_supply, gold_supply
[78:86]   Civ card slots: 4 × (card_id, figure_player)
[86:94]   Building slots: 4 × (building_id, figure_player)
[94:386]  Player states: 4 × (resources[4], food, figures, figures_available,
          agriculture, tools[3], tools_used[3], score,
          one_time_tool_counts[2], civ_card_counts[34], building_counts[28])
[386:420] Civ deck remaining: counts[34]
[420:424] Building stack sizes: sizes[4]
```

**Player rotation:** All player-indexed fields are rotated so that
`relative_index = (absolute_index - current_player) % num_players`.
The NN always sees "self" at position 0, "next opponent" at position 1, etc.
This is critical — without rotation, the NN would need to learn the same
strategy independently for each seat.

**What MCTS does with it:** Passes directly to `nn.predict()` as input.

---

### `legal_actions() → np.ndarray`  (shape: `(N,)`, dtype: `int32`)

**When called:** Once per leaf expansion during MCTS (inside `_expand_node`),
to know which children to create. Also called to build the policy output.

~400 calls per move.

**What it must do:** Return the set of action IDs that are valid in the
current sub-phase. This is context-dependent:

| Sub-Phase | What legal_actions returns |
|-----------|--------------------------|
| CHOOSE_LOCATION (placement) | All valid (location, figure_count) combos encoded as action IDs. E.g., "place 3 figures on forest" = action 23, "place 1 figure on card slot 2" = action 7. Must respect: location capacity (7 for resources), player exclusion rules (can't go where you already are), 2-3 player restrictions, village blocking, figure availability. |
| TOOL_DECISION | "use permanent tool 0", "use permanent tool 1", "use permanent tool 2", "use one-time tool 0", "use one-time tool 1", "done using tools". Only unused tools with value > 0 are legal. "Done" is always legal. |
| BUILDING_PAYMENT | All valid (wood, clay, stone, gold) resource combos that satisfy the building's requirements. For FIXED buildings: exactly one option (the fixed cost). For VAR_4_2: all combos of 4 resources using exactly 2 types that the player can afford. For VAR_1_TO_7: all combos of 1-7 resources. Each combo maps to a pre-enumerated action ID. |
| CARD_PAYMENT | All valid resource combos summing to the card's cost (1-4 resources, any types) that the player can afford. |
| ANY_RESOURCE_CHOICE | [wood, clay, stone, gold] — 4 actions. Repeated once per resource to pick. |
| POTPOURRI_PICK | Indices of remaining bonuses from the potpourri dice roll (up to 4 options). |
| FREE_CARD_CHOICE | Slot indices where a card exists (0-3). |
| FEED_RESOURCE_CHOICE | Resource combos to pay for feeding shortfall, plus "accept -10 penalty". |

**Invariant:** Must return at least 1 action. If the game is at a decision
point, there must be something the player can do (even if it's just "pass"
or "accept penalty").

---

### `legal_actions_mask() → np.ndarray`  (shape: `(action_space_size,)`, dtype: `bool`)

**When called:** Not directly by core MCTS (which uses `legal_actions()`),
but available for the training pipeline and NN to mask policy output.

**What it must do:** Return a boolean array where `mask[a] = True` iff
action `a` is legal. Equivalent to building from `legal_actions()` but
may be faster as a single C function filling a fixed-size buffer.

---

### `apply_action(action_id: int) → None`

**When called:** Once per step in the SELECT phase of MCTS (walking down
the tree), applied to the cloned simulation game. Also once per move in
the real game during self-play.

~hundreds of calls per move in total across all simulations.

**What it must do:** Execute the action, mutating the game state. This
includes all internal bookkeeping:

- **Placement action:** Place figures at the chosen location. Update
  `figures_available`. Check village blocking rules. If all players have
  placed all figures, transition to RESOLUTION phase.

- **Tool decision:** If "use tool X" — mark tool as used, add its value
  to the pending dice result. If "done" — compute final resource count
  (dice + tools) / divisor, award resources, advance to next resolution.

- **Building payment:** Deduct resources from player, return to supply,
  calculate points (fixed or value-sum), add building to player's
  collection, advance to next resolution.

- **Card payment:** Deduct resources, take card, trigger immediate
  effect. If the effect requires another decision (ANY_RESOURCE_CHOICE,
  POTPOURRI_PICK, FREE_CARD_CHOICE), transition to that sub-phase.
  Otherwise advance to next resolution.

- **Any resource choice:** Give the chosen resource to the player.
  Decrement `remaining_picks`. If picks remain, stay in this sub-phase.
  Otherwise return to the parent resolution flow.

- **Potpourri pick:** Award the chosen bonus to the current player.
  Advance to the next player's pick (or end if all picked).

- **Free card choice:** Take the card from the chosen slot, trigger
  its effect (may chain into another sub-phase).

- **Feeding:** Apply the chosen resource payment or penalty. If all
  players fed, check game-end conditions and advance to next round
  or game over.

**Phase transitions:** The engine must automatically advance through
phases. After the last placement → start resolution. After resolving
all locations for all players → start feeding. After feeding → check
game end → start next round or set terminal.

**Chance transitions:** If an action leads to a point where dice must
be rolled (e.g., player placed figures at forest, now we need to roll),
the engine transitions to a chance node state. The next call should be
`is_chance_node() → True`, NOT another `apply_action()`.

---

### `is_chance_node() → bool`

**When called:** At the top of every SELECT step in MCTS, and in the
self-play game loop, to determine whether to use `apply_action` or
`apply_chance_outcome`.

**What it must do:** Return True when the game needs a random event
(dice roll) before any player can act. Specifically:

1. **Resource gathering dice roll:** After resolution reaches a resource
   location where a player has figures, and BEFORE the tool decision.
   The chance outcomes are the possible dice sum totals.

2. **Dice-resource card effect:** A civilization card grants "roll 2
   dice for stone" — the chance outcomes are sums 2-12.

3. **Potpourri card effect:** Roll N dice for categorical bonuses.
   The chance outcomes are the possible multisets of face values.

**State machine:** The typical flow at a resource location is:
```
Resolution reaches forest (player has 3 figures there)
  → is_chance_node() = True     [need to roll 3 dice]
  → apply_chance_outcome(7)     [rolled a total of 10, mapped to outcome index 7]
  → is_chance_node() = False
  → sub_phase = TOOL_DECISION   [player decides whether to use tools]
  → apply_action(DONE)          [player declines tools]
  → resources awarded, advance to next location
```

---

### `chance_outcomes() → (outcomes, probs)`

**When called:** When `is_chance_node()` returns True. Called once to
get the distribution, then one outcome is sampled and applied.

**What it must do:** Return the possible outcomes and their probabilities.

**For dice sum rolls (N dice, sum S):**
```python
outcomes = np.arange(5*N + 1)     # indices 0..5N (sums N..6N)
probs[i] = P(sum = N + i)         # pre-computed convolution
```

Example for 3 dice: outcomes [0..15] mapping to sums [3..18],
with probs like [1/216, 3/216, 6/216, 10/216, ...].

**For potpourri (N dice, categorical faces 1-6):**
Each die face maps to a bonus category:
  1=wood, 2=clay, 3=stone, 4=gold, 5=tool, 6=agriculture

The outcomes are distinct sorted multisets of these categories.
For 2 dice: 21 outcomes. For 3 dice: 56 outcomes. For 4 dice: 126.

Each outcome has a calculable probability based on the number of
permutations that produce that multiset.

**Pre-computation:** These probability tables should be pre-computed
once at initialization (in C: static arrays). They don't change.

---

### `apply_chance_outcome(outcome_index: int) → None`

**When called:** After `chance_outcomes()`, once per chance node
encountered during a simulation.

**What it must do:** Apply the specific random outcome. For dice sums:
store the total in `dice_result` and advance to the TOOL_DECISION
sub-phase. For potpourri: store the rolled bonuses in
`potpourri_available` and advance to POTPOURRI_PICK sub-phase.

---

### `is_terminal() → bool`

**When called:** At the top of every SELECT step (to stop traversal),
and in the self-play game loop.

**What it must do:** Return True if the game is over. Game ends after
the round where:
- Not enough civ cards to fill 4 slots, OR
- Any building stack + slot are both empty

After that round completes (including feeding), end-game scoring
happens and the state becomes terminal.

---

### `get_rewards() → np.ndarray`  (shape: `(num_players,)`, dtype: `float32`)

**When called:** When `is_terminal()` returns True. Called once per
MCTS simulation that reaches a terminal state (by playing out the
game), and once at the end of each self-play game.

**What it must do:** Return the terminal reward for each player.
After end-game scoring (green card sets², sand card multipliers,
remaining resources), rank players by final score and assign rewards:

```
2 players: winner → +1.0, loser → -1.0
3 players: 1st → +1.0, 2nd → 0.0, 3rd → -1.0
4 players: 1st → +1.0, 2nd → +0.33, 3rd → -0.33, 4th → -1.0
```

Ties broken by agriculture + tool_value + (figures - 5).

**What MCTS does with it:** Backs the value up through the tree.
Each node in the search path gets this reward vector added to its
`value_sum`.

---

### `current_player() → int`

**When called:** During leaf expansion (to label the node) and in
self-play (to associate training examples with the right reward).

**What it must do:** Return the absolute player index (0-based) of
whoever must make the next decision. Only valid at decision nodes.

---

### `num_players() → int`

**When called:** At initialization and whenever the tree needs to
know the value vector size.

**What it must do:** Return 2, 3, or 4. Constant for the game's lifetime.

---

### `move_number() → int`

**When called:** In self-play, to decide the temperature for action
sampling. Early moves (< threshold) use temperature=1 for diversity;
later moves use temperature→0 for strength.

**What it must do:** Count player decisions (apply_action calls only,
not chance outcomes). Starts at 0.

---

## NeuralNetwork Functions — Detailed Breakdown

### `predict(observation) → (policy, value)`

**When called:** Once per leaf expansion in the non-batched MCTS
(`MCTSSearch.search`). In the batched variant, `predict_batch` is
used instead.

~400 calls per move (one per simulation) in non-batched mode.

**Input:** Raw int32 observation array from `game.get_observation()`.

**What it must do internally:**

1. **Preprocess** the int32 array into float32 NN input:
   ```
   # Normalize resource counts by their max possible values
   obs_float[wood_idx] /= 28.0     # max wood supply
   obs_float[clay_idx] /= 18.0     # max clay supply
   obs_float[food_idx] /= 50.0     # reasonable food cap
   obs_float[score_idx] /= 200.0   # reasonable score cap
   # Booleans stay as 0.0/1.0
   # Card/building IDs: embed via learned embedding layer
   ```

2. **Forward pass** through the network (in eval mode, no gradient):
   ```
   Input(416 float) → FC(416, 512) + ReLU + BN
                     → ResBlock(512) × 3
                     → FC(512, 256) + ReLU
                       ├─ PolicyHead: FC(256, 512) → softmax
                       └─ ValueHead:  FC(256, 128) → ReLU → FC(128, num_players) → tanh
   ```

3. **Return** raw policy (before legal masking) and value vector.

**Output shapes:**
- `policy`: `(action_space_size,)` float32, sums to ~1.0
- `value`:  `(num_players,)` float32, values in [-1, +1]

**What MCTS does with the policy:**
1. Extract entries at legal action indices: `legal_priors = policy[legal_actions]`
2. Renormalize: `legal_priors /= legal_priors.sum()`
3. Store as child priors for PUCT exploration.

**What MCTS does with the value:**
1. Un-rotate from relative (seat 0 = current) to absolute player indices.
2. Back up through the search path: `node.value_sum += abs_value`

---

### `predict_batch(observations) → (policies, values)`

**When called:** By `BatchedMCTS.search_batch()`, which collects leaf
observations from multiple parallel MCTS trees and evaluates them in
one GPU batch.

**Frequency:** Once per simulation step across all trees. If running
32 parallel trees with 400 simulations each, this is called ~400 times
with batch_size=32 (though some entries may be terminal/chance).

**Input:** `(batch_size, observation_size)` int32 array.

**What it must do:** Same as `predict()` but vectorized. The key
performance opportunity: this runs on GPU, so larger batches → better
throughput. Preprocessing is vectorized too.

**Output shapes:**
- `policies`: `(batch_size, action_space_size)` float32
- `values`:   `(batch_size, num_players)` float32

---

### `train_batch(observations, target_policies, target_values) → stats`

**When called:** During the training phase of each AlphaZero iteration.
Called `num_epochs × (buffer_size / batch_size)` times per iteration.

**Inputs:**
- `observations`: `(batch, 416)` int32 — game states from replay buffer.
- `target_policies`: `(batch, 512)` float32 — MCTS visit count
  distributions. Sparse: most entries are 0. Non-zero entries sum to 1.
  These are from the perspective of the player who was deciding.
- `target_values`: `(batch,)` float32 — the terminal reward that
  the deciding player eventually received. In [-1, +1].

**What it must do:**

1. **Preprocess** observations (same as predict).

2. **Forward pass** (in train mode, with dropout if used):
   ```
   pred_policy, pred_value = model(preprocessed_obs)
   ```

3. **Compute loss:**
   ```
   policy_loss = cross_entropy(target_policies, pred_policy)
              = -sum(target * log(pred + 1e-8)) / batch_size

   value_loss = MSE(pred_value[:, 0], target_values)
              # Only index 0 because target is from "self" perspective
              # and obs is rotated so self = seat 0

   total_loss = policy_loss + value_loss
   ```

4. **Backprop + optimizer step:**
   ```
   optimizer.zero_grad()
   total_loss.backward()
   optimizer.step()
   ```

5. **Return** loss components as a dict for logging.

**Why target_values is 1-D (not per-player):**
Each training example stores the reward for the player whose
perspective the observation is from (always seat 0 after rotation).
So we only need to predict and match a single scalar value per example.
The value head still outputs `num_players` values during inference
(for MCTS backup), but during training we only supervise seat 0.

---

### `save(filepath) / load(filepath)`

**When called:** After each training iteration (save), and when
reverting to the best model after a failed evaluation (load).

**What they must do:** Serialize/deserialize:
- Model weights (state_dict)
- Optimizer state (for continued training)
- Any preprocessing parameters (e.g., normalization stats)

---

### `copy_weights_to(other)`

**When called:** When the current model beats the best model in
evaluation, we want to update "best" without filesystem I/O.
Also used to revert to best when evaluation fails.

**What it must do:**
```python
other.model.load_state_dict(self.model.state_dict())
# Optionally copy optimizer state too
```

---

## Data Flow Through a Complete Training Iteration

```
Iteration N
│
├─ SELF-PLAY (100 games)
│   │
│   │  For each game:
│   │    game_factory() → new game
│   │    Loop:
│   │      game.is_chance_node()?
│   │        → game.chance_outcomes()
│   │        → game.apply_chance_outcome(sampled)
│   │      else:
│   │        obs = game.get_observation()         ← 416 int32
│   │        player = game.current_player()       ← int
│   │        policy = MCTS.search(game):          ← 512 float32
│   │          400× { clone, select, expand, evaluate, backup }
│   │          calls: game.clone()                ← 400×
│   │                 game.get_observation()       ← 400×
│   │                 game.legal_actions()         ← 400×
│   │                 nn.predict()                 ← 400×
│   │                 game.apply_action()          ← varies
│   │                 game.apply_chance_outcome()  ← varies
│   │                 game.is_terminal()           ← 400×
│   │                 game.get_rewards()           ← some
│   │        action = sample(policy)
│   │        game.apply_action(action)
│   │    rewards = game.get_rewards()              ← num_players float32
│   │    → TrainingExamples: [(obs, policy, reward_for_player), ...]
│   │
│   └─ All examples → ReplayBuffer
│
├─ TRAINING (10 epochs)
│   │
│   │  For each batch:
│   │    (obs, target_pol, target_val) = buffer.sample(256)
│   │    stats = nn.train_batch(obs, target_pol, target_val)
│   │      → forward, loss, backward, optimizer.step()
│   │
│   └─ Updated model weights
│
├─ EVALUATION (40 games)
│   │
│   │  new_model vs best_model
│   │  Each game: MCTS.search() with no noise, greedy selection
│   │  Win rate > 55%? → new model becomes best
│   │
│   └─ Result: accept or reject
│
└─ CHECKPOINT
    nn.save("checkpoints/model_N.pt")
```
