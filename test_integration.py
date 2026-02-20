"""
test_integration.py — Integration tests for Stone Age AlphaZero pipeline

Tests 4 levels:
  1. GameInterface compliance (wrapper matches ABC contract)
  2. NeuralNetwork smoke test (shapes, normalization, train step)
  3. MCTS search (single position search, policy validity)
  4. Mini self-play + training loop (end-to-end pipeline)

Usage:
    python test_integration.py          # runs all tests
    python test_integration.py --quick  # skip slow MCTS/training tests
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stone_age_wrapper import StoneAgeGame, ACTION_SIZE, OBS_SIZE, STATE_SIZE
from mcts_alphazero import (
    AlphaZeroConfig, MCTSSearch, MCTSNode, NeuralNetwork,
    SelfPlayManager, BatchedSelfPlayManager, ReplayBuffer, TrainingExample,
)

# ════════════════════════════════════════════════════════════
# Try importing the real NN; fall back to a mock
# ════════════════════════════════════════════════════════════

HAS_TORCH = False
try:
    import torch
    from stone_age_net import StoneAgeNN
    HAS_TORCH = True
except ImportError:
    pass


class MockNN(NeuralNetwork):
    """Random policy + zero value, for testing without torch."""

    def __init__(self, num_players=2):
        self._np = num_players

    def predict(self, observation):
        pol = np.random.dirichlet(np.ones(ACTION_SIZE)).astype(np.float32)
        val = np.zeros(self._np, dtype=np.float32)
        return pol, val

    def predict_batch(self, observations):
        n = observations.shape[0]
        pols = np.array([np.random.dirichlet(np.ones(ACTION_SIZE))
                         for _ in range(n)], dtype=np.float32)
        vals = np.zeros((n, self._np), dtype=np.float32)
        return pols, vals

    def train_batch(self, observations, target_policies, target_values):
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

    def copy_weights_to(self, other):
        pass


def make_nn(num_players=2):
    if HAS_TORCH:
        return StoneAgeNN(num_players=num_players, hidden=128, blocks=2)
    return MockNN(num_players)


# ════════════════════════════════════════════════════════════
# Test helpers
# ════════════════════════════════════════════════════════════

passed = 0
failed = 0


def check(condition, msg):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  ✗ FAIL: {msg}")


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ════════════════════════════════════════════════════════════
# §1  GameInterface compliance
# ════════════════════════════════════════════════════════════

def test_game_interface():
    section("§1  GameInterface compliance")

    for np_count in [2, 3, 4]:
        g = StoneAgeGame(num_players=np_count, seed=42)

        # Basic queries
        check(g.num_players() == np_count,
              f"{np_count}p: num_players()={g.num_players()}")
        check(g.move_number() == 0,
              f"{np_count}p: initial move_number()={g.move_number()}")
        check(not g.is_terminal(),
              f"{np_count}p: is_terminal() at start")

        # Observation
        obs = g.get_observation()
        check(obs.shape == (OBS_SIZE,),
              f"{np_count}p: obs shape={obs.shape}, expected ({OBS_SIZE},)")
        check(obs.dtype == np.int32,
              f"{np_count}p: obs dtype={obs.dtype}")

        # Legal actions
        if not g.is_chance_node():
            actions = g.legal_actions()
            check(len(actions) > 0,
                  f"{np_count}p: no legal actions at start")
            check(actions.dtype == np.int32,
                  f"{np_count}p: actions dtype={actions.dtype}")
            check(all(0 <= a < ACTION_SIZE for a in actions),
                  f"{np_count}p: action out of range: {actions}")
            check(len(set(actions)) == len(actions),
                  f"{np_count}p: duplicate actions")

            # Legal actions mask
            mask = g.legal_actions_mask()
            check(mask.shape == (ACTION_SIZE,),
                  f"{np_count}p: mask shape={mask.shape}")
            check(mask.sum() == len(actions),
                  f"{np_count}p: mask count={mask.sum()} != actions={len(actions)}")
            check(all(mask[a] for a in actions),
                  f"{np_count}p: mask missing legal action")

            # Current player
            cp = g.current_player()
            check(0 <= cp < np_count,
                  f"{np_count}p: current_player()={cp}")

        # Clone independence
        g2 = g.clone()
        obs2 = g2.get_observation()
        check(np.array_equal(obs, obs2),
              f"{np_count}p: clone obs mismatch")

        # Mutate clone, verify original unchanged
        if not g2.is_chance_node():
            a = int(g2.legal_actions()[0])
            g2.apply_action(a)
            obs_after = g.get_observation()
            check(np.array_equal(obs, obs_after),
                  f"{np_count}p: clone mutation affected original")

    # Play full random games to verify no crashes
    for np_count in [2, 3, 4]:
        g = StoneAgeGame(num_players=np_count, seed=99)
        moves = 0
        while not g.is_terminal() and moves < 5000:
            if g.is_chance_node():
                outcomes, probs = g.chance_outcomes()
                check(len(outcomes) > 0,
                      f"{np_count}p: empty chance outcomes at move {moves}")
                check(abs(probs.sum() - 1.0) < 0.01,
                      f"{np_count}p: chance probs sum={probs.sum()}")
                i = np.random.choice(len(outcomes), p=probs)
                g.apply_chance_outcome(int(outcomes[i]))
            else:
                actions = g.legal_actions()
                g.apply_action(int(actions[np.random.randint(len(actions))]))
                moves += 1

        check(g.is_terminal(),
              f"{np_count}p: game didn't terminate in {moves} moves")

        rewards = g.get_rewards()
        check(rewards.shape == (np_count,),
              f"{np_count}p: rewards shape={rewards.shape}")
        check(all(-1.01 <= r <= 1.01 for r in rewards),
              f"{np_count}p: rewards out of range: {rewards}")
        check(g.move_number() == moves,
              f"{np_count}p: move_number()={g.move_number()} != {moves}")

        print(f"  ✓ {np_count}p: {moves} moves, rewards={rewards}")

    print(f"  ✓ GameInterface compliance passed")


# ════════════════════════════════════════════════════════════
# §2  NeuralNetwork smoke test
# ════════════════════════════════════════════════════════════

def test_neural_net():
    section("§2  NeuralNetwork smoke test")

    if not HAS_TORCH:
        print("  ⚠ torch not available, testing MockNN only")

    nn = make_nn(num_players=2)

    # Single predict with real game observation
    g = StoneAgeGame(num_players=2, seed=42)
    obs = g.get_observation()
    pol, val = nn.predict(obs)

    check(pol.shape == (ACTION_SIZE,),
          f"policy shape={pol.shape}, expected ({ACTION_SIZE},)")
    check(val.shape == (2,),
          f"value shape={val.shape}, expected (2,)")
    check(abs(pol.sum() - 1.0) < 0.01,
          f"policy sum={pol.sum():.4f}")
    check(all(-1.01 <= v <= 1.01 for v in val),
          f"value out of range: {val}")
    check(all(np.isfinite(pol)),
          "policy contains non-finite values")
    check(all(np.isfinite(val)),
          "value contains non-finite values")

    # Policy masking: verify legal actions get nonzero probability
    if not g.is_chance_node():
        mask = g.legal_actions_mask()
        masked_pol = pol * mask
        check(masked_pol.sum() > 0,
              "all legal moves have zero policy mass")

    # Batch predict
    batch_obs = np.stack([g.get_observation() for _ in range(8)])
    pols, vals = nn.predict_batch(batch_obs)
    check(pols.shape == (8, ACTION_SIZE),
          f"batch policy shape={pols.shape}")
    check(vals.shape == (8, 2),
          f"batch value shape={vals.shape}")

    # Training step (only with real NN)
    if HAS_TORCH:
        tp = np.zeros((8, ACTION_SIZE), dtype=np.float32)
        for i in range(8):
            a = np.random.choice(ACTION_SIZE, 3, replace=False)
            tp[i, a] = 1.0 / 3
        tv = np.random.uniform(-1, 1, 8).astype(np.float32)
        stats = nn.train_batch(batch_obs, tp, tv)
        check("loss" in stats and stats["loss"] > 0,
              f"train_batch returned: {stats}")
        check("policy_loss" in stats,
              "missing policy_loss in train stats")
        check("value_loss" in stats,
              "missing value_loss in train stats")
        print(f"  ✓ Training step: {stats}")

    print(f"  ✓ NeuralNetwork smoke test passed")


# ════════════════════════════════════════════════════════════
# §3  MCTS search
# ════════════════════════════════════════════════════════════

def test_mcts_search():
    section("§3  MCTS search")

    nn = make_nn(num_players=2)

    config = AlphaZeroConfig(
        num_players=2,
        action_space_size=ACTION_SIZE,
        observation_size=OBS_SIZE,
        num_simulations=30,
        c_puct=1.5,
    )

    game = StoneAgeGame(num_players=2, seed=42)

    # Advance past any initial chance nodes
    while game.is_chance_node():
        outcomes, probs = game.chance_outcomes()
        i = np.random.choice(len(outcomes), p=probs)
        game.apply_chance_outcome(int(outcomes[i]))

    legal = game.legal_actions()
    print(f"  Position: player={game.current_player()}, "
          f"{len(legal)} legal actions, move #{game.move_number()}")

    mcts = MCTSSearch(config, nn)
    t0 = time.time()
    policy = mcts.search(game, add_noise=True)
    dt = time.time() - t0

    check(policy.shape == (ACTION_SIZE,),
          f"MCTS policy shape={policy.shape}")
    check(abs(policy.sum() - 1.0) < 0.01,
          f"MCTS policy sum={policy.sum():.4f}")

    # Only legal actions should have mass
    mask = game.legal_actions_mask()
    illegal_mass = policy[~mask].sum()
    check(illegal_mass < 0.001,
          f"MCTS policy has {illegal_mass:.4f} mass on illegal actions")

    # At least one action should have visits
    check(policy.max() > 0,
          "MCTS policy is all zeros")

    # Game state should not be corrupted by search
    check(game.num_players() == 2,
          "game corrupted after MCTS")

    top_actions = np.argsort(policy)[-5:][::-1]
    top_str = ", ".join(f"a{a}={policy[a]:.3f}" for a in top_actions if policy[a] > 0)
    print(f"  Top actions: {top_str}")
    print(f"  ✓ MCTS search: {config.num_simulations} sims in {dt:.2f}s "
          f"({dt/config.num_simulations*1000:.1f} ms/sim)")

    # Second search from mid-game position
    game2 = StoneAgeGame(num_players=2, seed=123)
    moves_done = 0
    for _ in range(100):
        if game2.is_terminal():
            break
        if game2.is_chance_node():
            outcomes, probs = game2.chance_outcomes()
            i = np.random.choice(len(outcomes), p=probs)
            game2.apply_chance_outcome(int(outcomes[i]))
        else:
            actions = game2.legal_actions()
            game2.apply_action(int(actions[np.random.randint(len(actions))]))
            moves_done += 1

    if not game2.is_terminal() and not game2.is_chance_node():
        policy2 = mcts.search(game2, add_noise=False)
        check(abs(policy2.sum() - 1.0) < 0.01,
              f"mid-game MCTS policy sum={policy2.sum():.4f}")
        print(f"  ✓ Mid-game search OK (move #{game2.move_number()})")

    print(f"  ✓ MCTS search tests passed")


# ════════════════════════════════════════════════════════════
# §4  Mini self-play + training
# ════════════════════════════════════════════════════════════

def test_self_play_training():
    section("§4  Self-play + training pipeline")

    nn = make_nn(num_players=2)
    seed_counter = [0]

    def game_factory():
        seed_counter[0] += 1
        return StoneAgeGame(num_players=2, seed=seed_counter[0])

    config = AlphaZeroConfig(
        num_players=2,
        action_space_size=ACTION_SIZE,
        observation_size=OBS_SIZE,
        num_simulations=10,       # very fast for testing
        c_puct=1.5,
        temperature_move_threshold=15,
        max_game_moves=2000,      # safety cap (games avg ~500 moves)
        batch_size=32,
        min_replay_size=32,
    )

    # Play self-play games using batched MCTS
    print(f"  Playing 2 self-play games (batched, {config.num_simulations} sims/move)...")
    sp = BatchedSelfPlayManager(config, game_factory, nn, batch_size=2)
    t0 = time.time()
    examples = sp.play_games(2)
    dt = time.time() - t0

    check(len(examples) > 0,
          "self-play produced 0 examples")
    check(all(isinstance(e, TrainingExample) for e in examples),
          "examples are not TrainingExample instances")

    ex = examples[0]
    check(ex.observation.shape == (OBS_SIZE,),
          f"example obs shape={ex.observation.shape}")
    check(ex.policy.shape == (ACTION_SIZE,),
          f"example policy shape={ex.policy.shape}")
    check(abs(ex.policy.sum() - 1.0) < 0.01,
          f"example policy sum={ex.policy.sum():.4f}")
    check(-1.01 <= ex.value <= 1.01,
          f"example value={ex.value} out of range")

    # Check that examples have both +1 and -1 values (winner and loser)
    values = set(e.value for e in examples)
    check(len(values) >= 2,
          f"all examples have same value: {values}")

    n_moves = max(len(examples), 1)
    print(f"  ✓ Batched self-play: {len(examples)} examples in {dt:.1f}s "
          f"({dt/n_moves*1000:.0f} ms/example)")

    # Build replay buffer and sample
    buf = ReplayBuffer(max_size=10000)
    buf.add_batch(examples)
    check(len(buf) == len(examples),
          f"buffer size={len(buf)} != {len(examples)}")

    if len(buf) >= config.batch_size:
        obs_b, pol_b, val_b = buf.sample(config.batch_size)
        check(obs_b.shape == (config.batch_size, OBS_SIZE),
              f"sample obs shape={obs_b.shape}")
        check(pol_b.shape == (config.batch_size, ACTION_SIZE),
              f"sample policy shape={pol_b.shape}")
        check(val_b.shape == (config.batch_size,),
              f"sample values shape={val_b.shape}")
        print(f"  ✓ Replay buffer sample OK")

        # Training step
        if HAS_TORCH:
            stats = nn.train_batch(obs_b, pol_b, val_b)
            check(stats["loss"] > 0,
                  f"training loss={stats['loss']}")
            check(np.isfinite(stats["loss"]),
                  f"training loss is not finite: {stats['loss']}")
            print(f"  ✓ Training on self-play data: {stats}")

            # Second training step
            stats2 = nn.train_batch(obs_b, pol_b, val_b)
            print(f"  ✓ Second train step: loss {stats['loss']:.3f} → {stats2['loss']:.3f}")
    else:
        print(f"  ⚠ Only {len(buf)} examples, need {config.batch_size} for training test")

    print(f"  ✓ Self-play + training pipeline passed")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow MCTS and training tests")
    args = parser.parse_args()

    print("=" * 60)
    print("  Stone Age AlphaZero — Integration Tests")
    print("=" * 60)
    print(f"  Engine: STATE={STATE_SIZE} OBS={OBS_SIZE} ACT={ACTION_SIZE}")
    print(f"  PyTorch: {'available' if HAS_TORCH else 'NOT available (using MockNN)'}")
    if HAS_TORCH:
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

    t_start = time.time()

    test_game_interface()
    test_neural_net()

    if not args.quick:
        test_mcts_search()
        test_self_play_training()
    else:
        print("\n  (skipping §3 MCTS and §4 self-play — use without --quick for full test)")

    dt = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed — {dt:.1f}s")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    print("  All tests passed!")


if __name__ == "__main__":
    main()
