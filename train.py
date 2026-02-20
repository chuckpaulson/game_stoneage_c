"""
train.py — Launch AlphaZero training for Stone Age

Usage:
    python train.py                     # start fresh
    python train.py --resume            # resume from latest checkpoint
    python train.py --resume --iter 5   # resume from specific iteration

First run settings (conservative, ~1-2 days on M2 Max):
  - 50 MCTS sims/move  (~130ms per move)
  - 20 games per iteration × 100 iterations = 2,000 games
  - hidden=256, blocks=3 (~1.5M params)
  - Eval every iteration with 10 games (reduced from 40 to save time)
"""

import sys, os, logging, argparse, glob, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stone_age_wrapper import StoneAgeGame, ACTION_SIZE, OBS_SIZE
from stone_age_net import StoneAgeNN
from mcts_alphazero import AlphaZeroConfig, AlphaZeroTrainer, set_seed

# ════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════

CHECKPOINT_DIR = "checkpoints"

config = AlphaZeroConfig(
    # Game
    num_players=2,
    action_space_size=ACTION_SIZE,
    observation_size=OBS_SIZE,

    # MCTS — 200 sims gives search enough budget to improve on the policy
    num_simulations=200,
    c_puct=1.5,
    dirichlet_alpha=0.15,         # sharper noise → focused exploration
    dirichlet_fraction=0.4,       # more noise at root → discover alternatives
    temperature_move_threshold=30,

    # Self-play — 256 games per iteration, ~400 examples each = ~100K per iter
    num_self_play_games=256,
    max_game_moves=1000,

    # Training
    num_iterations=100,
    num_epochs=4,                  # reduced from 10 — avoid overfitting the buffer
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-4,
    replay_buffer_size=200_000,
    min_replay_size=2000,       # start training after ~4 iterations

    # Evaluation — reduced to 10 games to save time early on
    eval_games=10,
    eval_threshold=0.55,

    # Infrastructure
    checkpoint_dir=CHECKPOINT_DIR,
    log_interval=5,
    seed=42,
)


# ════════════════════════════════════════════════════════════
# Setup
# ════════════════════════════════════════════════════════════

def setup_logging():
    """Configure logging to both console and file."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_file = os.path.join(CHECKPOINT_DIR, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a'),
        ],
    )


def make_game():
    """Game factory — each call returns a fresh game with random seed."""
    seed = np.random.randint(0, 2**31)
    return StoneAgeGame(num_players=config.num_players, seed=seed)


def find_latest_checkpoint():
    """Find the most recent model checkpoint."""
    pattern = os.path.join(CHECKPOINT_DIR, "model_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, -1
    # Try numbered checkpoints first, fall back to latest file
    for f in reversed(files):
        basename = os.path.basename(f)
        part = basename.split("_")[1].split(".")[0]
        try:
            return f, int(part)
        except ValueError:
            continue
    # Only non-numeric checkpoints (e.g. model_interrupted.pt)
    return files[-1], 0


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stone Age AlphaZero Training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--iter", type=int, default=None,
                        help="Resume from specific iteration number")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    set_seed(config.seed)

    # Create network
    nn = StoneAgeNN(
        num_players=config.num_players,
        hidden=256,
        blocks=3,
        lr=config.learning_rate,
        wd=config.weight_decay,
    )

    n_params = sum(p.numel() for p in nn.model.parameters())
    logger.info("=" * 60)
    logger.info("Stone Age AlphaZero Training")
    logger.info("=" * 60)
    logger.info("Network: %s params on %s", f"{n_params:,}", nn.device)
    logger.info("MCTS: %d sims/move", config.num_simulations)
    logger.info("Self-play: %d games/iter × %d iters = %d total games",
                config.num_self_play_games, config.num_iterations,
                config.num_self_play_games * config.num_iterations)

    # Resume from checkpoint?
    if args.resume or args.iter is not None:
        if args.iter is not None:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_{args.iter:04d}.pt")
        else:
            ckpt_path, _ = find_latest_checkpoint()

        if ckpt_path and os.path.exists(ckpt_path):
            nn.load(ckpt_path)
            logger.info("Resumed from checkpoint: %s", ckpt_path)
        else:
            logger.warning("No checkpoint found, starting fresh")

    # Estimate timing
    logger.info("")
    logger.info("  Self-play: %d games × ~500 moves × %d sims × ~1.7ms/batch ≈ %.0f min",
                config.num_self_play_games, config.num_simulations,
                config.num_self_play_games * 500 * config.num_simulations * 0.0017 / 60 / config.num_self_play_games)
    logger.info("  Training:  ~50s")
    logger.info("  Eval:      disabled")
    logger.info("")

    # Create trainer and run
    trainer = AlphaZeroTrainer(config, make_game, nn)

    try:
        trainer.run()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Training interrupted by user (Ctrl+C)")
        logger.info("Saving emergency checkpoint...")
        nn.save(os.path.join(CHECKPOINT_DIR, "model_interrupted.pt"))
        trainer.replay_buffer.save(
            os.path.join(CHECKPOINT_DIR, "buffer_interrupted.pkl"))
        logger.info("Saved. Resume later with: python train.py --resume")

    logger.info("Done!")


if __name__ == "__main__":
    main()
