"""
tournament.py — Play matches between two Stone Age players.

Each player can be:
  - A trained model checkpoint (with optional MCTS search)
  - A random player (uniform random legal moves)

Usage:
    # Trained model vs random
    python tournament.py --p1 checkpoints/model_0018.pt --p2 random --games 50

    # Two checkpoints against each other
    python tournament.py --p1 checkpoints/model_0050.pt --p2 checkpoints/model_0018.pt --games 50

    # With MCTS search (slower but stronger)
    python tournament.py --p1 checkpoints/model_0050.pt --p1-sims 100 --p2 random --games 20

    # Random vs random baseline
    python tournament.py --p1 random --p2 random --games 100
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stone_age_wrapper import StoneAgeGame, ACTION_SIZE, OBS_SIZE
from mcts_alphazero import AlphaZeroConfig, MCTSSearch


def make_random_player():
    """Returns a player function that picks uniformly at random."""
    def play(game):
        actions = game.legal_actions()
        return int(actions[np.random.randint(len(actions))])
    return play


def make_nn_player(model_path, num_players, sims=0):
    """
    Returns a player function using a trained neural network.
    sims=0: raw policy (fast). sims>0: MCTS search (stronger).
    """
    from stone_age_net import StoneAgeNN

    nn = StoneAgeNN(num_players=num_players, hidden=256, blocks=3)
    nn.load(model_path)

    if sims > 0:
        config = AlphaZeroConfig(
            num_players=num_players,
            action_space_size=ACTION_SIZE,
            observation_size=OBS_SIZE,
            num_simulations=sims,
        )
        mcts = MCTSSearch(config, nn)

        def play(game):
            policy = mcts.search(game, add_noise=False)
            return int(np.argmax(policy))
    else:
        def play(game):
            obs = game.get_observation()
            policy, _ = nn.predict(obs)
            mask = game.legal_actions_mask()
            policy[~mask] = 0
            return int(np.argmax(policy))

    return play


def play_one_game(p1_play, p2_play, num_players=2, seed=None):
    """
    Play one game. Returns (p1_reward, p2_reward, scores, moves).
    P1 is always player 0, P2 is always player 1.
    """
    if seed is None:
        seed = np.random.randint(0, 2**31)
    game = StoneAgeGame(num_players=num_players, seed=seed)
    players = [p1_play, p2_play]
    moves = 0

    while not game.is_terminal():
        if game.is_chance_node():
            outcomes, probs = game.chance_outcomes()
            idx = np.random.choice(len(outcomes), p=probs)
            game.apply_chance_outcome(int(outcomes[idx]))
        else:
            current = game.current_player()
            action = players[current](game)
            game.apply_action(action)
            moves += 1

    rewards = game.get_rewards()
    # Extract raw scores from state for display
    # S_SCORE offsets: player block starts at index dependent on engine
    return float(rewards[0]), float(rewards[1]), moves


def main():
    parser = argparse.ArgumentParser(description="Stone Age Tournament")
    parser.add_argument("--p1", type=str, required=True,
                        help="Player 1: 'random' or path to model checkpoint")
    parser.add_argument("--p2", type=str, required=True,
                        help="Player 2: 'random' or path to model checkpoint")
    parser.add_argument("--p1-sims", type=int, default=0,
                        help="MCTS sims for P1 (0 = raw policy, default: 0)")
    parser.add_argument("--p2-sims", type=int, default=0,
                        help="MCTS sims for P2 (0 = raw policy, default: 0)")
    parser.add_argument("--games", type=int, default=50,
                        help="Number of games (default: 50)")
    parser.add_argument("--players", type=int, default=2,
                        help="Number of players (default: 2)")
    parser.add_argument("--swap", action="store_true",
                        help="Play each game twice with swapped seats (doubles game count)")
    args = parser.parse_args()

    # Build players
    p1_name = "random" if args.p1 == "random" else f"{os.path.basename(args.p1)}({args.p1_sims}sim)"
    p2_name = "random" if args.p2 == "random" else f"{os.path.basename(args.p2)}({args.p2_sims}sim)"

    print(f"{'=' * 60}")
    print(f"  Stone Age Tournament — {args.games} games")
    print(f"  P1: {p1_name}")
    print(f"  P2: {p2_name}")
    if args.swap:
        print(f"  Seat swapping: ON (total {args.games * 2} games)")
    print(f"{'=' * 60}")

    if args.p1 == "random":
        p1_play = make_random_player()
    else:
        p1_play = make_nn_player(args.p1, args.players, args.p1_sims)
        print(f"  Loaded P1 model: {args.p1}")

    if args.p2 == "random":
        p2_play = make_random_player()
    else:
        p2_play = make_nn_player(args.p2, args.players, args.p2_sims)
        print(f"  Loaded P2 model: {args.p2}")

    print()

    # Run games
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_moves = 0
    results = []

    matchups = [(p1_play, p2_play, False)]
    if args.swap:
        matchups.append((p2_play, p1_play, True))

    games_played = 0

    for game_num in range(args.games):
        seed = np.random.randint(0, 2**31)

        for seat0_play, seat1_play, swapped in matchups:
            r0, r1, moves = play_one_game(seat0_play, seat1_play, args.players, seed)
            total_moves += moves
            games_played += 1

            # Map rewards back to P1/P2
            if swapped:
                p1_r, p2_r = r1, r0
            else:
                p1_r, p2_r = r0, r1

            if p1_r > p2_r:
                p1_wins += 1
                result = "P1"
            elif p2_r > p1_r:
                p2_wins += 1
                result = "P2"
            else:
                draws += 1
                result = "draw"

            results.append((result, moves))

            if games_played % 10 == 0:
                pct = p1_wins / games_played * 100
                print(f"  [{games_played}/{args.games * len(matchups)}] "
                      f"P1: {p1_wins}  P2: {p2_wins}  draws: {draws}  "
                      f"(P1 win rate: {pct:.1f}%)")

    # Final summary
    total = p1_wins + p2_wins + draws
    avg_moves = total_moves / total

    print()
    print(f"{'=' * 60}")
    print(f"  RESULTS ({total} games)")
    print(f"{'=' * 60}")
    print(f"  P1 ({p1_name}):  {p1_wins} wins  ({p1_wins/total*100:.1f}%)")
    print(f"  P2 ({p2_name}):  {p2_wins} wins  ({p2_wins/total*100:.1f}%)")
    print(f"  Draws: {draws}  ({draws/total*100:.1f}%)")
    print(f"  Avg moves/game: {avg_moves:.0f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
