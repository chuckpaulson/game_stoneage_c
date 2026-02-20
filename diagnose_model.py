"""
diagnose_model.py — Inspect what a trained model thinks about game positions.

Shows top actions, value estimates, and whether the model plays reasonably.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stone_age_wrapper import StoneAgeGame, ACTION_SIZE, OBS_SIZE

# Borrow action_name from trace_game
from trace_game import action_name

def diagnose(model_path, num_seeds=3):
    from stone_age_net import StoneAgeNN
    
    nn = StoneAgeNN(num_players=2, hidden=256, blocks=3)
    nn.load(model_path)
    print(f"Model: {model_path}")
    print(f"Device: {nn.device}")
    print()

    for seed in range(1, num_seeds + 1):
        print(f"{'=' * 60}")
        print(f"  Game seed={seed}")
        print(f"{'=' * 60}")
        
        game = StoneAgeGame(num_players=2, seed=seed)
        
        # Advance past initial chance nodes
        while game.is_chance_node() and not game.is_terminal():
            outcomes, probs = game.chance_outcomes()
            idx = np.random.choice(len(outcomes), p=probs)
            game.apply_chance_outcome(int(outcomes[idx]))
        
        # Check a few decision points
        move_num = 0
        for step in range(20):
            if game.is_terminal():
                print(f"  Game ended at move {move_num}")
                break
            if game.is_chance_node():
                outcomes, probs = game.chance_outcomes()
                idx = np.random.choice(len(outcomes), p=probs)
                game.apply_chance_outcome(int(outcomes[idx]))
                continue
            
            current = game.current_player()
            obs = game.get_observation()
            policy, value = nn.predict(obs)
            
            legal = game.legal_actions()
            mask = game.legal_actions_mask()
            
            # Mask and get top actions
            masked_policy = policy.copy()
            masked_policy[~mask] = 0
            total_legal_mass = masked_policy.sum()
            
            # Sort legal actions by policy weight
            legal_probs = [(int(a), float(policy[a])) for a in legal]
            legal_probs.sort(key=lambda x: -x[1])
            
            print(f"\n  Move {move_num}, P{current}, {len(legal)} legal actions")
            print(f"  Value: P0={value[0]:.4f}  P1={value[1]:.4f}")
            print(f"  Legal mass: {total_legal_mass:.4f} / total policy sum: {policy.sum():.4f}")
            
            # Show top 5
            print(f"  Top actions:")
            for a, p in legal_probs[:5]:
                aname = action_name(a, state=game.state)
                print(f"    {p:.4f}  a{a:3d}  {aname}")
            
            # Show what fraction goes to illegal actions
            illegal_mass = policy[~mask].sum()
            if illegal_mass > 0.01:
                print(f"  ⚠ {illegal_mass:.4f} probability on ILLEGAL actions!")
                # Top illegal
                illegal_probs = [(int(a), float(policy[a])) for a in range(ACTION_SIZE) if not mask[a] and policy[a] > 0.001]
                illegal_probs.sort(key=lambda x: -x[1])
                for a, p in illegal_probs[:3]:
                    print(f"    illegal: {p:.4f}  a{a:3d}")
            
            # Play the model's top choice
            action = int(np.argmax(masked_policy))
            aname = action_name(action, state=game.state)
            print(f"  → plays: a{action} {aname}")
            
            game.apply_action(action)
            move_num += 1
            
            # Advance past chance nodes
            while game.is_chance_node() and not game.is_terminal():
                outcomes, probs = game.chance_outcomes()
                idx = np.random.choice(len(outcomes), p=probs)
                game.apply_chance_outcome(int(outcomes[idx]))
        
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model checkpoint")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    diagnose(args.model, args.seeds)
