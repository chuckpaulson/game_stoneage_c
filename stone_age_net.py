"""
stone_age_net.py — Policy-Value network for Stone Age AlphaZero

Implements the NeuralNetwork ABC from mcts_alphazero.py.
Architecture: residual MLP with separate policy and value heads.

    Input(424 int32) → normalize → FC(hidden) → [ResBlock × N] → trunk(256)
      ├─ Policy: FC(406) → softmax
      └─ Value:  FC(128) → FC(num_players) → tanh

Device auto-detection: MPS (Apple Silicon) → CUDA → CPU

Build:
    Requires: pip install torch
    The C engine (libstoneage.so) is NOT needed by this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcts_alphazero import NeuralNetwork

OBS_SIZE = 424
ACT_SIZE = 409


# ════════════════════════════════════════════════════════════
# Observation normalization
# ════════════════════════════════════════════════════════════
# Static divisors that scale each obs field to roughly [0, 1].
# Matches the 424-element layout from stone_age_state.py / stone_age.h.

def _build_norm():
    d = np.ones(OBS_SIZE, dtype=np.float32)

    # Meta [0:4]
    d[0] = 4; d[1] = 20; d[2] = 4; d[3] = 4

    # Phase [4:41]
    d[4] = 4; d[5] = 4; d[6] = 8; d[7] = 4          # phase, active, sub, res_player
    # [8:12] resolution_completed — bool, fine at 1
    # [12:28] locations_resolved — bool, fine at 1
    d[28] = 42; d[29] = 5; d[30] = 4; d[31] = 4      # dice, pend_res, pend_slot, rem_picks
    # [32:36] potpourri_available — small ints, fine
    # [36:40] players_fed — bool
    d[40] = 10                                          # food_still_needed

    # Locations [41:73]: figure counts per loc×player
    # 8 locations × 4 player slots: forest, clay, quarry, river (cap 7), hunt (cap 10),
    # toolmaker, hut, field (cap 1-2)
    d[41:73] = 7
    d[57:61] = 10   # hunt (location 4) allows up to 10 figures

    # Board misc [73:78]
    d[73] = 3                                           # blocked_village
    d[74] = 28; d[75] = 18; d[76] = 12; d[77] = 10    # supply w,c,s,g

    # Card slots [78:86]: (id, fig) × 4
    for i in range(78, 86, 2):
        d[i] = 34; d[i + 1] = 4

    # Building slots [86:94]: (id, fig) × 4
    for i in range(86, 94, 2):
        d[i] = 22; d[i + 1] = 4

    # Player blocks [94:386]: 4 × 73
    for pi in range(4):
        b = 94 + pi * 73
        d[b:b + 4] = [28, 18, 12, 10]     # resources w,c,s,g
        d[b + 4] = 50                      # food
        d[b + 5] = 10; d[b + 6] = 10      # figures, figures_available
        d[b + 7] = 10                      # agriculture
        d[b + 8:b + 11] = 4               # tools
        # [+11:+14] tools_used — bool
        d[b + 14] = 200                    # score
        d[b + 15] = 4; d[b + 16] = 4      # one-time tool counts
        d[b + 17:b + 51] = 3              # civ_counts[34]
        d[b + 51:b + 73] = 3              # bld_counts[22]

    # Deck [386:420]
    d[386:420] = 3

    # Stacks [420:424]
    d[420:424] = 7

    return d

_NORM = _build_norm()


# ════════════════════════════════════════════════════════════
# Network architecture
# ════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Pre-activation residual block: BN → ReLU → FC → BN → ReLU → FC + skip."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return F.relu(out + x)


class StoneAgeNet(nn.Module):
    def __init__(self, num_players=2, hidden=512, blocks=4):
        super().__init__()
        self.input_fc = nn.Linear(OBS_SIZE, hidden)
        self.input_bn = nn.BatchNorm1d(hidden)
        self.res_blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
        self.trunk_fc = nn.Linear(hidden, 256)
        self.trunk_bn = nn.BatchNorm1d(256)

        # Policy head
        self.pol_fc = nn.Linear(256, ACT_SIZE)

        # Value head
        self.val_fc1 = nn.Linear(256, 128)
        self.val_fc2 = nn.Linear(128, num_players)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_fc(x)))
        for block in self.res_blocks:
            x = block(x)
        x = F.relu(self.trunk_bn(self.trunk_fc(x)))

        policy = F.softmax(self.pol_fc(x), dim=-1)
        value = torch.tanh(self.val_fc2(F.relu(self.val_fc1(x))))
        return policy, value


# ════════════════════════════════════════════════════════════
# NeuralNetwork interface implementation
# ════════════════════════════════════════════════════════════

def _auto_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class StoneAgeNN(NeuralNetwork):
    """
    Full NeuralNetwork interface for Stone Age AlphaZero.

    Args:
        num_players: number of player seats (value head width)
        hidden:      hidden layer width (512 for production, 256 for fast dev)
        blocks:      number of residual blocks (4 recommended)
        lr:          learning rate for Adam
        wd:          weight decay (L2 regularization)
        device:      "mps", "cuda", "cpu", or None for auto-detect
    """

    def __init__(self, num_players=2, hidden=512, blocks=4,
                 lr=1e-3, wd=1e-4, device=None):
        self.device = torch.device(device or _auto_device())
        self.num_players = num_players

        self.model = StoneAgeNet(num_players, hidden, blocks).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=wd
        )

        # Static normalization divisors — registered so .to(device) moves them
        self._norm = torch.from_numpy(_NORM).to(self.device)

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """int32 numpy obs → normalized float32 tensor on device."""
        t = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t / self._norm

    # ── Inference ──

    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(self._to_tensor(observation))
        return policy[0].cpu().numpy(), value[0].cpu().numpy()

    def predict_batch(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(self._to_tensor(observations))
        return policy.cpu().numpy(), value.cpu().numpy()

    # ── Training ──

    def train_batch(
        self,
        observations: np.ndarray,       # (batch, 424) int32
        target_policies: np.ndarray,     # (batch, 406) float32
        target_values: np.ndarray,       # (batch,) float32
    ) -> Dict[str, float]:
        self.model.train()

        x = self._to_tensor(observations)
        tp = torch.from_numpy(target_policies).float().to(self.device)
        tv = torch.from_numpy(target_values).float().to(self.device)

        pred_pol, pred_val = self.model(x)

        # Policy: cross-entropy with MCTS visit distribution
        policy_loss = -(tp * torch.log(pred_pol + 1e-8)).sum(dim=1).mean()

        # Value: MSE on seat 0 (= current player, due to obs rotation)
        value_loss = F.mse_loss(pred_val[:, 0], tv)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    # ── Persistence ──

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_players": self.num_players,
        }, filepath)

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def copy_weights_to(self, other: "StoneAgeNN") -> None:
        other.model.load_state_dict(self.model.state_dict())


# ════════════════════════════════════════════════════════════
# Self-test
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    net = StoneAgeNN(num_players=2, hidden=256, blocks=3)
    n_params = sum(p.numel() for p in net.model.parameters())
    print(f"Model: {n_params:,} params on {net.device}")

    # Single predict
    obs = np.random.randint(-1, 10, size=OBS_SIZE, dtype=np.int32)
    pol, val = net.predict(obs)
    assert pol.shape == (ACT_SIZE,)
    assert val.shape == (2,)
    assert abs(pol.sum() - 1.0) < 0.01
    assert all(-1 <= v <= 1 for v in val)
    print(f"predict:       policy sum={pol.sum():.4f}, value={val}")

    # Batch predict
    batch = np.random.randint(-1, 10, size=(32, OBS_SIZE), dtype=np.int32)
    pols, vals = net.predict_batch(batch)
    assert pols.shape == (32, ACT_SIZE)
    assert vals.shape == (32, 2)
    print(f"predict_batch: policies={pols.shape}, values={vals.shape}")

    # Train
    tp = np.zeros((32, ACT_SIZE), dtype=np.float32)
    for i in range(32):
        a = np.random.choice(ACT_SIZE, 5, replace=False)
        tp[i, a] = 0.2
    tv = np.random.uniform(-1, 1, 32).astype(np.float32)
    stats = net.train_batch(batch, tp, tv)
    print(f"train_batch:   {stats}")

    # Save / load round-trip
    net.save("/tmp/sa_test.pt")
    net2 = StoneAgeNN(num_players=2, hidden=256, blocks=3)
    net2.load("/tmp/sa_test.pt")
    # Compare two fresh predictions from same weights (avoids MPS non-determinism)
    pol_a, _ = net.predict(obs)
    pol_b, _ = net2.predict(obs)
    diff = np.abs(pol_a - pol_b).max()
    assert diff < 0.01, f"save/load max diff = {diff}"
    print(f"save/load:     max diff = {diff:.8f}")

    # copy_weights_to
    net3 = StoneAgeNN(num_players=2, hidden=256, blocks=3)
    net.copy_weights_to(net3)
    pol_c, _ = net3.predict(obs)
    diff3 = np.abs(pol_a - pol_c).max()
    assert diff3 < 0.01, f"copy_weights max diff = {diff3}"
    print(f"copy_weights:  max diff = {diff3:.8f}")

    # Benchmark
    t0 = time.time()
    for _ in range(200):
        net.predict(obs)
    dt = time.time() - t0
    print(f"\n200 single predicts: {dt:.3f}s ({dt/200*1000:.1f} ms each)")

    t0 = time.time()
    for _ in range(200):
        net.predict_batch(batch)
    dt = time.time() - t0
    print(f"200 batch(32) predicts: {dt:.3f}s ({dt/200*1000:.1f} ms each)")

    print("\nAll tests passed!")
