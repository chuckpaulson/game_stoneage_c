"""
stone_age_wrapper.py — ctypes wrapper for the C Stone Age engine.

State is a plain numpy int32 array (505 elements, ~2 KB).
Clone = np.copy(state). No heap allocation, no pointers to manage.

Implements the GameInterface contract from mcts_alphazero.py.
"""
import ctypes, os
import numpy as np
from numpy.ctypeslib import ndpointer

# ── Load library ──
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "libstoneage.so"))

_i32p = ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_f32p = ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")

_lib.sa_init.restype = None
_lib.sa_init.argtypes = [_i32p, ctypes.c_int, ctypes.c_uint64]

_lib.sa_clone.restype = None
_lib.sa_clone.argtypes = [_i32p, _i32p]

_lib.sa_get_obs.restype = None
_lib.sa_get_obs.argtypes = [_i32p, _i32p]

_lib.sa_legal_actions.restype = ctypes.c_int
_lib.sa_legal_actions.argtypes = [_i32p, _i32p, ctypes.c_int]

_lib.sa_apply_action.restype = None
_lib.sa_apply_action.argtypes = [_i32p, ctypes.c_int32]

_lib.sa_is_terminal.restype = ctypes.c_int
_lib.sa_is_terminal.argtypes = [_i32p]

_lib.sa_current_player.restype = ctypes.c_int
_lib.sa_current_player.argtypes = [_i32p]

_lib.sa_num_players.restype = ctypes.c_int
_lib.sa_num_players.argtypes = [_i32p]

_lib.sa_move_number.restype = ctypes.c_int
_lib.sa_move_number.argtypes = [_i32p]

_lib.sa_get_rewards.restype = None
_lib.sa_get_rewards.argtypes = [_i32p, _f32p]

_lib.sa_is_chance.restype = ctypes.c_int
_lib.sa_is_chance.argtypes = [_i32p]

_lib.sa_chance_outcomes.restype = ctypes.c_int
_lib.sa_chance_outcomes.argtypes = [_i32p]

_lib.sa_chance_probs.restype = None
_lib.sa_chance_probs.argtypes = [_i32p, _f32p]

_lib.sa_apply_chance.restype = None
_lib.sa_apply_chance.argtypes = [_i32p, ctypes.c_int]

# ── Constants ──
STATE_SIZE = 505
OBS_SIZE = 424
ACTION_SIZE = 409


class StoneAgeGame:
    """
    GameInterface implementation. State = numpy int32[505].

    Usage:
        game = StoneAgeGame(num_players=2, seed=42)
        while not game.is_terminal():
            if game.is_chance_node():
                outcomes, probs = game.chance_outcomes()
                i = np.random.choice(len(outcomes), p=probs)
                game.apply_chance_outcome(int(outcomes[i]))
            else:
                actions = game.legal_actions()
                game.apply_action(int(actions[np.random.randint(len(actions))]))
        rewards = game.get_rewards()
    """

    __slots__ = ("state", "_np")

    def __init__(self, num_players=2, seed=0, *, _state=None):
        if _state is not None:
            self.state = _state
        else:
            self.state = np.zeros(STATE_SIZE, dtype=np.int32)
            _lib.sa_init(self.state, num_players, seed)
        self._np = int(self.state[0])  # S_NPLAYERS is index 0

    # ── Clone ──
    def clone(self):
        g = StoneAgeGame.__new__(StoneAgeGame)
        g.state = self.state.copy()
        g._np = self._np
        return g

    # ── Observation ──
    def get_observation(self):
        obs = np.zeros(OBS_SIZE, dtype=np.int32)
        _lib.sa_get_obs(self.state, obs)
        return obs

    # ── Legal actions ──
    def legal_actions(self):
        buf = np.zeros(ACTION_SIZE, dtype=np.int32)
        n = _lib.sa_legal_actions(self.state, buf, ACTION_SIZE)
        return buf[:n].copy()

    def legal_actions_mask(self):
        m = np.zeros(ACTION_SIZE, dtype=np.bool_)
        m[self.legal_actions()] = True
        return m

    # ── Apply action ──
    def apply_action(self, action_id):
        _lib.sa_apply_action(self.state, action_id)

    # ── Chance nodes ──
    def is_chance_node(self):
        return bool(_lib.sa_is_chance(self.state))

    def chance_outcomes(self):
        nc = _lib.sa_chance_outcomes(self.state)
        buf = np.zeros(max(nc, 51), dtype=np.float32)
        _lib.sa_chance_probs(self.state, buf)
        return np.arange(nc, dtype=np.int32), buf[:nc].copy()

    def apply_chance_outcome(self, outcome):
        _lib.sa_apply_chance(self.state, outcome)

    # ── Terminal ──
    def is_terminal(self):
        return bool(_lib.sa_is_terminal(self.state))

    def get_rewards(self):
        r = np.zeros(self._np, dtype=np.float32)
        _lib.sa_get_rewards(self.state, r)
        return r

    # ── Info ──
    def current_player(self):
        return _lib.sa_current_player(self.state)

    def num_players(self):
        return self._np

    def move_number(self):
        return _lib.sa_move_number(self.state)


# ════════════════════════════════════════════
#  Self-test & benchmarks
# ════════════════════════════════════════════

if __name__ == "__main__":
    import time

    def play_random(np_count, seed):
        g = StoneAgeGame(num_players=np_count, seed=seed)
        moves = chances = 0
        while not g.is_terminal() and moves < 10000:
            if g.is_chance_node():
                outs, probs = g.chance_outcomes()
                i = np.random.choice(len(outs), p=probs)
                g.apply_chance_outcome(int(outs[i]))
                chances += 1
            else:
                acts = g.legal_actions()
                assert len(acts) > 0, f"No legal actions at move {moves}"
                g.apply_action(int(acts[np.random.randint(len(acts))]))
                moves += 1
        return moves, chances, g.is_terminal(), g

    print(f"State: {STATE_SIZE} int32 = {STATE_SIZE*4} bytes")
    print(f"Obs:   {OBS_SIZE} int32")
    print(f"Acts:  {ACTION_SIZE}\n")

    # Correctness
    for npc in [2, 3, 4]:
        done = total_m = total_c = 0
        t0 = time.time()
        for s in range(100):
            m, c, term, g = play_random(npc, 100 + s)
            if term:
                done += 1; total_m += m; total_c += c
        dt = time.time() - t0
        print(f"{npc}p: {done}/100 games, avg {total_m//max(done,1)} moves, "
              f"{total_c//max(done,1)} chances, {dt:.3f}s")
    assert done == 100, "Not all games completed!"

    # Clone isolation
    g = StoneAgeGame(num_players=2, seed=1)
    g2 = g.clone()
    if not g.is_chance_node():
        acts = g.legal_actions()
        g.apply_action(int(acts[0]))
    assert not np.array_equal(g.state, g2.state), "Clone not independent"
    print("Clone isolation: OK")

    # Observation sanity
    g = StoneAgeGame(num_players=2, seed=42)
    obs = g.get_observation()
    assert obs.shape == (424,) and obs.dtype == np.int32
    assert obs[0] == 2
    print(f"Obs check: OK (meta={obs[:4]})")

    # Benchmarks
    print("\n--- Benchmarks ---")
    g = StoneAgeGame(num_players=2, seed=42)
    N = 200_000

    t0 = time.time()
    for _ in range(N): g2 = g.clone()
    dt = time.time() - t0
    print(f"Clone:       {dt/N*1e6:.2f} us  ({N} ops)")

    obs_buf = np.zeros(OBS_SIZE, dtype=np.int32)
    t0 = time.time()
    for _ in range(N): _lib.sa_get_obs(g.state, obs_buf)
    dt = time.time() - t0
    print(f"get_obs:     {dt/N*1e6:.2f} us  ({N} ops)")

    act_buf = np.zeros(ACTION_SIZE, dtype=np.int32)
    t0 = time.time()
    for _ in range(N): _lib.sa_legal_actions(g.state, act_buf, ACTION_SIZE)
    dt = time.time() - t0
    print(f"legal_acts:  {dt/N*1e6:.2f} us  ({N} ops)")

    NG = 2000
    t0 = time.time()
    for s in range(NG): play_random(2, s)
    dt = time.time() - t0
    print(f"Full game:   {dt/NG*1000:.2f} ms  ({NG} games)")

    print("\nAll tests passed!")
