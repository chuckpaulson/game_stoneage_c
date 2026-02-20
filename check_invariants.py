#!/usr/bin/env python3
"""
check_invariants.py — Invariant-checking harness for the Stone Age game engine.

Plays thousands of random games and asserts structural invariants after every
single move and chance outcome. Any violation is reported with full context
(seed, move number, phase, action) so you can reproduce it deterministically.

KNOWN BUGS FOUND BY THIS HARNESS:
──────────────────────────────────
1. ACTION SPACE COLLISION: Hunting allows unlimited figures, but the action
   encoding `loc*7 + (figs-1)` only allocates 7 slots per location (0..34).
   With 8+ available figures, hunting actions 4*7+7=35 and 4*7+8=36 collide
   with A_PL_TM=35 and A_PL_FLD=36. Fix: cap hunting figures at 7, or widen
   the action encoding.

2. STALE CARD_FIG IN SP_FREE: The FREE card-effect handler sets
   `card_id[slot] = -1` but does NOT clear `card_fig[slot] = -1`.
   Compare with SP_CARD_PAY which correctly clears both.

3. VILLAGE DOUBLE-OCCUPATION: A consequence of bug #1 — action 35 intended
   as "place 8 figures on hunting" is dispatched as "place on tool maker",
   allowing two players to occupy the same village location.

Usage:
    python check_invariants.py                  # default: 500 games × {2,3,4}p
    python check_invariants.py --games 2000     # more games
    python check_invariants.py --players 2      # only 2-player
    python check_invariants.py --seed 42        # start from specific seed
    python check_invariants.py --verbose        # print every move
"""

import argparse
import sys
import time
import traceback
import numpy as np

from stone_age_wrapper import StoneAgeGame, STATE_SIZE, OBS_SIZE, ACTION_SIZE

# ════════════════════════════════════════════════════════════════
#  Constants (mirrored from stone_age.h)
# ════════════════════════════════════════════════════════════════

MAX_PLAYERS = 4
NUM_LOCS = 8
N_CARDS = 34
N_BLDG = 22
DECK_TOTAL = 36
BLDG_TOTAL = 28
MAX_STACK = 7
MAX_OT = 8
SUPPLY_MAX = [28, 18, 12, 10]  # wood, clay, stone, gold

# State offsets
S_NPLAYERS = 0; S_ROUND = 1; S_FIRST = 2; S_GAMEOVER = 3; S_MOVES = 4
S_PHASE = 6; S_ACTIVE = 7; S_SUB = 8
S_RES_PLAYER = 9; S_RES_IDX = 10; S_RES_DONE = 11
S_LOC_RES = 15; S_DICE = 31
S_PEND_RES = 32; S_PEND_SLOT = 33; S_REM_PICKS = 34
S_POTP_BON = 35; S_PFED = 39; S_FOOD_NEED = 43; S_PL_PASS = 44
S_IS_CHANCE = 48; S_CH_TYPE = 49; S_CH_NDICE = 50; S_CH_RES = 51

S_LOCFIG = 59; S_BLOCKED = 91; S_SUPPLY = 92
S_CARD_ID = 96; S_CARD_FIG = 100
S_BLD_ID = 104; S_BLD_FIG = 108

PB_RES = 0; PB_FOOD = 4; PB_FIGS = 5; PB_AVAIL = 6; PB_AGRI = 7
PB_TOOL = 8; PB_TUSED = 11; PB_SCORE = 14
PB_NOT = 15; PB_OT = 16; PB_CIV = 24; PB_BLD = 58; PB_SIZE = 80
S_PLAYERS = 112

S_DECK_SZ = 432; S_DECK = 433
S_STK_SZ = 469; S_STK = 473

PH_PLACE = 0; PH_RESOLVE = 1; PH_FEED = 2; PH_OVER = 3
SP_CHOOSE = 0; SP_TOOL = 1; SP_BLD_PAY = 2; SP_CARD_PAY = 3
SP_ANY_RES = 4; SP_POTP = 5; SP_FREE = 6; SP_FEED = 7

PHASE_NAMES = {0: "PLACE", 1: "RESOLVE", 2: "FEED", 3: "OVER"}
SUB_NAMES = {0: "CHOOSE", 1: "TOOL", 2: "BLD_PAY", 3: "CARD_PAY",
             4: "ANY_RES", 5: "POTP", 6: "FREE", 7: "FEED"}


# ════════════════════════════════════════════════════════════════
#  Helper: read player block from raw state
# ════════════════════════════════════════════════════════════════

def pf(state, pi, field):
    """Read player field from flat state array."""
    return int(state[S_PLAYERS + pi * PB_SIZE + field])


# ════════════════════════════════════════════════════════════════
#  Invariant checks — each returns (ok, message)
# ════════════════════════════════════════════════════════════════

class InvariantChecker:
    """Checks all structural invariants on a Stone Age game state."""

    def __init__(self, state, context=""):
        self.s = state
        self.ctx = context
        self.np = int(state[S_NPLAYERS])
        self.errors = []

    def _fail(self, msg):
        self.errors.append(msg)

    def _check(self, cond, msg):
        if not cond:
            self._fail(msg)

    # ── Meta ──

    def check_meta(self):
        s = self.s
        self._check(self.np in (2, 3, 4),
                     f"num_players={self.np} not in {{2,3,4}}")
        self._check(int(s[S_ROUND]) >= 0,
                     f"round={s[S_ROUND]} negative")
        self._check(0 <= int(s[S_FIRST]) < self.np,
                     f"first_player={s[S_FIRST]} out of range [0,{self.np})")
        self._check(0 <= int(s[S_ACTIVE]) < self.np,
                     f"active_player={s[S_ACTIVE]} out of range [0,{self.np})")
        self._check(int(s[S_MOVES]) >= 0,
                     f"move_count={s[S_MOVES]} negative")

    # ── Phase ──

    def check_phase(self):
        s = self.s
        phase = int(s[S_PHASE])
        self._check(phase in (PH_PLACE, PH_RESOLVE, PH_FEED, PH_OVER),
                     f"phase={phase} invalid")
        sub = int(s[S_SUB])
        self._check(0 <= sub <= 7, f"sub_phase={sub} out of range")

    # ── Player resources (non-negative, bounded) ──

    def check_player_resources(self):
        for pi in range(self.np):
            for r in range(4):
                v = pf(self.s, pi, PB_RES + r)
                res_name = ["wood", "clay", "stone", "gold"][r]
                self._check(v >= 0,
                             f"p{pi} {res_name}={v} negative")
                self._check(v <= SUPPLY_MAX[r],
                             f"p{pi} {res_name}={v} exceeds supply max {SUPPLY_MAX[r]}")
            food = pf(self.s, pi, PB_FOOD)
            self._check(food >= 0, f"p{pi} food={food} negative")

    # ── Supply bounds and conservation ──

    def check_supply(self):
        s = self.s
        for r in range(4):
            supply = int(s[S_SUPPLY + r])
            res_name = ["wood", "clay", "stone", "gold"][r]
            self._check(supply >= 0,
                         f"supply[{res_name}]={supply} negative")
            self._check(supply <= SUPPLY_MAX[r],
                         f"supply[{res_name}]={supply} exceeds max {SUPPLY_MAX[r]}")

            # Conservation: supply + all player holdings <= SUPPLY_MAX
            # (supply can grow beyond initial if resources are returned, but
            #  total in-system should equal SUPPLY_MAX)
            total_held = sum(pf(self.s, pi, PB_RES + r) for pi in range(self.np))
            total = supply + total_held
            self._check(total <= SUPPLY_MAX[r],
                         f"supply conservation: supply[{res_name}]={supply} + "
                         f"held={total_held} = {total} > max {SUPPLY_MAX[r]}")

    # ── Figure conservation ──

    def check_figures(self):
        s = self.s
        for pi in range(self.np):
            total_figs = pf(self.s, pi, PB_FIGS)
            avail = pf(self.s, pi, PB_AVAIL)

            # Count figures on all board locations
            on_board = 0
            for loc in range(NUM_LOCS):
                figs_here = int(s[S_LOCFIG + loc * 4 + pi])
                self._check(figs_here >= 0,
                             f"p{pi} loc{loc} figures={figs_here} negative")
                on_board += figs_here

            # Count figures on card/building slots (1 each if claimed)
            on_cards = sum(1 for i in range(4) if int(s[S_CARD_FIG + i]) == pi)
            on_blds = sum(1 for i in range(4) if int(s[S_BLD_FIG + i]) == pi)

            self._check(total_figs >= 5,
                         f"p{pi} total_figs={total_figs} < 5 (starting)")
            self._check(total_figs <= 10,
                         f"p{pi} total_figs={total_figs} > 10 (max)")
            self._check(avail >= 0,
                         f"p{pi} avail={avail} negative")
            self._check(avail <= total_figs,
                         f"p{pi} avail={avail} > total_figs={total_figs}")

            # During PLACE phase: on_board + on_cards + on_blds + avail = total_figs
            # During RESOLVE/FEED: figures are still on locations
            placed = on_board + on_cards + on_blds
            accounted = placed + avail
            self._check(accounted <= total_figs,
                         f"p{pi} figure overcount: placed={placed} + avail={avail} "
                         f"= {accounted} > total_figs={total_figs}")

    # ── Tools ──

    def check_tools(self):
        for pi in range(self.np):
            prev = 99
            for t in range(3):
                val = pf(self.s, pi, PB_TOOL + t)
                used = pf(self.s, pi, PB_TUSED + t)
                self._check(0 <= val <= 4,
                             f"p{pi} tool[{t}]={val} out of [0,4]")
                self._check(used in (0, 1),
                             f"p{pi} tool_used[{t}]={used} not boolean")
            # Tools should be non-increasing (filled left to right)
            t0 = pf(self.s, pi, PB_TOOL)
            t1 = pf(self.s, pi, PB_TOOL + 1)
            t2 = pf(self.s, pi, PB_TOOL + 2)
            self._check(t0 >= t1 >= t2,
                         f"p{pi} tools not non-increasing: [{t0},{t1},{t2}]")

    # ── Agriculture ──

    def check_agriculture(self):
        for pi in range(self.np):
            agri = pf(self.s, pi, PB_AGRI)
            self._check(0 <= agri <= 10,
                         f"p{pi} agriculture={agri} out of [0,10]")

    # ── One-time tools ──

    def check_one_time_tools(self):
        for pi in range(self.np):
            n_ot = pf(self.s, pi, PB_NOT)
            self._check(0 <= n_ot <= MAX_OT,
                         f"p{pi} num_one_time_tools={n_ot} out of [0,{MAX_OT}]")
            for i in range(n_ot):
                v = pf(self.s, pi, PB_OT + i)
                self._check(v in (3, 4),
                             f"p{pi} one_time_tool[{i}]={v} not in {{3,4}}")

    # ── Deck ──

    def check_deck(self):
        s = self.s
        dsz = int(s[S_DECK_SZ])
        self._check(0 <= dsz <= DECK_TOTAL,
                     f"deck_size={dsz} out of [0,{DECK_TOTAL}]")
        for i in range(dsz):
            cid = int(s[S_DECK + i])
            self._check(0 <= cid < N_CARDS,
                         f"deck[{i}]={cid} invalid card ID")

    # ── Card slots ──

    def check_card_slots(self):
        s = self.s
        for i in range(4):
            cid = int(s[S_CARD_ID + i])
            cfig = int(s[S_CARD_FIG + i])
            if cid != -1:
                self._check(0 <= cid < N_CARDS,
                             f"card_slot[{i}] id={cid} invalid")
            self._check(cfig == -1 or (0 <= cfig < self.np),
                         f"card_slot[{i}] fig={cfig} invalid (np={self.np})")
            # If no card, can't have a figure
            if cid == -1:
                self._check(cfig == -1,
                             f"card_slot[{i}] empty (id=-1) but fig={cfig} "
                             f"(phase={PHASE_NAMES.get(int(s[S_PHASE]))}, "
                             f"sub={SUB_NAMES.get(int(s[S_SUB]))})")

    # ── Building slots and stacks ──

    def check_building_slots(self):
        s = self.s
        for i in range(self.np):
            bid = int(s[S_BLD_ID + i])
            bfig = int(s[S_BLD_FIG + i])
            if bid != -1:
                self._check(0 <= bid < N_BLDG,
                             f"bld_slot[{i}] id={bid} invalid")
            self._check(bfig == -1 or (0 <= bfig < self.np),
                         f"bld_slot[{i}] fig={bfig} invalid (np={self.np})")
            if bid == -1:
                self._check(bfig == -1,
                             f"bld_slot[{i}] empty but fig={bfig}")

    def check_stacks(self):
        s = self.s
        for si in range(self.np):
            sz = int(s[S_STK_SZ + si])
            self._check(0 <= sz <= MAX_STACK,
                         f"stack[{si}] size={sz} out of [0,{MAX_STACK}]")
            for j in range(sz):
                bid = int(s[S_STK + si * MAX_STACK + j])
                self._check(0 <= bid < N_BLDG,
                             f"stack[{si}][{j}]={bid} invalid building ID")

    # ── Location-specific rules ──

    def check_locations(self):
        s = self.s
        np = self.np
        # Resource locations (0-3): max 7 total figures
        for loc in range(4):
            total = sum(int(s[S_LOCFIG + loc * 4 + p]) for p in range(np))
            self._check(total <= 7,
                         f"loc {loc} has {total} figures > 7")

        # Village locations (5=TM, 6=Hut, 7=Field): at most 1 player occupies
        loc_names = {5: "TM", 6: "Hut", 7: "Field"}
        for loc in [5, 6, 7]:
            figs = [int(s[S_LOCFIG + loc * 4 + p]) for p in range(np)]
            occupied = sum(1 for f in figs if f > 0)
            self._check(occupied <= 1,
                         f"village loc {loc}({loc_names[loc]}) has {occupied} "
                         f"players occupying (figs={figs[:np]}, blocked={int(s[S_BLOCKED])})")

        # TM and field: exactly 1 figure when occupied
        for loc in [5, 7]:
            for p in range(np):
                f = int(s[S_LOCFIG + loc * 4 + p])
                self._check(f <= 1,
                             f"loc {loc} p{p} has {f} figures (TM/field max=1)")

        # Hut: exactly 2 figures when occupied
        for p in range(np):
            f = int(s[S_LOCFIG + 6 * 4 + p])
            self._check(f in (0, 2),
                         f"hut p{p} has {f} figures (must be 0 or 2)")

    # ── Observation sanity ──

    def check_observation(self, game):
        obs = game.get_observation()
        self._check(obs.shape == (OBS_SIZE,),
                     f"obs shape={obs.shape} expected ({OBS_SIZE},)")
        self._check(obs.dtype == np.int32,
                     f"obs dtype={obs.dtype} expected int32")
        # First element is num_players
        self._check(int(obs[0]) == self.np,
                     f"obs[0]={obs[0]} != num_players={self.np}")

    # ── Legal actions sanity ──

    def check_legal_actions(self, game):
        if game.is_terminal():
            return
        if game.is_chance_node():
            return

        acts = game.legal_actions()
        self._check(len(acts) > 0,
                     "no legal actions in non-terminal non-chance state")
        for a in acts:
            self._check(0 <= int(a) < ACTION_SIZE,
                         f"legal action {a} out of range [0,{ACTION_SIZE})")
        # No duplicates
        unique = set(int(a) for a in acts)
        if len(unique) != len(acts):
            from collections import Counter
            counts = Counter(int(a) for a in acts)
            dupes = {a: c for a, c in counts.items() if c > 1}
            dupe_info = ", ".join(f"act={a}×{c}" for a, c in sorted(dupes.items())[:5])
            self._fail(
                f"duplicate legal actions: {len(acts)} total, {len(unique)} unique "
                f"(dupes: {dupe_info})")

    # ── Chance node sanity ──

    def check_chance(self, game):
        if not game.is_chance_node():
            return
        outcomes, probs = game.chance_outcomes()
        self._check(len(outcomes) > 0, "chance node with 0 outcomes")
        self._check(len(outcomes) == len(probs),
                     f"outcomes/probs length mismatch: {len(outcomes)} vs {len(probs)}")
        self._check(all(p >= 0 for p in probs), "negative probability")
        prob_sum = float(np.sum(probs))
        self._check(abs(prob_sum - 1.0) < 0.01,
                     f"probabilities sum to {prob_sum}, expected ~1.0")

    # ── Terminal state ──

    def check_terminal(self, game):
        if not game.is_terminal():
            return
        rewards = game.get_rewards()
        self._check(len(rewards) == self.np,
                     f"rewards length={len(rewards)} != num_players={self.np}")
        for i, r in enumerate(rewards):
            self._check(-1.0 <= float(r) <= 1.0,
                         f"reward[{i}]={r} out of [-1,1]")

    # ── Blocked village ──

    def check_blocked(self):
        s = self.s
        blocked = int(s[S_BLOCKED])
        self._check(blocked in (0, 1, 2, 3),
                     f"blocked_village={blocked} not in {{0,1,2,3}}")

    # ── Civ and building counts ──

    def check_card_bld_counts(self):
        for pi in range(self.np):
            for c in range(N_CARDS):
                v = pf(self.s, pi, PB_CIV + c)
                self._check(v >= 0, f"p{pi} civ_count[{c}]={v} negative")
            for b in range(N_BLDG):
                v = pf(self.s, pi, PB_BLD + b)
                self._check(v >= 0, f"p{pi} bld_count[{b}]={v} negative")

    # ── Run all checks ──

    def run_all(self, game):
        self.check_meta()
        self.check_phase()
        self.check_player_resources()
        self.check_supply()
        self.check_figures()
        self.check_tools()
        self.check_agriculture()
        self.check_one_time_tools()
        self.check_deck()
        self.check_card_slots()
        self.check_building_slots()
        self.check_stacks()
        self.check_locations()
        self.check_blocked()
        self.check_card_bld_counts()
        self.check_observation(game)
        self.check_legal_actions(game)
        self.check_chance(game)
        self.check_terminal(game)
        return self.errors


# ════════════════════════════════════════════════════════════════
#  Score tracker (checks score never decreases except during feed)
# ════════════════════════════════════════════════════════════════

class ScoreTracker:
    """Track per-player scores and detect unexpected decreases.

    Note: feeding penalty (-10) is applied during a FEED-phase action, but
    the state machine can advance to PLACE in the same sa_apply_action call.
    So we allow decreases when the *previous* phase was FEED or current is FEED.
    """

    def __init__(self, np):
        self.np = np
        self.prev_scores = [0] * np
        self.prev_phase = PH_PLACE
        self.was_feed = False

    def update(self, state, phase):
        errors = []
        # A score decrease is only suspicious if neither current nor previous
        # phase was FEED (because the -10 penalty transitions immediately)
        feed_window = (phase == PH_FEED or self.was_feed)
        for pi in range(self.np):
            score = pf(state, pi, PB_SCORE)
            if score < self.prev_scores[pi] and not feed_window:
                errors.append(
                    f"p{pi} score decreased {self.prev_scores[pi]}→{score} "
                    f"outside FEED phase (prev_phase={PHASE_NAMES.get(self.prev_phase)}, "
                    f"cur_phase={PHASE_NAMES.get(phase)})")
            self.prev_scores[pi] = score
        self.was_feed = (phase == PH_FEED or self.prev_phase == PH_FEED)
        self.prev_phase = phase
        return errors


# ════════════════════════════════════════════════════════════════
#  Deterministic replay check
# ════════════════════════════════════════════════════════════════

def deterministic_replay_check(num_players, seed, max_moves=200):
    """Play the same game twice with the same random choices; states must match."""
    rng = np.random.RandomState(seed + 99999)

    def play_game(game_seed):
        g = StoneAgeGame(num_players=num_players, seed=game_seed)
        history = []
        moves = 0
        while not g.is_terminal() and moves < max_moves:
            if g.is_chance_node():
                outs, probs = g.chance_outcomes()
                # Use a deterministic choice based on move number
                i = int(rng.choice(len(outs), p=probs))
                g.apply_chance_outcome(int(outs[i]))
                history.append(("chance", int(outs[i])))
            else:
                acts = g.legal_actions()
                i = int(rng.choice(len(acts)))
                g.apply_action(int(acts[i]))
                history.append(("action", int(acts[i])))
                moves += 1
        return g.state.copy(), history

    # Play twice — must be identical
    rng1 = np.random.RandomState(seed + 99999)
    rng2 = np.random.RandomState(seed + 99999)

    # First playthrough
    rng = rng1
    state1, hist1 = play_game(seed)

    # Second playthrough
    rng = rng2
    state2, hist2 = play_game(seed)

    if not np.array_equal(state1, state2):
        diffs = np.where(state1 != state2)[0]
        return False, f"Non-deterministic! States differ at indices: {diffs[:20]}"

    return True, "OK"


# ════════════════════════════════════════════════════════════════
#  Clone isolation check
# ════════════════════════════════════════════════════════════════

def clone_isolation_check(game):
    """Verify that cloning produces an independent copy."""
    g2 = game.clone()
    original_state = game.state.copy()

    if not game.is_terminal():
        if game.is_chance_node():
            outs, probs = game.chance_outcomes()
            game.apply_chance_outcome(int(outs[0]))
        else:
            acts = game.legal_actions()
            game.apply_action(int(acts[0]))

    if np.array_equal(game.state, g2.state) and not np.array_equal(game.state, original_state):
        return False, "Clone was mutated by original"
    if not np.array_equal(g2.state, original_state):
        return False, "Clone state changed without any operation on it"
    return True, "OK"


# ════════════════════════════════════════════════════════════════
#  Apply-all-legal-actions crash test
# ════════════════════════════════════════════════════════════════

def legal_action_crash_test(game, max_test=50):
    """Clone the game and try applying every legal action — none should crash."""
    if game.is_terminal() or game.is_chance_node():
        return []
    errors = []
    acts = game.legal_actions()
    for a in acts[:max_test]:
        try:
            g2 = game.clone()
            g2.apply_action(int(a))
        except Exception as e:
            errors.append(f"Crash applying legal action {a}: {e}")
    return errors


# ════════════════════════════════════════════════════════════════
#  Main game loop with invariant checking
# ════════════════════════════════════════════════════════════════

def play_checked_game(num_players, seed, verbose=False, max_moves=5000):
    """
    Play one random game, checking all invariants after every step.
    Returns (completed, move_count, errors_list).
    """
    g = StoneAgeGame(num_players=num_players, seed=seed)
    rng = np.random.RandomState(seed)
    errors = []
    score_tracker = ScoreTracker(num_players)
    moves = 0
    chances = 0
    step = 0

    def ctx():
        phase = int(g.state[S_PHASE])
        sub = int(g.state[S_SUB])
        return (f"[seed={seed} np={num_players} step={step} move={moves} "
                f"chance={chances} phase={PHASE_NAMES.get(phase, phase)} "
                f"sub={SUB_NAMES.get(sub, sub)} active=p{int(g.state[S_ACTIVE])}]")

    # Check initial state
    checker = InvariantChecker(g.state, ctx())
    errs = checker.run_all(g)
    if errs:
        for e in errs:
            errors.append(f"INIT {ctx()}: {e}")

    while not g.is_terminal() and moves < max_moves:
        step += 1
        phase_before = int(g.state[S_PHASE])

        if g.is_chance_node():
            outs, probs = g.chance_outcomes()
            i = int(rng.choice(len(outs), p=probs))
            outcome = int(outs[i])
            if verbose:
                print(f"  {ctx()} CHANCE outcome={outcome}")
            g.apply_chance_outcome(outcome)
            chances += 1
        else:
            acts = g.legal_actions()
            if len(acts) == 0:
                errors.append(f"{ctx()}: no legal actions (should not happen)")
                break
            i = int(rng.choice(len(acts)))
            action = int(acts[i])
            if verbose:
                print(f"  {ctx()} ACTION={action} (of {len(acts)} legal)")

            # Crash test: try all legal actions (occasionally)
            if step % 50 == 0:
                crash_errs = legal_action_crash_test(g)
                for e in crash_errs:
                    errors.append(f"{ctx()}: {e}")

            g.apply_action(action)
            moves += 1

        # Check invariants after every step
        checker = InvariantChecker(g.state, ctx())
        errs = checker.run_all(g)
        for e in errs:
            errors.append(f"{ctx()}: {e}")

        # Score tracker
        score_errs = score_tracker.update(g.state, int(g.state[S_PHASE]))
        for e in score_errs:
            errors.append(f"{ctx()}: {e}")

        # Clone isolation (occasionally)
        if step % 100 == 0:
            g_backup = g.clone()
            ok, msg = clone_isolation_check(g_backup)
            if not ok:
                errors.append(f"{ctx()}: clone isolation: {msg}")

        # Bail early on excessive errors
        if len(errors) > 50:
            errors.append("(too many errors, stopping early)")
            break

    if moves >= max_moves and not g.is_terminal():
        errors.append(f"Game did not terminate in {max_moves} moves")

    return g.is_terminal(), moves, chances, errors


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stone Age invariant checker")
    parser.add_argument("--games", type=int, default=500,
                        help="Number of games per player count (default: 500)")
    parser.add_argument("--players", type=int, default=0,
                        help="Only test this player count (0 = all)")
    parser.add_argument("--seed", type=int, default=1000,
                        help="Starting seed (default: 1000)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every move")
    parser.add_argument("--max-moves", type=int, default=5000,
                        help="Max moves before declaring stuck (default: 5000)")
    args = parser.parse_args()

    player_counts = [args.players] if args.players else [2, 3, 4]
    all_errors = []
    total_games = 0
    total_completed = 0
    total_moves = 0
    total_chances = 0

    print("=" * 70)
    print("  Stone Age Invariant Checker")
    print("=" * 70)
    print(f"  Games per player count: {args.games}")
    print(f"  Player counts: {player_counts}")
    print(f"  Starting seed: {args.seed}")
    print(f"  Max moves: {args.max_moves}")
    print("=" * 70)
    print()

    for np_count in player_counts:
        t0 = time.time()
        completed = 0
        np_errors = []
        np_moves = 0
        np_chances = 0

        for i in range(args.games):
            seed = args.seed + i
            try:
                done, moves, chances, errs = play_checked_game(
                    np_count, seed, args.verbose, args.max_moves)
            except Exception as e:
                errs = [f"EXCEPTION: {e}\n{traceback.format_exc()}"]
                done = False
                moves = 0
                chances = 0

            if errs:
                for e in errs:
                    np_errors.append(f"[{np_count}p seed={seed}] {e}")
                # Print first few errors immediately
                if len(np_errors) <= 5:
                    for e in errs[:3]:
                        print(f"  ✗ [{np_count}p seed={seed}] {e}")

            if done:
                completed += 1
                np_moves += moves
                np_chances += chances

            total_games += 1

        dt = time.time() - t0
        avg_moves = np_moves // max(completed, 1)
        avg_chances = np_chances // max(completed, 1)

        status = "✓ PASS" if not np_errors else f"✗ FAIL ({len(np_errors)} errors)"
        print(f"{np_count}-player: {completed}/{args.games} completed, "
              f"avg {avg_moves} moves + {avg_chances} chances, "
              f"{dt:.1f}s — {status}")

        total_completed += completed
        total_moves += np_moves
        total_chances += np_chances
        all_errors.extend(np_errors)

    # Deterministic replay
    print(f"\nDeterministic replay check...")
    for np_count in player_counts:
        for seed in [args.seed, args.seed + 1, args.seed + 7]:
            ok, msg = deterministic_replay_check(np_count, seed)
            if not ok:
                err = f"Replay [{np_count}p seed={seed}]: {msg}"
                all_errors.append(err)
                print(f"  ✗ {err}")
    if not any("Replay" in e for e in all_errors):
        print("  ✓ All replay checks passed")

    # Summary
    print()
    print("=" * 70)
    if all_errors:
        print(f"  FAILED — {len(all_errors)} invariant violation(s) found")
        print("=" * 70)
        # Print unique errors (deduplicated by message suffix)
        seen = set()
        for e in all_errors:
            # Extract the invariant message (after the last ]: )
            key = e.split("]")[-1].strip() if "]" in e else e
            if key not in seen:
                seen.add(key)
                print(f"  {e}")
            if len(seen) >= 30:
                remaining = len(all_errors) - len(seen)
                if remaining > 0:
                    print(f"  ... and {remaining} more (possibly duplicates)")
                break
        return 1
    else:
        print(f"  ALL PASSED — {total_games} games, "
              f"{total_completed} completed, 0 invariant violations")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
