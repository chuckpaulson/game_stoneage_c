#!/usr/bin/env python3
"""
trace_game.py — Play a Stone Age game and produce a compact readable trace.

Outputs a text file showing every action and chance outcome with human-readable
names, grouped by round and phase. Key state changes (resource deltas, scores)
are shown inline so you can follow the game logic step by step.

Usage:
    python trace_game.py                         # 2p, seed=42, stdout
    python trace_game.py -o game.trace           # write to file
    python trace_game.py --players 3 --seed 99   # 3-player, seed 99
    python trace_game.py --max-moves 200         # short game
    python trace_game.py --model checkpoints/model_0099.pt   # trained NN (raw policy)
    python trace_game.py --model checkpoints/model_0099.pt --mcts 50  # NN + MCTS search
"""

import argparse
import sys
import numpy as np

from stone_age_wrapper import StoneAgeGame, STATE_SIZE, ACTION_SIZE, OBS_SIZE

# ════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════

S_NPLAYERS = 0; S_ROUND = 1; S_FIRST = 2; S_MOVES = 4
S_PHASE = 6; S_ACTIVE = 7; S_SUB = 8
S_RES_PLAYER = 9; S_RES_IDX = 10
S_DICE = 31; S_PEND_RES = 32; S_PEND_SLOT = 33; S_REM_PICKS = 34
S_FOOD_NEED = 43; S_IS_CHANCE = 48; S_CH_TYPE = 49; S_CH_NDICE = 50
S_CH_RES = 51

S_LOCFIG = 59; S_BLOCKED = 91; S_SUPPLY = 92
S_CARD_ID = 96; S_CARD_FIG = 100; S_BLD_ID = 104; S_BLD_FIG = 108
S_DECK_SZ = 432; S_STK_SZ = 469

PB_RES = 0; PB_FOOD = 4; PB_FIGS = 5; PB_AVAIL = 6; PB_AGRI = 7
PB_TOOL = 8; PB_TUSED = 11; PB_SCORE = 14; PB_NOT = 15; PB_OT = 16
PB_CIV = 24; PB_BLD = 58; PB_SIZE = 80
S_PLAYERS = 112

NUM_LOCS = 8; N_CARDS = 34; N_BLDG = 22

PH_PLACE = 0; PH_RESOLVE = 1; PH_FEED = 2; PH_OVER = 3
SP_CHOOSE = 0; SP_TOOL = 1; SP_BLD_PAY = 2; SP_CARD_PAY = 3
SP_ANY_RES = 4; SP_POTP = 5; SP_FREE = 6; SP_FEED = 7

PHASE_NAME = {0: "PLACE", 1: "RESOLVE", 2: "FEED", 3: "GAME OVER"}
SUB_NAME = {0: "choose", 1: "tool", 2: "bld_pay", 3: "card_pay",
            4: "any_res", 5: "potp", 6: "free", 7: "feed"}

RES_NAME = ["wood", "clay", "stone", "gold"]
LOC_NAME = ["Forest", "ClayPit", "Quarry", "River", "Hunt", "ToolMkr", "Hut", "Field"]
CH_NAME = {0: "none", 1: "resource", 2: "card_dice", 3: "potpourri"}
POTP_BONUS = ["wood", "clay", "stone", "gold", "tool", "agri"]

# ── Card names (34 unique types, from stone_age.c CARD table) ──
# Format: "Color/Type: effect"
# Green cards: green_type 0=Pottery 1=Weaving 2=Art 3=Time 4=Transport 5=Music 6=Medicine 7=Writing
# Sand cards:  sand_type  0=Farmers 1=ToolMakers 2=Builders 3=Shamans
CARD_NAME = [
    # Green cards (0-14)
    "Medicine/any×2",       #  0: green Medicine, EFF_ANY 2
    "Medicine/food+5",      #  1: green Medicine, EFF_FOOD 5
    "Writing/free",         #  2: green Writing, EFF_FREE
    "Writing/potp",         #  3: green Writing, EFF_POTP
    "Time/agri+1",          #  4: green Time, EFF_AGRI 1
    "Time/potp",            #  5: green Time, EFF_POTP
    "Pottery/food+7",       #  6: green Pottery, EFF_FOOD 7
    "Pottery/potp",         #  7: green Pottery, EFF_POTP
    "Transport/potp",       #  8: green Transport, EFF_POTP
    "Transport/stone×2",    #  9: green Transport, EFF_RES 2 stone
    "Art/tool+1",           # 10: green Art, EFF_TOOL 1
    "Art/dice→gold",        # 11: green Art, EFF_DICE 2 gold
    "Weaving/food+3",       # 12: green Weaving, EFF_FOOD 3
    "Weaving/food+1",       # 13: green Weaving, EFF_FOOD 1
    "Music/pts+3",          # 14: green Music, EFF_PTS 3 (×2 in deck)
    # Sand cards (15-33)
    "Shamans×2/dice→wood",  # 15: sand Shamans mult=2, EFF_DICE 2 wood
    "Shamans×2/clay+1",     # 16: sand Shamans mult=2, EFF_RES 1 clay
    "Shamans×1/gold+1",     # 17: sand Shamans mult=1, EFF_RES 1 gold
    "Shamans×1/dice→stone", # 18: sand Shamans mult=1, EFF_DICE 2 stone
    "Shamans×1/stone+1",    # 19: sand Shamans mult=1, EFF_RES 1 stone
    "Builders×3/pts+3",     # 20: sand Builders mult=3, EFF_PTS 3
    "Builders×2/potp",      # 21: sand Builders mult=2, EFF_POTP
    "Builders×2/food+2",    # 22: sand Builders mult=2, EFF_FOOD 2
    "Builders×1/potp",      # 23: sand Builders mult=1, EFF_POTP
    "Builders×1/food+4",    # 24: sand Builders mult=1, EFF_FOOD 4
    "Farmers×1/agri+1",     # 25: sand Farmers mult=1, EFF_AGRI 1
    "Farmers×2/potp",       # 26: sand Farmers mult=2, EFF_POTP
    "Farmers×2/food+3",     # 27: sand Farmers mult=2, EFF_FOOD 3
    "Farmers×1/potp",       # 28: sand Farmers mult=1, EFF_POTP
    "Farmers×1/stone+1",    # 29: sand Farmers mult=1, EFF_RES 1 stone
    "ToolMkrs×2/potp",      # 30: sand ToolMakers mult=2, EFF_POTP (×2 in deck)
    "ToolMkrs×2/tool+2",    # 31: sand ToolMakers mult=2, EFF_TOOL 2
    "ToolMkrs×1/ott(4)",    # 32: sand ToolMakers mult=1, EFF_OTT val=4
    "ToolMkrs×1/ott(3)",    # 33: sand ToolMakers mult=1, EFF_OTT val=3
]

# Card metadata for final scoring (mirrors C CARD table)
CARD_IS_GREEN = [1]*15 + [0]*19
CARD_GREEN_TYPE = [6,6,7,7,3,3,0,0,4,4,2,2,1,1,5] + [-1]*19
CARD_SAND_TYPE = [-1]*15 + [3,3,3,3,3, 2,2,2,2,2, 0,0,0,0,0, 1,1,1,1]
CARD_MULTIPLIER = [0]*15 + [2,2,1,1,1, 3,2,2,1,1, 1,2,2,1,1, 2,2,1,1]
GREEN_TYPE_NAME = ["Pottery","Weaving","Art","Time","Transport","Music","Medicine","Writing"]
SAND_TYPE_NAME = ["Farmers","ToolMakers","Builders","Shamans"]
SAND_STAT_NAME = ["agri","tools","buildings","figures"]

# ── Building names (22 unique types) ──
# Fixed: show cost→pts.  Variable: show type.
_BT_NAME = {0: "Fixed", 1: "1-7any", 2: "4/1type", 3: "4/2type",
             4: "4/3type", 5: "4/4type", 6: "5/1type", 7: "5/2type",
             8: "5/3type", 9: "5/4type"}
_BLD_TYPE = [0,0,0,0,0,0,0,0,0,0,0,0,0, 1,2,3,4,5,6,7,8,9]
_BLD_COST = [
    (2,1,0,0),(2,0,1,0),(1,2,0,0),(1,1,1,0),(2,0,0,1),
    (1,0,2,0),(0,2,1,0),(1,1,0,1),(1,0,1,1),(0,2,0,1),
    (0,1,2,0),(0,1,1,1),(0,0,2,1),
    None,None,None,None,None,None,None,None,None,
]
_BLD_PTS = [10,11,11,12,12,13,13,13,14,14,14,15,16,
            0,0,0,0,0,0,0,0,0]

def bld_name(bid):
    """Human-readable building name."""
    if bid < 0 or bid >= N_BLDG:
        return f"bld?{bid}"
    bt = _BLD_TYPE[bid]
    if bt == 0:
        w, c, s, g = _BLD_COST[bid]
        cost_parts = []
        if w: cost_parts.append(f"{w}w")
        if c: cost_parts.append(f"{c}c")
        if s: cost_parts.append(f"{s}s")
        if g: cost_parts.append(f"{g}g")
        return f"{'+'.join(cost_parts)}→{_BLD_PTS[bid]}pt"
    return _BT_NAME[bt]


# ════════════════════════════════════════════════════════════════
#  State readers
# ════════════════════════════════════════════════════════════════

def pf(state, pi, f):
    return int(state[S_PLAYERS + pi * PB_SIZE + f])


def player_summary(state, pi):
    """One-line summary: resources, food, score, tools, agri, cards, buildings."""
    res = [pf(state, pi, PB_RES + r) for r in range(4)]
    food = pf(state, pi, PB_FOOD)
    score = pf(state, pi, PB_SCORE)
    figs = pf(state, pi, PB_FIGS)
    avail = pf(state, pi, PB_AVAIL)
    agri = pf(state, pi, PB_AGRI)
    tools = [pf(state, pi, PB_TOOL + t) for t in range(3)]
    n_ot = pf(state, pi, PB_NOT)
    ot = [pf(state, pi, PB_OT + t) for t in range(n_ot)]
    n_cards = sum(pf(state, pi, PB_CIV + c) for c in range(N_CARDS))
    n_blds = sum(pf(state, pi, PB_BLD + b) for b in range(N_BLDG))
    parts = [
        f"res=w{res[0]}c{res[1]}s{res[2]}g{res[3]}",
        f"food={food}",
        f"score={score}",
        f"fig={avail}/{figs}",
        f"tools={tools}",
        f"agri={agri}",
        f"cards={n_cards}",
        f"bldgs={n_blds}",
    ]
    if ot:
        parts.append(f"ot={ot}")
    return "  ".join(parts)


def card_slots_str(state, show_figs=False):
    """Card slot names. show_figs=True adds (pN) during PLACE/RESOLVE."""
    parts = []
    for i in range(4):
        cid = int(state[S_CARD_ID + i])
        cfig = int(state[S_CARD_FIG + i])
        cost = 4 - i
        if cid >= 0:
            s = f"[{cost}] {CARD_NAME[cid]}"
            if show_figs and cfig >= 0:
                s += f" (p{cfig})"
            parts.append(s)
        else:
            parts.append(f"[{cost}] ---")
    return "  ".join(parts)


def bld_slots_str(state, np, show_figs=False):
    """Building slot names. show_figs=True adds (pN) during PLACE/RESOLVE."""
    parts = []
    for i in range(np):
        bid = int(state[S_BLD_ID + i])
        bfig = int(state[S_BLD_FIG + i])
        if bid >= 0:
            s = f"#{i}: {bld_name(bid)}"
            if show_figs and bfig >= 0:
                s += f" (p{bfig})"
            parts.append(s)
        else:
            parts.append(f"#{i}: ---")
    return "  ".join(parts)


def board_state(state, np, indent="  "):
    """Multi-line board state: players, supply, cards, buildings."""
    lines = []
    # Players
    for pi in range(np):
        lines.append(f"{indent}P{pi}: {player_summary(state, pi)}")
    # Supply
    supply = [int(state[S_SUPPLY + r]) for r in range(4)]
    lines.append(f"{indent}Supply: w={supply[0]} c={supply[1]} "
                 f"s={supply[2]} g={supply[3]}")
    # Civ cards — deck count on same line as slots
    deck_sz = int(state[S_DECK_SZ])
    lines.append(f"{indent}Civ cards ({deck_sz} in deck): "
                 f"{card_slots_str(state)}")
    # Buildings — pile counts on same line as slots
    piles = [int(state[S_STK_SZ + i]) for i in range(np)]
    pile_str = "+".join(str(p) for p in piles)
    lines.append(f"{indent}Buildings ({pile_str} in piles): "
                 f"{bld_slots_str(state, np)}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  Action name decoder
# ════════════════════════════════════════════════════════════════

# Payment combo table (mirrors init_pay in C)
_PAY_TABLE = []
def _init_pay():
    if _PAY_TABLE:
        return
    for t in range(1, 8):
        for w in range(t + 1):
            for c in range(t - w + 1):
                for s in range(t - w - c + 1):
                    g = t - w - c - s
                    _PAY_TABLE.append((w, c, s, g))
_init_pay()


def action_name(a, state=None):
    """Human-readable name for an action ID."""
    if 0 <= a < 28:
        loc = a // 7
        figs = (a % 7) + 1
        return f"place {figs} on {LOC_NAME[loc]}"
    if 28 <= a <= 37:
        figs = (a - 28) + 1
        return f"place {figs} on Hunt"
    if a == 38: return "place ToolMaker"
    if a == 39: return "place Field"
    if a == 40: return "place Hut"
    if 41 <= a <= 44:
        slot = a - 41
        cost = 4 - slot
        if state is not None:
            cid = int(state[S_CARD_ID + slot])
            if 0 <= cid < N_CARDS:
                return f"place Card [{cost}] {CARD_NAME[cid]}"
        return f"place Card [{cost}]"
    if 45 <= a <= 48:
        slot = a - 45
        if state is not None:
            bid = int(state[S_BLD_ID + slot])
            if 0 <= bid < N_BLDG:
                return f"place Bldg #{slot} ({bld_name(bid)})"
        return f"place Bldg #{slot}"
    if a == 49: return "pass (placement)"
    if 50 <= a <= 52: return f"use tool[{a - 50}]"
    if 53 <= a <= 60: return f"use one-time tool[{a - 53}]"
    if a == 61: return "done (tools)"
    if 62 <= a <= 390:
        k = a - 62
        if k < len(_PAY_TABLE):
            w, c, s, g = _PAY_TABLE[k]
            parts = []
            if w: parts.append(f"{w}w")
            if c: parts.append(f"{c}c")
            if s: parts.append(f"{s}s")
            if g: parts.append(f"{g}g")
            return f"pay {'+'.join(parts)}"
        return f"pay[{k}]"
    if 391 <= a <= 394: return f"take {RES_NAME[a - 391]}"
    if 395 <= a <= 398: return f"pick potp bonus #{a - 395}"
    if 399 <= a <= 402:
        slot = a - 399
        cost = 4 - slot
        if state is not None:
            cid = int(state[S_CARD_ID + slot])
            if 0 <= cid < N_CARDS:
                return f"take free card [{cost}] {CARD_NAME[cid]}"
        return f"take free card [{cost}]"
    if 403 <= a <= 406: return f"feed with {RES_NAME[a - 403]}"
    if a == 407: return "feed penalty (-10)"
    if a == 408: return "pass"
    return f"action[{a}]"


def chance_name(state, outcome):
    """Human-readable name for a chance outcome."""
    ch_type = int(state[S_CH_TYPE])
    ndice = int(state[S_CH_NDICE])
    if ch_type == 3:  # potpourri
        if 0 <= outcome < len(POTP_BONUS):
            return f"potp die → {POTP_BONUS[outcome]}"
        return f"potp die → {outcome}"
    dice_sum = ndice + outcome
    if ch_type == 1:  # resource gather
        return f"{ndice}d6 → sum {dice_sum}"
    if ch_type == 2:  # card dice
        return f"card 2d6 → sum {dice_sum}"
    return f"chance({outcome})"


# ════════════════════════════════════════════════════════════════
#  Delta tracker
# ════════════════════════════════════════════════════════════════

class DeltaTracker:
    """Snapshot player state before an action, then report what changed.

    Separates the acting player's direct changes from automatic side-effects
    (village resolutions, feeding) that the state machine processes in the
    same call.
    """

    def __init__(self, state, np, active):
        self.np = np
        self.active = active
        self.snap = {}
        for pi in range(np):
            self.snap[pi] = {
                'res': [pf(state, pi, PB_RES + r) for r in range(4)],
                'food': pf(state, pi, PB_FOOD),
                'score': pf(state, pi, PB_SCORE),
                'agri': pf(state, pi, PB_AGRI),
                'tools': [pf(state, pi, PB_TOOL + t) for t in range(3)],
                'figs': pf(state, pi, PB_FIGS),
            }

    def _player_changes(self, state, pi):
        """Return list of change strings for one player."""
        old = self.snap[pi]
        changes = []
        for r in range(4):
            new_v = pf(state, pi, PB_RES + r)
            d = new_v - old['res'][r]
            if d != 0:
                sign = "+" if d > 0 else ""
                changes.append(f"{sign}{d}{RES_NAME[r][0]}")
        food_d = pf(state, pi, PB_FOOD) - old['food']
        if food_d != 0:
            changes.append(f"{'+' if food_d > 0 else ''}{food_d}food")
        score_d = pf(state, pi, PB_SCORE) - old['score']
        if score_d != 0:
            changes.append(f"{'+' if score_d > 0 else ''}{score_d}pts")
        agri_d = pf(state, pi, PB_AGRI) - old['agri']
        if agri_d != 0:
            changes.append(f"{'+' if agri_d > 0 else ''}{agri_d}agri")
        new_tools = [pf(state, pi, PB_TOOL + t) for t in range(3)]
        if new_tools != old['tools']:
            changes.append(f"tools→{new_tools}")
        figs_d = pf(state, pi, PB_FIGS) - old['figs']
        if figs_d != 0:
            changes.append(f"{'+' if figs_d > 0 else ''}{figs_d}fig")
        return changes

    def diff(self, state, village_as_auto=False):
        """Return (action_delta_str, auto_lines_list).

        action_delta_str: changes for the acting player (inline on action line)
        auto_lines_list: separate lines for other players' automatic changes
            and village auto-resolutions

        If village_as_auto=True, village effects (agri, tools, figs) for the
        active player are also moved to auto lines (for PLACE→RESOLVE transitions).
        """
        auto_lines = []

        # --- Other players: all changes are auto ---
        for pi in range(self.np):
            if pi == self.active:
                continue
            changes = self._player_changes(state, pi)
            if not changes:
                continue
            old = self.snap[pi]
            agri_d = pf(state, pi, PB_AGRI) - old['agri']
            new_tools = [pf(state, pi, PB_TOOL + t) for t in range(3)]
            tool_changed = new_tools != old['tools']
            figs_d = pf(state, pi, PB_FIGS) - old['figs']
            food_d = pf(state, pi, PB_FOOD) - old['food']
            score_d = pf(state, pi, PB_SCORE) - old['score']

            parts = []
            if agri_d > 0:
                parts.append(f"Field → +{agri_d}agri")
            if tool_changed:
                parts.append(f"ToolMaker → tools={new_tools}")
            if figs_d > 0:
                parts.append(f"Hut → +{figs_d}fig")
            if food_d < 0 and not any(pf(state, pi, PB_RES + r) != old['res'][r]
                                       for r in range(4)):
                parts.append(f"fed {-food_d}food")
            if score_d < 0 and food_d == 0:
                parts.append(f"feed penalty {score_d}pts")

            if parts:
                auto_lines.append(f"      auto P{pi}: {', '.join(parts)}")
            else:
                auto_lines.append(f"      auto P{pi}: {', '.join(changes)}")

        # --- Active player: split village effects if requested ---
        old_a = self.snap[self.active]
        pi = self.active

        if village_as_auto:
            # Separate village effects from other changes
            village_parts = []
            agri_d = pf(state, pi, PB_AGRI) - old_a['agri']
            new_tools = [pf(state, pi, PB_TOOL + t) for t in range(3)]
            tool_changed = new_tools != old_a['tools']
            figs_d = pf(state, pi, PB_FIGS) - old_a['figs']

            if agri_d > 0:
                village_parts.append(f"Field → +{agri_d}agri")
            if tool_changed:
                village_parts.append(f"ToolMaker → tools={new_tools}")
            if figs_d > 0:
                village_parts.append(f"Hut → +{figs_d}fig")
            if village_parts:
                auto_lines.insert(0, f"      auto P{pi}: {', '.join(village_parts)}")

            # Inline delta: only non-village changes (resources, food, score)
            inline_changes = []
            for r in range(4):
                new_v = pf(state, pi, PB_RES + r)
                d = new_v - old_a['res'][r]
                if d != 0:
                    sign = "+" if d > 0 else ""
                    inline_changes.append(f"{sign}{d}{RES_NAME[r][0]}")
            food_d = pf(state, pi, PB_FOOD) - old_a['food']
            if food_d != 0:
                inline_changes.append(f"{'+' if food_d > 0 else ''}{food_d}food")
            score_d = pf(state, pi, PB_SCORE) - old_a['score']
            if score_d != 0:
                inline_changes.append(f"{'+' if score_d > 0 else ''}{score_d}pts")
            action_delta = f"p{pi}({','.join(inline_changes)})" if inline_changes else ""
        else:
            active_changes = self._player_changes(state, pi)
            action_delta = f"p{pi}({','.join(active_changes)})" if active_changes else ""

        return action_delta, auto_lines


# ════════════════════════════════════════════════════════════════
#  Trace generator
# ════════════════════════════════════════════════════════════════

def scoring_breakdown(state, np_):
    """Detailed final scoring breakdown for each player."""
    lines = []
    for pi in range(np_):
        lines.append(f"  P{pi} scoring breakdown:")
        game_score = pf(state, pi, PB_SCORE)
        # We'll compute bonus components and subtract to get in-game score

        # --- Green cards: count by type, then sets ---
        tc = [0] * 8
        green_cards = []
        for c in range(N_CARDS):
            cnt = pf(state, pi, PB_CIV + c)
            if cnt > 0 and CARD_IS_GREEN[c]:
                gt = CARD_GREEN_TYPE[c]
                tc[gt] += cnt
                green_cards.append((CARD_NAME[c], cnt))

        # Compute set scores
        green_total = 0
        set_num = 0
        tc_copy = list(tc)
        set_details = []
        while True:
            d = sum(1 for t in tc_copy if t > 0)
            if d == 0:
                break
            set_num += 1
            types_in_set = [GREEN_TYPE_NAME[t] for t in range(8) if tc_copy[t] > 0]
            set_details.append(f"set {set_num}: {d} types ({', '.join(types_in_set)}) = {d}×{d} = {d*d}")
            green_total += d * d
            for t in range(8):
                if tc_copy[t] > 0:
                    tc_copy[t] -= 1

        if green_cards:
            owned = ", ".join(f"{n}×{cnt}" if cnt > 1 else n
                              for n, cnt in green_cards)
            lines.append(f"    Green cards ({len(green_cards)} types): {owned}")
            for sd in set_details:
                lines.append(f"      {sd}")
            lines.append(f"      Green total: {green_total} pts")
        else:
            lines.append(f"    Green cards: none (0 pts)")

        # --- Sand cards: multiplier × stat ---
        sand_total = 0
        # Compute stats
        agri = pf(state, pi, PB_AGRI)
        tool_val = sum(pf(state, pi, PB_TOOL + t) for t in range(3))
        n_bldg = sum(pf(state, pi, PB_BLD + b) for b in range(N_BLDG))
        n_figs = pf(state, pi, PB_FIGS)
        stats = [agri, tool_val, n_bldg, n_figs]

        sand_contribs = {}  # sand_type → list of (card_name, cnt, mult, stat, pts)
        for c in range(N_CARDS):
            cnt = pf(state, pi, PB_CIV + c)
            if cnt <= 0 or CARD_IS_GREEN[c]:
                continue
            st = CARD_SAND_TYPE[c]
            m = CARD_MULTIPLIER[c]
            stat = stats[st]
            pts = cnt * m * stat
            sand_total += pts
            if st not in sand_contribs:
                sand_contribs[st] = []
            sand_contribs[st].append((CARD_NAME[c], cnt, m, pts))

        if sand_contribs:
            lines.append(f"    Sand cards:")
            for st in sorted(sand_contribs.keys()):
                stat = stats[st]
                sname = SAND_TYPE_NAME[st]
                stat_label = SAND_STAT_NAME[st]
                lines.append(f"      {sname} ({stat_label}={stat}):")
                for cname, cnt, m, pts in sand_contribs[st]:
                    prefix = f"{cnt}× " if cnt > 1 else ""
                    lines.append(f"        {prefix}{cname}: {cnt}×{m}×{stat} = {pts}")
            lines.append(f"      Sand total: {sand_total} pts")
        else:
            lines.append(f"    Sand cards: none (0 pts)")

        # --- Remaining resources ---
        res = [pf(state, pi, PB_RES + r) for r in range(4)]
        res_total = sum(res)
        res_parts = []
        for r in range(4):
            if res[r] > 0:
                res_parts.append(f"{res[r]}{RES_NAME[r][0]}")
        if res_parts:
            lines.append(f"    Resources: {'+'.join(res_parts)} = {res_total} pts")
        else:
            lines.append(f"    Resources: none (0 pts)")

        # --- In-game score (buildings + card VP effects) ---
        bonus_total = green_total + sand_total + res_total
        in_game = game_score - bonus_total
        lines.append(f"    ────")
        lines.append(f"    In-game (buildings/card pts): {in_game}")
        lines.append(f"    Green card sets:              {green_total}")
        lines.append(f"    Sand card multipliers:        {sand_total}")
        lines.append(f"    Remaining resources:          {res_total}")
        lines.append(f"    TOTAL:                        {game_score}")
        lines.append("")
    return "\n".join(lines)


def _emit_phase_header(w, state, phase, np_, prev_phase):
    """Print a phase header if it hasn't been printed yet. Returns new prev_phase."""
    if phase == prev_phase:
        return prev_phase
    w(f"  ── {PHASE_NAME[phase]} ──")
    if phase == PH_RESOLVE:
        loc_parts = []
        for loc in range(NUM_LOCS):
            figs = [int(state[S_LOCFIG + loc * 4 + p]) for p in range(np_)]
            if any(f > 0 for f in figs):
                who = ",".join(f"p{p}:{figs[p]}" for p in range(np_)
                               if figs[p] > 0)
                loc_parts.append(f"{LOC_NAME[loc]}[{who}]")
        w(f"     Figures: {' '.join(loc_parts)}")
    return phase


def trace_game(num_players, seed, out, max_moves=5000, pick_action=None):
    g = StoneAgeGame(num_players=num_players, seed=seed)
    rng = np.random.RandomState(seed)
    np_ = num_players

    if pick_action is None:
        pick_action = lambda game, acts: int(acts[rng.choice(len(acts))])

    def w(line=""):
        out.write(line + "\n")

    w(f"{'=' * 70}")
    w(f"  STONE AGE TRACE — {np_}p  seed={seed}")
    w(f"{'=' * 70}")
    w()

    # Initial state
    w("INITIAL STATE:")
    w(board_state(g.state, np_))
    w()

    prev_round = -1
    prev_phase = -1
    moves = 0
    chances = 0
    step = 0

    while not g.is_terminal() and moves < max_moves:
        s = g.state
        rnd = int(s[S_ROUND])
        phase = int(s[S_PHASE])
        sub = int(s[S_SUB])
        active = int(s[S_ACTIVE])

        # Round header
        if rnd != prev_round:
            w(f"{'─' * 70}")
            w(f"  ROUND {rnd}   first=P{int(s[S_FIRST])}")
            w(f"{'─' * 70}")
            prev_round = rnd
            prev_phase = -1

        # Phase header
        if phase != prev_phase:
            prev_phase = _emit_phase_header(
                w, s, phase, np_, prev_phase)

        step += 1
        dt = DeltaTracker(s, np_, active)

        if g.is_chance_node():
            outs, probs = g.chance_outcomes()
            i = int(rng.choice(len(outs), p=probs))
            outcome = int(outs[i])
            cname = chance_name(s, outcome)
            g.apply_chance_outcome(outcome)
            chances += 1
            delta, auto = dt.diff(g.state)
            line = f"    ⚄ {cname}"
            if delta:
                line += f"  → {delta}"
            w(line)
            # If phase changed within same round, print new header before auto lines
            new_ph = int(g.state[S_PHASE])
            new_rnd = int(g.state[S_ROUND])
            if new_ph != phase and new_rnd == rnd:
                prev_phase = _emit_phase_header(
                    w, g.state, new_ph, np_, prev_phase)
            for al in auto:
                w(al)
        else:
            acts = g.legal_actions()
            if len(acts) == 0:
                w(f"    !! NO LEGAL ACTIONS (bug)")
                break
            i = int(rng.choice(len(acts)))  # fallback index, not used if pick_action set
            action = pick_action(g, acts)
            aname = action_name(action, state=s)
            n_legal = len(acts)

            # Capture gathering yield before applying done(tools)
            gather_str = ""
            if sub == SP_TOOL and action == 61:  # A_TOOL_DN
                dice_total = int(s[S_DICE])
                ch_res = int(s[S_CH_RES])
                if ch_res < 0:  # hunting → food
                    qty = dice_total // 2
                    gather_str = f"  ({dice_total}÷2 = {qty} food)"
                else:
                    divisor = ch_res + 3  # wood=3, clay=4, stone=5, gold=6
                    qty = dice_total // divisor
                    rname = RES_NAME[ch_res]
                    gather_str = f"  ({dice_total}÷{divisor} = {qty} {rname})"

            # Context for pay actions: what card/building is being paid for
            pay_ctx = ""
            if 62 <= action <= 390 or action == 408:  # A_PAY range or A_PASS
                pend_slot = int(s[S_PEND_SLOT])
                if sub == SP_CARD_PAY:
                    cid = int(s[S_CARD_ID + pend_slot])
                    cost = 4 - pend_slot
                    if 0 <= cid < N_CARDS:
                        pay_ctx = f" for card [{cost}] {CARD_NAME[cid]}"
                elif sub == SP_BLD_PAY:
                    bid = int(s[S_BLD_ID + pend_slot])
                    if 0 <= bid < N_BLDG:
                        pay_ctx = f" for bldg #{pend_slot} ({bld_name(bid)})"

            g.apply_action(action)
            moves += 1
            # Detect PLACE→RESOLVE transition: village effects should be auto lines
            new_ph = int(g.state[S_PHASE])
            new_rnd = int(g.state[S_ROUND])
            place_to_resolve = (phase == PH_PLACE and new_ph != PH_PLACE)
            delta, auto = dt.diff(g.state, village_as_auto=place_to_resolve)

            line = f"    P{active}: {aname}{pay_ctx}"
            if n_legal > 1:
                line += f"  ({n_legal} legal)"
            line += gather_str
            if delta:
                line += f"  → {delta}"

            # Extra context for tool decisions
            if sub == SP_TOOL and action != 61:
                line += f"  dice→{int(g.state[S_DICE])}"

            w(line)
            # If phase changed within same round, print new header before auto lines
            if new_ph != phase and new_rnd == rnd:
                prev_phase = _emit_phase_header(
                    w, g.state, new_ph, np_, prev_phase)
            for al in auto:
                w(al)

        # End of round: show board state after round
        new_phase = int(g.state[S_PHASE])
        new_round = int(g.state[S_ROUND])
        if (phase == PH_FEED and new_phase != PH_FEED) or new_round != rnd:
            w(f"  ── end of round {rnd} ──")
            w(board_state(g.state, np_, indent="    "))
            w()

    # Final state
    w()
    w(f"{'=' * 70}")
    if g.is_terminal():
        w(f"  GAME OVER — {moves} moves, {chances} chance outcomes, {step} steps")
    else:
        w(f"  DID NOT FINISH — {moves}/{max_moves} moves")
    w(f"{'=' * 70}")
    w()
    w("FINAL STATE:")
    w(board_state(g.state, np_))
    if g.is_terminal():
        w()
        w("SCORING BREAKDOWN:")
        w(scoring_breakdown(g.state, np_))
        rewards = g.get_rewards()
        w("REWARDS: " + "  ".join(f"P{i}={rewards[i]:+.2f}" for i in range(np_)))
        winner = int(np.argmax(rewards))
        w(f"WINNER:  P{winner}")
    w()

    return moves, chances


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stone Age game trace generator")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--players", type=int, default=2,
                        help="Number of players (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max-moves", type=int, default=5000,
                        help="Max moves (default: 5000)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model checkpoint (e.g. checkpoints/model_0099.pt)")
    parser.add_argument("--mcts", type=int, default=0,
                        help="MCTS simulations per move (0 = use raw policy, default: 0)")
    args = parser.parse_args()

    pick_action = None
    if args.model:
        from stone_age_net import StoneAgeNN
        from mcts_alphazero import AlphaZeroConfig, MCTSSearch
        nn = StoneAgeNN(num_players=args.players, hidden=256, blocks=3)
        nn.load(args.model)
        print(f"Loaded model: {args.model}", file=sys.stderr)

        if args.mcts > 0:
            config = AlphaZeroConfig(
                num_players=args.players,
                action_space_size=ACTION_SIZE,
                observation_size=OBS_SIZE,
                num_simulations=args.mcts,
            )
            mcts = MCTSSearch(config, nn)
            def pick_action(game, acts):
                policy = mcts.search(game, add_noise=False)
                return int(np.argmax(policy))
        else:
            # Raw policy — fast, no search
            def pick_action(game, acts):
                obs = game.get_observation()
                policy, _ = nn.predict(obs)
                # Mask illegal actions, pick highest
                mask = np.zeros(ACTION_SIZE, dtype=bool)
                mask[acts] = True
                policy[~mask] = 0
                return int(np.argmax(policy))

    if args.output:
        with open(args.output, "w") as f:
            moves, chances = trace_game(args.players, args.seed, f, args.max_moves, pick_action)
        print(f"Wrote trace to {args.output} ({moves} moves, {chances} chances)")
    else:
        trace_game(args.players, args.seed, sys.stdout, args.max_moves, pick_action)


if __name__ == "__main__":
    main()
