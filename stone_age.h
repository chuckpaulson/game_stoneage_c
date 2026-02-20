/*
 * stone_age.h — Stone Age board game engine for AlphaZero MCTS
 *
 * The game state is a flat int32_t array (STATE_SIZE elements).
 * This makes cloning trivial (memcpy), and numpy interop zero-cost.
 *
 * Observation output is a separate 424-element int32_t array,
 * player-rotated, matching the Python stone_age_state.py format.
 */
#ifndef STONE_AGE_H
#define STONE_AGE_H

#include <stdint.h>

/* ════════════════════════════════════════════
   Game constants
   ════════════════════════════════════════════ */
#define MAX_PLAYERS   4
#define NUM_LOCS      8   /* board locations */
#define NUM_RESOLVE  16   /* resolve slots per player */
#define N_CARDS      34   /* unique card types */
#define N_BLDG       22   /* unique building types */
#define DECK_TOTAL   36   /* cards in full deck */
#define BLDG_TOTAL   28   /* buildings in full set */
#define MAX_STACK     7
#define MAX_OT        8   /* one-time tool slots */
#define OBS_SIZE    424
#define ACT_SIZE    409
#define N_PAY       329   /* payment combo count */

/* Phases */
#define PH_PLACE  0
#define PH_RESOLVE 1
#define PH_FEED   2
#define PH_OVER   3

/* Sub-phases */
#define SP_CHOOSE   0
#define SP_TOOL     1
#define SP_BLD_PAY  2
#define SP_CARD_PAY 3
#define SP_ANY_RES  4
#define SP_POTP     5
#define SP_FREE     6
#define SP_FEED     7

/* Board locations */
#define LOC_FOREST   0
#define LOC_CLAY     1
#define LOC_QUARRY   2
#define LOC_RIVER    3
#define LOC_HUNT     4
#define LOC_TM       5  /* tool maker */
#define LOC_HUT      6
#define LOC_FIELD    7

/* Resources */
#define RES_W 0  /* wood */
#define RES_C 1  /* clay */
#define RES_S 2  /* stone */
#define RES_G 3  /* gold */

/* Card effects */
#define EFF_FOOD     0
#define EFF_RES      1
#define EFF_ANY      2
#define EFF_TOOL     3
#define EFF_AGRI     4
#define EFF_PTS      5
#define EFF_DICE     6
#define EFF_POTP     7
#define EFF_FREE     8
#define EFF_OTT      9  /* one-time tool */

/* Building types */
#define BT_FIXED  0
#define BT_17     1  /* 1-7 any */
#define BT_41     2
#define BT_42     3
#define BT_43     4
#define BT_44     5
#define BT_51     6
#define BT_52     7
#define BT_53     8
#define BT_54     9

/* Chance types */
#define CH_NONE   0
#define CH_RES    1  /* resource gathering dice */
#define CH_CARD   2  /* card dice-resource effect */
#define CH_POTP   3  /* potpourri single die */

/* ── Action IDs (must match Python action_space.py) ── */
#define A_PL_RES    0   /*  0..27: 4 locs × 7 fig counts (forest/clay/quarry/river) */
#define A_PL_HUNT  28   /* 28..37: hunt 1-10 figures */
#define A_PL_TM    38
#define A_PL_FLD   39
#define A_PL_HUT   40
#define A_PL_CARD  41   /* 41..44 */
#define A_PL_BLD   45   /* 45..48 */
#define A_PL_PASS  49
#define A_TOOL_P   50   /* 50..52: perm tools */
#define A_TOOL_OT  53   /* 53..60: one-time tools */
#define A_TOOL_DN  61
#define A_PAY      62   /* 62..390: 329 payment combos */
#define A_ANY      391  /* 391..394 */
#define A_POTP     395  /* 395..398 */
#define A_FREE     399  /* 399..402 */
#define A_FEED     403  /* 403..406: pay resource */
#define A_FEED_PEN 407
#define A_PASS     408

/* ════════════════════════════════════════════
   State layout — flat int32_t array
   ════════════════════════════════════════════

   All player indices are ABSOLUTE (0..num_players-1).
   get_obs() produces the player-rotated 424-element view.

   Offsets are #defined below. Total: STATE_SIZE.
*/

/* ── Meta ── */
#define S_NPLAYERS     0
#define S_ROUND        1
#define S_FIRST        2   /* first player index */
#define S_GAMEOVER     3
#define S_MOVES        4
/* 5 reserved */
#define S_META_END     6

/* ── Phase ── */
#define S_PHASE        6
#define S_ACTIVE       7
#define S_SUB          8
#define S_RES_PLAYER   9   /* resolution player */
#define S_RES_IDX     10   /* current resolve slot 0-15 */
#define S_RES_DONE    11   /* ..14: resolution_completed[4] */
#define S_LOC_RES     15   /* ..30: locations_resolved[16] */
#define S_DICE        31
#define S_PEND_RES    32   /* pending resource type */
#define S_PEND_SLOT   33
#define S_REM_PICKS   34
#define S_POTP_BON    35   /* ..38: potpourri_bonuses[4] */
#define S_PFED        39   /* ..42: players_fed[4] */
#define S_FOOD_NEED   43
#define S_PL_PASS     44   /* ..47: placement_passed[4] */
#define S_IS_CHANCE   48
#define S_CH_TYPE     49
#define S_CH_NDICE    50
#define S_CH_RES      51
#define S_POTP_ORD    52   /* ..55: potpourri order[4] */
#define S_POTP_PIDX   56   /* which player picks next */
#define S_POTP_NTOT   57
#define S_POTP_NROLL  58
#define S_PHASE_END   59

/* ── Board locations ── */
#define S_LOCFIG      59   /* 8 locs × 4 players = 32 */
/* loc l, player p: S_LOCFIG + l*4 + p */
#define S_BLOCKED     91   /* 0=none 1=TM 2=hut 3=field */
#define S_SUPPLY      92   /* ..95: supply[4] */
#define S_CARD_ID     96   /* ..99: card_ids[4] */
#define S_CARD_FIG   100   /* ..103: card_figures[4] (-1=none) */
#define S_BLD_ID     104   /* ..107: building_ids[4] */
#define S_BLD_FIG    108   /* ..111: building_figures[4] */
#define S_BOARD_END  112

/* ── Players ── */
/* Each player block: 80 ints */
#define PB_RES        0   /* +0..+3: resources[4] */
#define PB_FOOD       4
#define PB_FIGS       5
#define PB_AVAIL      6
#define PB_AGRI       7
#define PB_TOOL       8   /* +8..+10: tools[3] */
#define PB_TUSED     11   /* +11..+13: tools_used[3] */
#define PB_SCORE     14
#define PB_NOT       15   /* num one-time tools */
#define PB_OT        16   /* +16..+23: one_time_tools[8] */
#define PB_CIV       24   /* +24..+57: civ_counts[34] */
#define PB_BLD       58   /* +58..+79: bld_counts[22] */
#define PB_SIZE      80

#define S_PLAYERS    112  /* start of player blocks */
/* player p, field f: S_PLAYERS + p*PB_SIZE + f */
#define S_PLAYERS_END (S_PLAYERS + MAX_PLAYERS * PB_SIZE)  /* 432 */

/* ── Deck ── */
#define S_DECK_SZ    432
#define S_DECK       433  /* ..468: deck[36] ordered card IDs */
#define S_DECK_END   469

/* ── Building stacks ── */
#define S_STK_SZ     469  /* ..472: stack_sizes[4] */
#define S_STK        473  /* 4 stacks × 7 = 28 */
/* stack s, pos j: S_STK + s*7 + j */
#define S_STK_END    501

/* ── RNG ── */
#define S_RNG        501  /* 4 ints: rng0_lo, rng0_hi, rng1_lo, rng1_hi */
#define S_RNG_END    505

#define STATE_SIZE   505

/* ════════════════════════════════════════════
   Public API
   ════════════════════════════════════════════ */

/* Initialize a new game. Caller provides int32_t state[STATE_SIZE]. */
void sa_init(int32_t* state, int num_players, uint64_t seed);

/* Deep copy (just memcpy STATE_SIZE × 4 bytes). */
void sa_clone(int32_t* dst, const int32_t* src);

/* Write 424-element player-rotated observation. */
void sa_get_obs(const int32_t* state, int32_t* obs);

/* Write legal action IDs into out[]. Returns count. */
int  sa_legal_actions(const int32_t* state, int32_t* out, int max_n);

/* Apply a player action. */
void sa_apply_action(int32_t* state, int32_t action);

/* Queries. */
int  sa_is_terminal(const int32_t* state);
int  sa_current_player(const int32_t* state);
int  sa_num_players(const int32_t* state);
int  sa_move_number(const int32_t* state);

/* Rewards: writes num_players floats. Only valid when terminal. */
void sa_get_rewards(const int32_t* state, float* out);

/* Chance nodes. */
int  sa_is_chance(const int32_t* state);
int  sa_chance_outcomes(const int32_t* state);  /* returns count */
void sa_chance_probs(const int32_t* state, float* out);
void sa_apply_chance(int32_t* state, int outcome);

#endif /* STONE_AGE_H */
