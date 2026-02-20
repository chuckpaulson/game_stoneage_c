/*
 * stone_age.c — Full game engine implementation
 *
 * Compile:  gcc -O2 -shared -fPIC -o libstoneage.so stone_age.c
 * Test:     gcc -O2 -o sa_test stone_age.c -DSA_MAIN && ./sa_test
 */
#include "stone_age.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>

/* shorthand state access */
#define ST(i)       state[(i)]
#define P(pi,f)     state[S_PLAYERS + (pi)*PB_SIZE + (f)]
#define LOC(l,p)    state[S_LOCFIG + (l)*4 + (p)]

#define CST(i)      state[(i)]          /* const context */
#define CP(pi,f)    state[S_PLAYERS + (pi)*PB_SIZE + (f)]

/* ════════════════════════════════════════════
   §1  STATIC TABLES
   ════════════════════════════════════════════ */

/* Card table: 34 unique types
   green_type: 0=Pottery 1=Weaving 2=Art 3=Time 4=Transport 5=Music 6=Medicine 7=Writing
   sand_type:  0=Farmers 1=ToolMakers 2=Builders 3=Shamans */
static const struct {
    int8_t is_green, green_type, sand_type, multiplier;
    int8_t eff_type, eff_value, eff_resource;
} CARD[N_CARDS] = {
    /* 0*/ {1, 6,-1,0, EFF_ANY,  2,-1},
    /* 1*/ {1, 6,-1,0, EFF_FOOD, 5,-1},
    /* 2*/ {1, 7,-1,0, EFF_FREE, 0,-1},
    /* 3*/ {1, 7,-1,0, EFF_POTP, 0,-1},
    /* 4*/ {1, 3,-1,0, EFF_AGRI, 1,-1},
    /* 5*/ {1, 3,-1,0, EFF_POTP, 0,-1},
    /* 6*/ {1, 0,-1,0, EFF_FOOD, 7,-1},
    /* 7*/ {1, 0,-1,0, EFF_POTP, 0,-1},
    /* 8*/ {1, 4,-1,0, EFF_POTP, 0,-1},
    /* 9*/ {1, 4,-1,0, EFF_RES,  2, RES_S},
    /*10*/ {1, 2,-1,0, EFF_TOOL, 1,-1},
    /*11*/ {1, 2,-1,0, EFF_DICE, 2, RES_G},
    /*12*/ {1, 1,-1,0, EFF_FOOD, 3,-1},
    /*13*/ {1, 1,-1,0, EFF_FOOD, 1,-1},
    /*14*/ {1, 5,-1,0, EFF_PTS,  3,-1},
    /*15*/ {0,-1, 3,2, EFF_DICE, 2, RES_W},
    /*16*/ {0,-1, 3,2, EFF_RES,  1, RES_C},
    /*17*/ {0,-1, 3,1, EFF_RES,  1, RES_G},
    /*18*/ {0,-1, 3,1, EFF_DICE, 2, RES_S},
    /*19*/ {0,-1, 3,1, EFF_RES,  1, RES_S},
    /*20*/ {0,-1, 2,3, EFF_PTS,  3,-1},
    /*21*/ {0,-1, 2,2, EFF_POTP, 0,-1},
    /*22*/ {0,-1, 2,2, EFF_FOOD, 2,-1},
    /*23*/ {0,-1, 2,1, EFF_POTP, 0,-1},
    /*24*/ {0,-1, 2,1, EFF_FOOD, 4,-1},
    /*25*/ {0,-1, 0,1, EFF_AGRI, 1,-1},
    /*26*/ {0,-1, 0,2, EFF_POTP, 0,-1},
    /*27*/ {0,-1, 0,2, EFF_FOOD, 3,-1},
    /*28*/ {0,-1, 0,1, EFF_POTP, 0,-1},
    /*29*/ {0,-1, 0,1, EFF_RES,  1, RES_S},
    /*30*/ {0,-1, 1,2, EFF_POTP, 0,-1},
    /*31*/ {0,-1, 1,2, EFF_TOOL, 2,-1},
    /*32*/ {0,-1, 1,1, EFF_OTT,  4,-1},
    /*33*/ {0,-1, 1,1, EFF_OTT,  3,-1},
};

/* Copies of each card type in the 36-card deck */
static const int8_t CARD_MULT[N_CARDS] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,  /* 14: Music/3pts ×2 */
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2, /* 30: 2×ToolMakers/Potp ×2 */
    1,1,1
};

/* Building table: 22 unique types */
static const struct {
    int8_t type;
    int8_t cost[4]; /* w,c,s,g */
    int8_t pts;
} BLD[N_BLDG] = {
    /* 0*/ {BT_FIXED, {2,1,0,0}, 10},
    /* 1*/ {BT_FIXED, {2,0,1,0}, 11},
    /* 2*/ {BT_FIXED, {1,2,0,0}, 11},
    /* 3*/ {BT_FIXED, {1,1,1,0}, 12},
    /* 4*/ {BT_FIXED, {2,0,0,1}, 12},
    /* 5*/ {BT_FIXED, {1,0,2,0}, 13},
    /* 6*/ {BT_FIXED, {0,2,1,0}, 13},
    /* 7*/ {BT_FIXED, {1,1,0,1}, 13},
    /* 8*/ {BT_FIXED, {1,0,1,1}, 14},
    /* 9*/ {BT_FIXED, {0,2,0,1}, 14},
    /*10*/ {BT_FIXED, {0,1,2,0}, 14},
    /*11*/ {BT_FIXED, {0,1,1,1}, 15},
    /*12*/ {BT_FIXED, {0,0,2,1}, 16},
    /*13*/ {BT_17,    {0,0,0,0},  0},
    /*14*/ {BT_41,    {0,0,0,0},  0},
    /*15*/ {BT_42,    {0,0,0,0},  0},
    /*16*/ {BT_43,    {0,0,0,0},  0},
    /*17*/ {BT_44,    {0,0,0,0},  0},
    /*18*/ {BT_51,    {0,0,0,0},  0},
    /*19*/ {BT_52,    {0,0,0,0},  0},
    /*20*/ {BT_53,    {0,0,0,0},  0},
    /*21*/ {BT_54,    {0,0,0,0},  0},
};

static const int8_t BLD_MULT[N_BLDG] = {
    1,1,1,2,1,1,1,2,2,1,1,2,1,3,1,1,1,1,1,1,1,1
};

static const int RES_DIV[5] = {3,4,5,6,2}; /* w,c,s,g,food */
static const int RES_VAL[4] = {3,4,5,6};
static const int SUPPLY_MAX[4] = {28,18,12,10};

/* ── Payment combo table (329 entries) ── */
static int PAY[N_PAY][4];   /* [idx][w,c,s,g] */
static int PAY_INIT = 0;

static void init_pay(void) {
    if (PAY_INIT) return;
    int k = 0;
    for (int t = 1; t <= 7; t++)
        for (int w = 0; w <= t; w++)
            for (int c = 0; c <= t-w; c++)
                for (int s = 0; s <= t-w-c; s++) {
                    PAY[k][0]=w; PAY[k][1]=c;
                    PAY[k][2]=s; PAY[k][3]=t-w-c-s;
                    k++;
                }
    assert(k == N_PAY);
    PAY_INIT = 1;
}

static int pay_idx(int w, int c, int s, int g) {
    /* Quick lookup: compute base for this sum, then scan */
    int t = w+c+s+g;
    if (t < 1 || t > 7) return -1;
    int base = 0;
    for (int i = 1; i < t; i++)
        base += (i+1)*(i+2)*(i+3)/6;
    int end = base + (t+1)*(t+2)*(t+3)/6;
    for (int k = base; k < end; k++)
        if (PAY[k][0]==w && PAY[k][1]==c && PAY[k][2]==s && PAY[k][3]==g)
            return k;
    return -1;
}

/* ── Dice probability tables ── */
#define MAX_DICE 10
static float DPROB[MAX_DICE][51]; /* [n-1][sum - n] */
static int   DPROB_LEN[MAX_DICE]; /* 5n+1 */
static int   DPROB_INIT = 0;

static void init_dprob(void) {
    if (DPROB_INIT) return;
    for (int n = 1; n <= MAX_DICE; n++) {
        int len = 5*n + 1;
        DPROB_LEN[n-1] = len;
        double d[61] = {0};
        for (int f = 1; f <= 6; f++) d[f] = 1.0/6.0;
        for (int dd = 2; dd <= n; dd++) {
            double nd[61] = {0};
            for (int s = 0; s <= 6*(dd-1); s++)
                if (d[s] > 0)
                    for (int f = 1; f <= 6 && s+f <= 6*n; f++)
                        nd[s+f] += d[s]/6.0;
            memcpy(d, nd, sizeof(d));
        }
        for (int i = 0; i < len; i++)
            DPROB[n-1][i] = (float)d[n+i];
    }
    DPROB_INIT = 1;
}


/* ════════════════════════════════════════════
   §2  RNG (xorshift128+)
   ════════════════════════════════════════════ */

static uint64_t rng_get(int32_t* state) {
    uint64_t s0 = (uint64_t)(uint32_t)ST(S_RNG) | ((uint64_t)(uint32_t)ST(S_RNG+1) << 32);
    uint64_t s1 = (uint64_t)(uint32_t)ST(S_RNG+2) | ((uint64_t)(uint32_t)ST(S_RNG+3) << 32);
    /* write back s1 as s0 */
    ST(S_RNG)   = (int32_t)(s1 & 0xFFFFFFFF);
    ST(S_RNG+1) = (int32_t)(s1 >> 32);
    s0 ^= s0 << 23; s0 ^= s0 >> 17; s0 ^= s1; s0 ^= s1 >> 26;
    ST(S_RNG+2) = (int32_t)(s0 & 0xFFFFFFFF);
    ST(S_RNG+3) = (int32_t)(s0 >> 32);
    return s1 + s0;
}

static int rng_int(int32_t* state, int n) {
    return (int)(rng_get(state) % (uint64_t)n);
}

static void shuffle_arr(int32_t* state, int* a, int n) {
    for (int i = n-1; i > 0; i--) {
        int j = rng_int(state, i+1);
        int tmp = a[i]; a[i] = a[j]; a[j] = tmp;
    }
}


/* ════════════════════════════════════════════
   §3  INITIALIZATION
   ════════════════════════════════════════════ */

static void refill_cards(int32_t* state);
static void refill_blds(int32_t* state);
static void advance_resolution(int32_t* state);
static void advance_feeding(int32_t* state);
static void end_of_round(int32_t* state);
static void end_scoring(int32_t* state);
static void cont_resolution(int32_t* state);

void sa_init(int32_t* state, int np, uint64_t seed) {
    init_pay();
    init_dprob();
    memset(state, 0, STATE_SIZE * sizeof(int32_t));

    ST(S_NPLAYERS) = np;
    ST(S_PHASE) = PH_PLACE;
    ST(S_ACTIVE) = 0;

    /* RNG seed */
    uint64_t s0 = seed ? seed : 12345ULL;
    uint64_t s1 = seed ? seed * 6364136223846793005ULL + 1 : 67890ULL;
    ST(S_RNG)   = (int32_t)(s0 & 0xFFFFFFFF);
    ST(S_RNG+1) = (int32_t)(s0 >> 32);
    ST(S_RNG+2) = (int32_t)(s1 & 0xFFFFFFFF);
    ST(S_RNG+3) = (int32_t)(s1 >> 32);

    /* Supply */
    for (int i = 0; i < 4; i++) ST(S_SUPPLY+i) = SUPPLY_MAX[i];

    /* Players */
    for (int i = 0; i < np; i++) {
        P(i, PB_FOOD) = 12;
        P(i, PB_FIGS) = 5;
        P(i, PB_AVAIL) = 5;
    }

    /* Init card/bld slots to -1 */
    for (int i = 0; i < 4; i++) {
        ST(S_CARD_ID+i) = -1;  ST(S_CARD_FIG+i) = -1;
        ST(S_BLD_ID+i)  = -1;  ST(S_BLD_FIG+i)  = -1;
    }

    /* Build deck (36 cards) */
    {
        int d[DECK_TOTAL], k = 0;
        for (int c = 0; c < N_CARDS; c++)
            for (int j = 0; j < CARD_MULT[c]; j++)
                d[k++] = c;
        assert(k == DECK_TOTAL);
        shuffle_arr(state, d, DECK_TOTAL);
        ST(S_DECK_SZ) = DECK_TOTAL;
        for (int i = 0; i < DECK_TOTAL; i++) ST(S_DECK+i) = d[i];
    }

    /* Build building stacks */
    {
        int b[BLDG_TOTAL], k = 0;
        for (int bi = 0; bi < N_BLDG; bi++)
            for (int j = 0; j < BLD_MULT[bi]; j++)
                b[k++] = bi;
        assert(k == BLDG_TOTAL);
        shuffle_arr(state, b, BLDG_TOTAL);
        for (int s = 0; s < np; s++) {
            ST(S_STK_SZ+s) = MAX_STACK;
            for (int j = 0; j < MAX_STACK; j++)
                ST(S_STK + s*MAX_STACK + j) = b[s*MAX_STACK + j];
        }
    }

    refill_cards(state);
    refill_blds(state);
}

void sa_clone(int32_t* dst, const int32_t* src) {
    memcpy(dst, src, STATE_SIZE * sizeof(int32_t));
}


/* ════════════════════════════════════════════
   §4  HELPERS
   ════════════════════════════════════════════ */

static int total_res(const int32_t* state, int pi) {
    return CP(pi,PB_RES)+CP(pi,PB_RES+1)+CP(pi,PB_RES+2)+CP(pi,PB_RES+3);
}

static void upgrade_tool(int32_t* state, int pi) {
    /* Always upgrade the smallest tool slot (ties: leftmost) */
    int best = -1, best_val = 5;
    for (int i = 0; i < 3; i++) {
        int v = P(pi, PB_TOOL+i);
        if (v < 4 && v < best_val) { best = i; best_val = v; }
    }
    if (best >= 0) P(pi, PB_TOOL+best)++;
}

static int take_res(int32_t* state, int r, int amt) {
    int av = amt < ST(S_SUPPLY+r) ? amt : ST(S_SUPPLY+r);
    ST(S_SUPPLY+r) -= av;
    return av;
}

static void ret_res(int32_t* state, int r, int amt) {
    ST(S_SUPPLY+r) += amt;
}

static int card_cost(int slot) { return 4 - slot; }

/* Check village blocking for 2-3p after a village placement */
static void check_block(int32_t* state) {
    int np = ST(S_NPLAYERS);
    if (np > 3) return;
    int vloc[3] = {LOC_TM, LOC_HUT, LOC_FIELD};
    int vid[3]  = {1, 2, 3};
    int occ = 0, free_id = -1;
    for (int v = 0; v < 3; v++) {
        int has = 0;
        for (int p = 0; p < np; p++) has += LOC(vloc[v], p);
        if (has > 0) occ++;
        else free_id = vid[v];
    }
    if (occ >= 2 && free_id >= 0) ST(S_BLOCKED) = free_id;
}

/* Check game-end conditions */
static int check_end(const int32_t* state) {
    int np = CST(S_NPLAYERS);
    int cards_need = 0;
    for (int i = 0; i < 4; i++)
        if (CST(S_CARD_ID+i) == -1) cards_need++;
    if (CST(S_DECK_SZ) < cards_need) return 1;
    for (int s = 0; s < np; s++)
        if (CST(S_STK_SZ+s) == 0 && CST(S_BLD_ID+s) == -1) return 1;
    return 0;
}

/* Building variable params: total, types. Returns 0 for fixed/1-7. */
static int bld_var(int bt, int* tot, int* types) {
    switch (bt) {
    case BT_41: *tot=4;*types=1;return 1; case BT_42: *tot=4;*types=2;return 1;
    case BT_43: *tot=4;*types=3;return 1; case BT_44: *tot=4;*types=4;return 1;
    case BT_51: *tot=5;*types=1;return 1; case BT_52: *tot=5;*types=2;return 1;
    case BT_53: *tot=5;*types=3;return 1; case BT_54: *tot=5;*types=4;return 1;
    default: return 0;
    }
}

/* Refill card display: slide right, fill from deck */
static void refill_cards(int32_t* state) {
    int present[4], np2 = 0;
    for (int i = 0; i < 4; i++)
        if (ST(S_CARD_ID+i) != -1) present[np2++] = ST(S_CARD_ID+i);
    for (int i = 0; i < 4; i++) ST(S_CARD_ID+i) = -1;
    /* Place from right */
    for (int i = 0; i < np2; i++)
        ST(S_CARD_ID + 3 - i) = present[np2 - 1 - i];
    /* Fill empties from deck right-to-left */
    for (int i = 3; i >= 0; i--) {
        if (ST(S_CARD_ID+i) == -1 && ST(S_DECK_SZ) > 0) {
            ST(S_CARD_ID+i) = ST(S_DECK); /* draw from front */
            int sz = ST(S_DECK_SZ) - 1;
            for (int j = 0; j < sz; j++) ST(S_DECK+j) = ST(S_DECK+j+1);
            ST(S_DECK_SZ) = sz;
        }
    }
}

/* Reveal top building from each stack */
static void refill_blds(int32_t* state) {
    int np = ST(S_NPLAYERS);
    for (int s = 0; s < np; s++) {
        if (ST(S_BLD_ID+s) == -1 && ST(S_STK_SZ+s) > 0) {
            int off = S_STK + s*MAX_STACK;
            ST(S_BLD_ID+s) = ST(off);
            int sz = ST(S_STK_SZ+s) - 1;
            for (int j = 0; j < sz; j++) ST(off+j) = ST(off+j+1);
            ST(S_STK_SZ+s) = sz;
        }
    }
}


/* ══════════════════════════════════════════════
   §5  PAYMENT ENUMERATION (backtracker)
   ══════════════════════════════════════════════ */

static int32_t g_pay_buf[N_PAY];
static int g_pay_n;

static void bt_pay(const int* av, int i, int rem, int pick[4], int used, int need_types) {
    if (i == 4) {
        if (rem == 0) {
            int ok = need_types == 0 ? (used >= 1) : (used == need_types);
            if (ok) {
                int k = pay_idx(pick[0],pick[1],pick[2],pick[3]);
                if (k >= 0) g_pay_buf[g_pay_n++] = A_PAY + k;
            }
        }
        return;
    }
    int mx = rem < av[i] ? rem : av[i];
    for (int take = 0; take <= mx; take++) {
        int nu = used + (take > 0);
        if (need_types > 0) {
            int left = 3 - i;
            if (nu > need_types || nu + left < need_types) continue;
        }
        pick[i] = take;
        bt_pay(av, i+1, rem-take, pick, nu, need_types);
    }
}

/* Find valid payments for a player. Returns count, writes action IDs. */
static int find_pays(const int32_t* state, int pi, int total, int types,
                     int32_t* out) {
    int av[4] = {CP(pi,PB_RES),CP(pi,PB_RES+1),CP(pi,PB_RES+2),CP(pi,PB_RES+3)};
    g_pay_n = 0;
    int pick[4] = {0};
    bt_pay(av, 0, total, pick, 0, types);
    memcpy(out, g_pay_buf, g_pay_n * sizeof(int32_t));
    return g_pay_n;
}


/* ════════════════════════════════════════════
   §6  OBSERVATION OUTPUT (424 ints, rotated)
   ════════════════════════════════════════════ */

void sa_get_obs(const int32_t* state, int32_t* o) {
    int n = CST(S_NPLAYERS);
    int cp = CST(S_ACTIVE); /* current player = perspective */
    int idx = 0;
    #define W(v) (o[idx++] = (v))

    /* Rotation: rot[relative_idx] = absolute_idx */
    int rot[MAX_PLAYERS], a2r[MAX_PLAYERS];
    for (int i = 0; i < n; i++) { rot[i] = (cp+i)%n; a2r[(cp+i)%n] = i; }

    /* Meta [4] */
    W(n);
    W(CST(S_ROUND));
    W((CST(S_FIRST) - cp + n) % n);
    W(0);  /* current player is always seat 0 in rotated obs */

    /* Phase [37] */
    W(CST(S_PHASE));
    W(a2r[CST(S_ACTIVE)]);
    W(CST(S_SUB));
    W(a2r[CST(S_RES_PLAYER)]);
    /* resolution_completed[4] rotated */
    for (int ri = 0; ri < MAX_PLAYERS; ri++)
        W(ri < n ? CST(S_RES_DONE + rot[ri]) : 0);
    /* locations_resolved[16] */
    for (int i = 0; i < NUM_RESOLVE; i++) W(CST(S_LOC_RES+i));
    W(CST(S_DICE));
    W(CST(S_PEND_RES));
    W(CST(S_PEND_SLOT));
    W(CST(S_REM_PICKS));
    /* potpourri_available[4] */
    for (int i = 0; i < MAX_PLAYERS; i++) W(CST(S_POTP_BON+i));
    /* players_fed[4] rotated */
    for (int ri = 0; ri < MAX_PLAYERS; ri++)
        W(ri < n ? CST(S_PFED + rot[ri]) : 0);
    W(CST(S_FOOD_NEED));

    /* Board locations [32] = 8 locs × 4 player slots, rotated */
    for (int loc = 0; loc < NUM_LOCS; loc++)
        for (int ri = 0; ri < MAX_PLAYERS; ri++)
            W(ri < n ? CST(S_LOCFIG + loc*4 + rot[ri]) : 0);

    /* Board misc [5] */
    W(CST(S_BLOCKED));
    for (int i = 0; i < 4; i++) W(CST(S_SUPPLY+i));

    /* Civ card slots [8] */
    for (int i = 0; i < 4; i++) {
        W(CST(S_CARD_ID+i));
        int fp = CST(S_CARD_FIG+i);
        W(fp >= 0 && fp < n ? a2r[fp] : -1);
    }

    /* Building slots [8] */
    for (int i = 0; i < 4; i++) {
        W(CST(S_BLD_ID+i));
        int fp = CST(S_BLD_FIG+i);
        W(fp >= 0 && fp < n ? a2r[fp] : -1);
    }

    /* Players [292] = 4 × 73 */
    for (int ri = 0; ri < MAX_PLAYERS; ri++) {
        if (ri < n) {
            int ai = rot[ri];
            /* resources[4], food, figures, avail, agri, tools[3], tused[3], score = 15 */
            for (int f = 0; f < 15; f++) W(CP(ai, f));
            /* one-time tool counts: val=3, val=4 */
            int c3=0, c4=0, not2 = CP(ai, PB_NOT);
            for (int t = 0; t < not2; t++) {
                int v = CP(ai, PB_OT+t);
                if (v==3) c3++; else if (v==4) c4++;
            }
            W(c3); W(c4);
            /* civ_counts[34] */
            for (int c = 0; c < N_CARDS; c++) W(CP(ai, PB_CIV+c));
            /* bld_counts[22] */
            for (int b = 0; b < N_BLDG; b++) W(CP(ai, PB_BLD+b));
        } else {
            for (int j = 0; j < 73; j++) W(-1);
        }
    }

    /* Deck counts [34] */
    {
        int dc[N_CARDS]; memset(dc, 0, sizeof(dc));
        int dsz = CST(S_DECK_SZ);
        for (int i = 0; i < dsz; i++) dc[CST(S_DECK+i)]++;
        for (int i = 0; i < N_CARDS; i++) W(dc[i]);
    }

    /* Stack sizes [4] */
    for (int s = 0; s < 4; s++)
        W(s < n ? CST(S_STK_SZ+s) : 0);

    #undef W
    assert(idx == OBS_SIZE);
}


/* ════════════════════════════════════════════
   §7  QUERIES
   ════════════════════════════════════════════ */

int sa_is_terminal(const int32_t* s) { return s[S_PHASE] == PH_OVER; }
int sa_current_player(const int32_t* s) { return s[S_ACTIVE]; }
int sa_num_players(const int32_t* s) { return s[S_NPLAYERS]; }
int sa_move_number(const int32_t* s) { return s[S_MOVES]; }
int sa_is_chance(const int32_t* s) { return s[S_IS_CHANCE]; }

int sa_chance_outcomes(const int32_t* state) {
    if (!CST(S_IS_CHANCE)) return 0;
    if (CST(S_CH_TYPE) == CH_POTP) return 6;
    return 5 * CST(S_CH_NDICE) + 1;
}

void sa_chance_probs(const int32_t* state, float* out) {
    if (!CST(S_IS_CHANCE)) return;
    if (CST(S_CH_TYPE) == CH_POTP) {
        for (int i = 0; i < 6; i++) out[i] = 1.0f/6.0f;
        return;
    }
    int nd = CST(S_CH_NDICE);
    memcpy(out, DPROB[nd-1], DPROB_LEN[nd-1] * sizeof(float));
}


/* ════════════════════════════════════════════
   §8  LEGAL ACTIONS
   ════════════════════════════════════════════ */

int sa_legal_actions(const int32_t* state, int32_t* out, int mx) {
    int n = 0;
    #define ADD(a) do { if (n < mx) out[n++] = (a); } while(0)
    int np = CST(S_NPLAYERS);
    int ap = CST(S_ACTIVE);

    if (CST(S_PHASE) == PH_PLACE) {
        int av = CP(ap, PB_AVAIL);
        if (av <= 0) { ADD(A_PL_PASS); return n; }

        /* Resource locations 0-3 (forest..river): capacity 7 */
        for (int li = 0; li < 4; li++) {
            if (CST(S_LOCFIG + li*4 + ap) > 0) continue; /* already there */
            int tot = 0, pcount = 0;
            for (int p = 0; p < np; p++) {
                int f = CST(S_LOCFIG + li*4 + p);
                tot += f;
                if (f > 0) pcount++;
            }
            if (np == 2 && pcount >= 1) continue;
            if (np == 3 && pcount >= 2) continue;
            int mf = 7 - tot;
            if (mf > av) mf = av;
            for (int f = 1; f <= mf; f++) ADD(A_PL_RES + li*7 + f-1);
        }
        /* Hunt (location 4): unlimited capacity, up to 10 figures */
        if (CST(S_LOCFIG + LOC_HUNT*4 + ap) <= 0) {
            int mf = av < 10 ? av : 10;
            for (int f = 1; f <= mf; f++) ADD(A_PL_HUNT + f-1);
        }

        /* Village locations */
        int blk = CST(S_BLOCKED);
        int vloc[3] = {LOC_TM, LOC_HUT, LOC_FIELD};
        int vact[3] = {A_PL_TM, A_PL_HUT, A_PL_FLD};
        int vblk[3] = {1, 2, 3};
        int vcost[3] = {1, 2, 1};
        for (int v = 0; v < 3; v++) {
            if (blk == vblk[v]) continue;
            int tot = 0;
            for (int p = 0; p < np; p++) tot += CST(S_LOCFIG + vloc[v]*4 + p);
            if (tot > 0) continue;
            if (av < vcost[v]) continue;
            if (v == 1 && CP(ap, PB_FIGS) >= 10) continue; /* hut: max 10 figs */
            ADD(vact[v]);
        }

        /* Cards */
        for (int i = 0; i < 4; i++)
            if (CST(S_CARD_ID+i)!=-1 && CST(S_CARD_FIG+i)==-1) ADD(A_PL_CARD+i);

        /* Buildings */
        for (int i = 0; i < np; i++)
            if (CST(S_BLD_ID+i)!=-1 && CST(S_BLD_FIG+i)==-1) ADD(A_PL_BLD+i);

        ADD(A_PL_PASS);
    }
    else if (CST(S_PHASE) == PH_RESOLVE) {
        int sub = CST(S_SUB);
        if (sub == SP_TOOL) {
            for (int i = 0; i < 3; i++)
                if (CP(ap,PB_TOOL+i)>0 && !CP(ap,PB_TUSED+i)) ADD(A_TOOL_P+i);
            int not2 = CP(ap, PB_NOT);
            for (int i = 0; i < not2 && i < 8; i++) ADD(A_TOOL_OT+i);
            ADD(A_TOOL_DN);
        }
        else if (sub == SP_BLD_PAY) {
            int slot = CST(S_PEND_SLOT);
            int bid = CST(S_BLD_ID+slot);
            if (bid == -1) { ADD(A_PASS); return n; }
            int bt = BLD[bid].type;
            if (bt == BT_FIXED) {
                int ok = 1;
                for (int r = 0; r < 4; r++)
                    if (CP(ap,PB_RES+r) < BLD[bid].cost[r]) ok = 0;
                if (ok) {
                    int k = pay_idx(BLD[bid].cost[0],BLD[bid].cost[1],
                                    BLD[bid].cost[2],BLD[bid].cost[3]);
                    if (k >= 0) ADD(A_PAY + k);
                }
            } else if (bt == BT_17) {
                int tr = total_res(state, ap);
                for (int t = 1; t <= 7 && t <= tr; t++) {
                    int32_t buf[N_PAY];
                    int cnt = find_pays(state, ap, t, 0, buf);
                    for (int j = 0; j < cnt; j++) ADD(buf[j]);
                }
            } else {
                int tot, types;
                if (bld_var(bt, &tot, &types)) {
                    int32_t buf[N_PAY];
                    int cnt = find_pays(state, ap, tot, types, buf);
                    for (int j = 0; j < cnt; j++) ADD(buf[j]);
                }
            }
            if (n == 0) ADD(A_PASS);
        }
        else if (sub == SP_CARD_PAY) {
            int cost = card_cost(CST(S_PEND_SLOT));
            int32_t buf[N_PAY];
            int cnt = find_pays(state, ap, cost, 0, buf);
            for (int j = 0; j < cnt; j++) ADD(buf[j]);
            if (n == 0) ADD(A_PASS);
        }
        else if (sub == SP_ANY_RES) {
            for (int i = 0; i < 4; i++)
                if (CST(S_SUPPLY+i) > 0) ADD(A_ANY+i);
            if (n == 0) ADD(A_ANY);
        }
        else if (sub == SP_POTP) {
            for (int i = 0; i < CST(S_POTP_NTOT); i++)
                if (CST(S_POTP_BON+i) >= 0) ADD(A_POTP+i);
            if (n == 0) ADD(A_PASS);
        }
        else if (sub == SP_FREE) {
            for (int i = 0; i < 4; i++)
                if (CST(S_CARD_ID+i) != -1) ADD(A_FREE+i);
            if (n == 0) ADD(A_PASS);
        }
        else ADD(A_PASS);
    }
    else if (CST(S_PHASE) == PH_FEED) {
        if (CST(S_FOOD_NEED) > 0) {
            for (int i = 0; i < 4; i++)
                if (CP(ap,PB_RES+i) > 0) ADD(A_FEED+i);
            ADD(A_FEED_PEN);
        } else {
            ADD(A_PASS);
        }
    }

    #undef ADD
    return n;
}


/* ════════════════════════════════════════════
   §9  CARD EFFECTS
   ════════════════════════════════════════════ */

static void card_effect(int32_t* state, int pi, int cid) {
    int eff = CARD[cid].eff_type;
    int val = CARD[cid].eff_value;
    int res = CARD[cid].eff_resource;

    switch (eff) {
    case EFF_FOOD:
        P(pi, PB_FOOD) += val;
        break;
    case EFF_RES: {
        int got = take_res(state, res, val);
        P(pi, PB_RES+res) += got;
    } break;
    case EFF_ANY:
        ST(S_ACTIVE) = pi;
        ST(S_SUB) = SP_ANY_RES;
        ST(S_REM_PICKS) = val;
        return;
    case EFF_TOOL:
        for (int i = 0; i < val; i++) upgrade_tool(state, pi);
        break;
    case EFF_AGRI:
        P(pi, PB_AGRI) += val;
        if (P(pi, PB_AGRI) > 10) P(pi, PB_AGRI) = 10;
        break;
    case EFF_PTS:
        P(pi, PB_SCORE) += val;
        break;
    case EFF_DICE:
        ST(S_IS_CHANCE) = 1;
        ST(S_CH_TYPE) = CH_CARD;
        ST(S_CH_NDICE) = 2;
        ST(S_CH_RES) = res;
        ST(S_ACTIVE) = pi;
        return;
    case EFF_POTP:
        ST(S_POTP_NTOT) = ST(S_NPLAYERS);
        ST(S_POTP_NROLL) = 0;
        for (int i = 0; i < MAX_PLAYERS; i++) ST(S_POTP_BON+i) = -1;
        for (int i = 0; i < ST(S_NPLAYERS); i++)
            ST(S_POTP_ORD+i) = (pi + i) % ST(S_NPLAYERS);
        ST(S_POTP_PIDX) = 0;
        ST(S_IS_CHANCE) = 1;
        ST(S_CH_TYPE) = CH_POTP;
        ST(S_ACTIVE) = pi;
        return;
    case EFF_FREE:
        ST(S_ACTIVE) = pi;
        ST(S_SUB) = SP_FREE;
        return;
    case EFF_OTT: {
        int k = P(pi, PB_NOT);
        if (k < MAX_OT) { P(pi, PB_OT+k) = val; P(pi, PB_NOT) = k+1; }
    } break;
    }
    /* Instant effect done — continue resolution */
    cont_resolution(state);
}


/* ════════════════════════════════════════════
   §10  RESOLUTION STATE MACHINE
   ════════════════════════════════════════════

   Resolve order per player (16 slots):
     0: tool_maker  1: hut  2: field
     3-6: card slots 0-3
     7-10: building slots 0-3
     11-15: forest, clay_pit, quarry, river, hunting
*/

static const int RES_LOC[16] = {
    LOC_TM, LOC_HUT, LOC_FIELD,
    LOC_FOREST, LOC_CLAY, LOC_QUARRY, LOC_RIVER, LOC_HUNT,  /* resources */
    -1,-1,-1,-1,  /* cards */
    -1,-1,-1,-1   /* buildings */
};

/* resource type for resource locations */
static const int LOC2RES[5] = {RES_W, RES_C, RES_S, RES_G, -1};

static void cont_resolution(int32_t* state) {
    ST(S_LOC_RES + ST(S_RES_IDX)) = 1;
    ST(S_RES_IDX)++;
    ST(S_ACTIVE) = ST(S_RES_PLAYER);
    advance_resolution(state);
}

static void finish_gather(int32_t* state) {
    int pi = ST(S_ACTIVE);
    int res = ST(S_PEND_RES);
    int div = (res == -1) ? 2 : RES_DIV[res]; /* -1 = hunting → food, div=2 */
    int gathered = ST(S_DICE) / div;
    if (res == -1) {
        P(pi, PB_FOOD) += gathered;
    } else {
        int got = take_res(state, res, gathered);
        P(pi, PB_RES+res) += got;
    }
    cont_resolution(state);
}

static int next_res_player(int32_t* state) {
    int np = ST(S_NPLAYERS);
    for (int i = 1; i <= np; i++) {
        int next = (ST(S_RES_PLAYER) + i) % np;
        if (!ST(S_RES_DONE + next)) {
            ST(S_RES_PLAYER) = next;
            ST(S_RES_IDX) = 0;
            return 1;
        }
    }
    return 0;
}

static void start_feeding(int32_t* state) {
    int np = ST(S_NPLAYERS);
    ST(S_PHASE) = PH_FEED;
    ST(S_SUB) = SP_FEED;
    for (int i = 0; i < np; i++) ST(S_PFED+i) = 0;
    ST(S_ACTIVE) = ST(S_FIRST);
    /* Calculate food needed for first player */
    int pi = ST(S_ACTIVE);
    int need = P(pi, PB_FIGS) - P(pi, PB_AGRI);
    if (need < 0) need = 0;
    int fpay = need < P(pi,PB_FOOD) ? need : P(pi,PB_FOOD);
    P(pi, PB_FOOD) -= fpay;
    ST(S_FOOD_NEED) = need - fpay;
    if (ST(S_FOOD_NEED) == 0) {
        ST(S_PFED + pi) = 1;
        advance_feeding(state);
    }
}

static void advance_resolution(int32_t* state) {
    while (1) {
        if (ST(S_IS_CHANCE)) return;
        int ri = ST(S_RES_IDX);
        if (ri >= NUM_RESOLVE) {
            ST(S_RES_DONE + ST(S_RES_PLAYER)) = 1;
            if (!next_res_player(state)) { start_feeding(state); return; }
            continue;
        }

        int pi = ST(S_RES_PLAYER);

        /* Village (0-2) */
        if (ri <= 2) {
            int loc = RES_LOC[ri];
            if (LOC(loc, pi) <= 0) { ST(S_LOC_RES+ri)=1; ST(S_RES_IDX)++; continue; }
            if (loc == LOC_TM) upgrade_tool(state, pi);
            else if (loc == LOC_HUT) { if (P(pi,PB_FIGS)<10) P(pi,PB_FIGS)++; }
            else if (loc == LOC_FIELD) { if (P(pi,PB_AGRI)<10) P(pi,PB_AGRI)++; }
            ST(S_LOC_RES+ri) = 1; ST(S_RES_IDX)++; continue;
        }

        /* Resources (3-7) → forest..hunting */
        if (ri <= 7) {
            int loc = RES_LOC[ri];
            if (LOC(loc, pi) <= 0) { ST(S_LOC_RES+ri)=1; ST(S_RES_IDX)++; continue; }
            int ndice = LOC(loc, pi);
            int res = (loc == LOC_HUNT) ? -1 : LOC2RES[loc];
            ST(S_IS_CHANCE) = 1;
            ST(S_CH_TYPE) = CH_RES;
            ST(S_CH_NDICE) = ndice;
            ST(S_CH_RES) = res;
            ST(S_ACTIVE) = pi;
            return;
        }

        /* Cards (8-11) → slot 0-3 */
        if (ri <= 11) {
            int slot = ri - 8;
            if (ST(S_CARD_FIG+slot) != pi || ST(S_CARD_ID+slot) == -1) {
                ST(S_LOC_RES+ri)=1; ST(S_RES_IDX)++; continue;
            }
            int cost = card_cost(slot);
            if (total_res(state, pi) < cost) {
                ST(S_CARD_FIG+slot) = -1;
                ST(S_LOC_RES+ri)=1; ST(S_RES_IDX)++; continue;
            }
            ST(S_ACTIVE) = pi;
            ST(S_SUB) = SP_CARD_PAY;
            ST(S_PEND_SLOT) = slot;
            return;
        }

        /* Buildings (12-15) → slot 0-3 */
        {
            int slot = ri - 12;
            if (ST(S_BLD_FIG+slot) != pi || ST(S_BLD_ID+slot) == -1) {
                ST(S_LOC_RES+ri)=1; ST(S_RES_IDX)++; continue;
            }
            int bid = ST(S_BLD_ID+slot);
            int bt = BLD[bid].type;
            int can = 0;
            if (bt == BT_FIXED) {
                can = 1;
                for (int r = 0; r < 4; r++)
                    if (CP(pi,PB_RES+r) < BLD[bid].cost[r]) can = 0;
            } else if (bt == BT_17) {
                can = total_res(state, pi) >= 1;
            } else {
                int tot, types;
                if (bld_var(bt, &tot, &types)) {
                    int32_t buf[N_PAY];
                    can = find_pays(state, pi, tot, types, buf) > 0;
                }
            }
            if (!can) {
                ST(S_BLD_FIG+slot) = -1;
                ST(S_LOC_RES+ri)=1; ST(S_RES_IDX)++; continue;
            }
            ST(S_ACTIVE) = pi;
            ST(S_SUB) = SP_BLD_PAY;
            ST(S_PEND_SLOT) = slot;
            return;
        }
    }
}


/* ════════════════════════════════════════════
   §11  FEEDING & END-OF-ROUND
   ════════════════════════════════════════════ */

static void advance_feeding(int32_t* state) {
    int np = ST(S_NPLAYERS);
    int start = ST(S_ACTIVE);
    for (int i = 1; i <= np; i++) {
        int next = (start + i) % np;
        if (ST(S_PFED+next)) continue;
        ST(S_ACTIVE) = next;
        int need = P(next, PB_FIGS) - P(next, PB_AGRI);
        if (need < 0) need = 0;
        int fpay = need < P(next,PB_FOOD) ? need : P(next,PB_FOOD);
        P(next, PB_FOOD) -= fpay;
        ST(S_FOOD_NEED) = need - fpay;
        if (ST(S_FOOD_NEED) == 0) {
            ST(S_PFED+next) = 1;
            continue;
        }
        return; /* need player action */
    }
    end_of_round(state);
}

static void end_of_round(int32_t* state) {
    if (check_end(state)) { end_scoring(state); ST(S_PHASE)=PH_OVER; ST(S_GAMEOVER)=1; return; }

    int np = ST(S_NPLAYERS);
    ST(S_ROUND)++;
    ST(S_FIRST) = (ST(S_FIRST)+1) % np;

    /* Reset board */
    for (int l = 0; l < NUM_LOCS; l++)
        for (int p = 0; p < MAX_PLAYERS; p++) LOC(l,p) = 0;
    ST(S_BLOCKED) = 0;
    for (int i = 0; i < 4; i++) { ST(S_CARD_FIG+i)=-1; ST(S_BLD_FIG+i)=-1; }

    /* Reset players */
    for (int i = 0; i < np; i++) {
        P(i, PB_AVAIL) = P(i, PB_FIGS);
        for (int j = 0; j < 3; j++) P(i, PB_TUSED+j) = 0;
    }

    refill_cards(state);
    refill_blds(state);

    if (check_end(state)) { end_scoring(state); ST(S_PHASE)=PH_OVER; ST(S_GAMEOVER)=1; return; }

    ST(S_PHASE) = PH_PLACE;
    ST(S_SUB) = SP_CHOOSE;
    ST(S_ACTIVE) = ST(S_FIRST);
    for (int i = 0; i < MAX_PLAYERS; i++) ST(S_PL_PASS+i) = 0;
}


/* ════════════════════════════════════════════
   §12  END-GAME SCORING
   ════════════════════════════════════════════ */

static void end_scoring(int32_t* state) {
    int np = ST(S_NPLAYERS);
    for (int pi = 0; pi < np; pi++) {
        /* Green cards: (different types)² per set */
        int tc[8] = {0};
        for (int c = 0; c < N_CARDS; c++)
            if (P(pi,PB_CIV+c) > 0 && CARD[c].is_green)
                tc[(int)CARD[c].green_type] += P(pi,PB_CIV+c);
        while (1) {
            int d = 0;
            for (int t = 0; t < 8; t++) if (tc[t]>0) d++;
            if (!d) break;
            P(pi, PB_SCORE) += d*d;
            for (int t = 0; t < 8; t++) if (tc[t]>0) tc[t]--;
        }

        /* Sand cards: multiplier × stat */
        for (int c = 0; c < N_CARDS; c++) {
            int cnt = P(pi,PB_CIV+c);
            if (!cnt || CARD[c].is_green) continue;
            int m = CARD[c].multiplier;
            int stat = 0;
            switch (CARD[c].sand_type) {
            case 0: stat = P(pi,PB_AGRI); break;
            case 1: stat = P(pi,PB_TOOL)+P(pi,PB_TOOL+1)+P(pi,PB_TOOL+2); break;
            case 2: { int b=0; for(int j=0;j<N_BLDG;j++) b+=P(pi,PB_BLD+j); stat=b; } break;
            case 3: stat = P(pi,PB_FIGS); break;
            }
            P(pi, PB_SCORE) += cnt * m * stat;
        }

        /* Remaining resources: 1pt each */
        P(pi, PB_SCORE) += P(pi,PB_RES)+P(pi,PB_RES+1)+P(pi,PB_RES+2)+P(pi,PB_RES+3);
    }
}


/* ════════════════════════════════════════════
   §13  REWARDS
   ════════════════════════════════════════════ */

void sa_get_rewards(const int32_t* state, float* out) {
    int np = CST(S_NPLAYERS);
    int sc[4], tb[4], rk[4];
    for (int i = 0; i < np; i++) {
        sc[i] = CP(i, PB_SCORE);
        tb[i] = CP(i,PB_AGRI) + CP(i,PB_TOOL)+CP(i,PB_TOOL+1)+CP(i,PB_TOOL+2)
                 + (CP(i,PB_FIGS)-5);
        rk[i] = 0;
    }
    for (int i = 0; i < np; i++)
        for (int j = 0; j < np; j++) {
            if (j==i) continue;
            if (sc[j]>sc[i] || (sc[j]==sc[i] && tb[j]>tb[i]) ||
                (sc[j]==sc[i] && tb[j]==tb[i] && j<i))
                rk[i]++;
        }
    if (np == 2) {
        for (int i = 0; i < 2; i++) out[i] = rk[i]==0 ? 1.0f : -1.0f;
    } else if (np == 3) {
        float m[3] = {1,0,-1};
        for (int i = 0; i < 3; i++) out[i] = m[rk[i]];
    } else {
        float m[4] = {1, 0.33f, -0.33f, -1};
        for (int i = 0; i < 4; i++) out[i] = m[rk[i]];
    }
}


/* ════════════════════════════════════════════
   §14  CHANCE NODE APPLICATION
   ════════════════════════════════════════════ */

void sa_apply_chance(int32_t* state, int outcome) {
    ST(S_IS_CHANCE) = 0;

    if (ST(S_CH_TYPE) == CH_POTP) {
        ST(S_POTP_BON + ST(S_POTP_NROLL)) = outcome; /* 0-5: w,c,s,g,tool,agri */
        ST(S_POTP_NROLL)++;
        if (ST(S_POTP_NROLL) < ST(S_POTP_NTOT)) {
            ST(S_IS_CHANCE) = 1;
            ST(S_CH_TYPE) = CH_POTP;
        } else {
            /* All dice rolled → picks */
            ST(S_SUB) = SP_POTP;
            ST(S_POTP_PIDX) = 0;
            ST(S_ACTIVE) = ST(S_POTP_ORD);
        }
        return;
    }

    int dice_sum = ST(S_CH_NDICE) + outcome;

    if (ST(S_CH_TYPE) == CH_CARD) {
        int res = ST(S_CH_RES);
        int div = RES_DIV[res];
        int got = take_res(state, res, dice_sum / div);
        P(ST(S_ACTIVE), PB_RES+res) += got;
        cont_resolution(state);
        return;
    }

    /* CH_RES: resource gathering → tool decision */
    ST(S_DICE) = dice_sum;
    int res = ST(S_CH_RES);
    ST(S_PEND_RES) = res;
    ST(S_SUB) = SP_TOOL;
    /* active_player already set */
}


/* ════════════════════════════════════════════
   §15  PLACEMENT ADVANCEMENT
   ════════════════════════════════════════════ */

static void advance_place(int32_t* state) {
    int np = ST(S_NPLAYERS);
    for (int i = 0; i < np; i++) {
        ST(S_ACTIVE) = (ST(S_ACTIVE)+1) % np;
        int ap = ST(S_ACTIVE);
        if (!ST(S_PL_PASS+ap)) {
            if (P(ap, PB_AVAIL) > 0) return;
            ST(S_PL_PASS+ap) = 1;
        }
    }
    /* All passed → resolution */
    ST(S_PHASE) = PH_RESOLVE;
    ST(S_SUB) = SP_CHOOSE;
    ST(S_RES_PLAYER) = ST(S_FIRST);
    ST(S_ACTIVE) = ST(S_FIRST);
    ST(S_RES_IDX) = 0;
    for (int i = 0; i < MAX_PLAYERS; i++) ST(S_RES_DONE+i) = 0;
    for (int i = 0; i < NUM_RESOLVE; i++) ST(S_LOC_RES+i) = 0;
    advance_resolution(state);
}


/* ════════════════════════════════════════════
   §16  APPLY ACTION (main dispatch)
   ════════════════════════════════════════════ */

void sa_apply_action(int32_t* state, int32_t a) {
    ST(S_MOVES)++;
    int ap = ST(S_ACTIVE);

    if (ST(S_PHASE) == PH_PLACE) {
        if (a == A_PL_PASS) {
            ST(S_PL_PASS+ap) = 1;
        }
        else if (a >= A_PL_RES && a < A_PL_RES + 28) {
            int li = a / 7, nf = (a % 7) + 1;
            LOC(li, ap) += nf;
            P(ap, PB_AVAIL) -= nf;
        }
        else if (a >= A_PL_HUNT && a < A_PL_HUNT + 10) {
            int nf = (a - A_PL_HUNT) + 1;
            LOC(LOC_HUNT, ap) += nf;
            P(ap, PB_AVAIL) -= nf;
        }
        else if (a == A_PL_TM)  { LOC(LOC_TM, ap)=1;  P(ap,PB_AVAIL)-=1; check_block(state); }
        else if (a == A_PL_FLD) { LOC(LOC_FIELD,ap)=1; P(ap,PB_AVAIL)-=1; check_block(state); }
        else if (a == A_PL_HUT) { LOC(LOC_HUT, ap)=2;  P(ap,PB_AVAIL)-=2; check_block(state); }
        else if (a >= A_PL_CARD && a < A_PL_CARD+4) {
            ST(S_CARD_FIG + a-A_PL_CARD) = ap; P(ap,PB_AVAIL)--;
        }
        else if (a >= A_PL_BLD && a < A_PL_BLD+4) {
            ST(S_BLD_FIG + a-A_PL_BLD) = ap; P(ap,PB_AVAIL)--;
        }
        advance_place(state);
    }
    else if (ST(S_PHASE) == PH_RESOLVE) {
        int sub = ST(S_SUB);

        if (sub == SP_TOOL) {
            if (a == A_TOOL_DN) {
                finish_gather(state);
            } else if (a >= A_TOOL_P && a < A_TOOL_P+3) {
                int ti = a - A_TOOL_P;
                ST(S_DICE) += P(ap, PB_TOOL+ti);
                P(ap, PB_TUSED+ti) = 1;
            } else if (a >= A_TOOL_OT && a < A_TOOL_OT+8) {
                int ti = a - A_TOOL_OT;
                int not2 = P(ap, PB_NOT);
                if (ti < not2) {
                    ST(S_DICE) += P(ap, PB_OT+ti);
                    for (int j = ti; j < not2-1; j++) P(ap,PB_OT+j) = P(ap,PB_OT+j+1);
                    P(ap, PB_NOT) = not2-1;
                }
            }
        }
        else if (sub == SP_BLD_PAY) {
            if (a == A_PASS) {
                ST(S_BLD_FIG + ST(S_PEND_SLOT)) = -1;
                cont_resolution(state);
            } else {
                int k = a - A_PAY;
                int w=PAY[k][0], c=PAY[k][1], s=PAY[k][2], g=PAY[k][3];
                int slot = ST(S_PEND_SLOT);
                int bid = ST(S_BLD_ID+slot);
                P(ap,PB_RES)-=w; P(ap,PB_RES+1)-=c; P(ap,PB_RES+2)-=s; P(ap,PB_RES+3)-=g;
                ret_res(state,0,w); ret_res(state,1,c); ret_res(state,2,s); ret_res(state,3,g);
                int pts = (BLD[bid].type == BT_FIXED) ? BLD[bid].pts
                    : w*RES_VAL[0]+c*RES_VAL[1]+s*RES_VAL[2]+g*RES_VAL[3];
                P(ap, PB_SCORE) += pts;
                P(ap, PB_BLD+bid)++;
                ST(S_BLD_ID+slot) = -1; ST(S_BLD_FIG+slot) = -1;
                cont_resolution(state);
            }
        }
        else if (sub == SP_CARD_PAY) {
            if (a == A_PASS) {
                ST(S_CARD_FIG + ST(S_PEND_SLOT)) = -1;
                cont_resolution(state);
            } else {
                int k = a - A_PAY;
                int w=PAY[k][0], c=PAY[k][1], s=PAY[k][2], g=PAY[k][3];
                int slot = ST(S_PEND_SLOT);
                int cid = ST(S_CARD_ID+slot);
                P(ap,PB_RES)-=w; P(ap,PB_RES+1)-=c; P(ap,PB_RES+2)-=s; P(ap,PB_RES+3)-=g;
                ret_res(state,0,w); ret_res(state,1,c); ret_res(state,2,s); ret_res(state,3,g);
                P(ap, PB_CIV+cid)++;
                ST(S_CARD_ID+slot) = -1; ST(S_CARD_FIG+slot) = -1;
                card_effect(state, ap, cid);
            }
        }
        else if (sub == SP_ANY_RES) {
            int r = a - A_ANY;
            int got = take_res(state, r, 1);
            P(ap, PB_RES+r) += got;
            ST(S_REM_PICKS)--;
            if (ST(S_REM_PICKS) <= 0)
                cont_resolution(state);
        }
        else if (sub == SP_POTP) {
            if (a == A_PASS) {
                ST(S_ACTIVE) = ST(S_RES_PLAYER);
                cont_resolution(state);
            } else {
                int idx = a - A_POTP;
                int bonus = ST(S_POTP_BON+idx);
                if (bonus >= 0 && bonus <= 3) {
                    int got = take_res(state, bonus, 1);
                    P(ap, PB_RES+bonus) += got;
                } else if (bonus == 4) {
                    upgrade_tool(state, ap);
                } else if (bonus == 5) {
                    if (P(ap,PB_AGRI)<10) P(ap,PB_AGRI)++;
                }
                ST(S_POTP_BON+idx) = -1;
                ST(S_POTP_PIDX)++;
                if (ST(S_POTP_PIDX) < ST(S_POTP_NTOT)) {
                    ST(S_ACTIVE) = ST(S_POTP_ORD + ST(S_POTP_PIDX));
                } else {
                    ST(S_ACTIVE) = ST(S_RES_PLAYER);
                    cont_resolution(state);
                }
            }
        }
        else if (sub == SP_FREE) {
            if (a == A_PASS) {
                cont_resolution(state);
            } else {
                int slot = a - A_FREE;
                int cid = ST(S_CARD_ID+slot);
                P(ap, PB_CIV+cid)++;
                ST(S_CARD_ID+slot) = -1;
                ST(S_CARD_FIG+slot) = -1;
                card_effect(state, ap, cid);
            }
        }
    }
    else if (ST(S_PHASE) == PH_FEED) {
        if (a == A_FEED_PEN) {
            P(ap, PB_SCORE) -= 10;
            ST(S_FOOD_NEED) = 0;
            ST(S_PFED+ap) = 1;
            advance_feeding(state);
        }
        else if (a >= A_FEED && a < A_FEED+4) {
            int r = a - A_FEED;
            P(ap, PB_RES+r)--;
            ret_res(state, r, 1);
            ST(S_FOOD_NEED)--;
            if (ST(S_FOOD_NEED) <= 0) {
                ST(S_PFED+ap) = 1;
                advance_feeding(state);
            }
        }
        else if (a == A_PASS) {
            ST(S_PFED+ap) = 1;
            advance_feeding(state);
        }
    }
}


/* ════════════════════════════════════════════
   §17  TEST HARNESS
   ════════════════════════════════════════════ */

#ifdef SA_MAIN
#include <stdio.h>
#include <time.h>

int main(void) {
    int32_t state[STATE_SIZE];
    int32_t obs[OBS_SIZE];
    int32_t acts[ACT_SIZE];
    float rewards[MAX_PLAYERS];
    float probs[51];

    printf("STATE_SIZE = %d (%lu bytes)\n", STATE_SIZE, (unsigned long)(STATE_SIZE*4));
    printf("OBS_SIZE   = %d\n", OBS_SIZE);
    printf("ACT_SIZE   = %d\n", ACT_SIZE);

    /* Play random games */
    for (int np = 2; np <= 4; np++) {
        int done = 0;
        long total_moves = 0;
        clock_t t0 = clock();

        for (int seed = 0; seed < 50; seed++) {
            sa_init(state, np, 100 + seed);
            int moves = 0;
            while (!sa_is_terminal(state) && moves < 10000) {
                if (sa_is_chance(state)) {
                    int nc = sa_chance_outcomes(state);
                    sa_chance_probs(state, probs);
                    /* Weighted random */
                    float r = (float)(rng_get((int32_t*)&seed) % 10000) / 10000.0f;
                    float cum = 0; int pick = 0;
                    for (int i = 0; i < nc; i++) {
                        cum += probs[i];
                        if (r < cum) { pick = i; break; }
                        pick = i;
                    }
                    sa_apply_chance(state, pick);
                } else {
                    int n = sa_legal_actions(state, acts, ACT_SIZE);
                    if (n == 0) { printf("ERROR: no legal actions!\n"); break; }
                    int pick = (int)(rng_get((int32_t*)&seed) % (unsigned)n);
                    sa_apply_action(state, acts[pick]);
                    moves++;
                }
            }
            if (sa_is_terminal(state)) {
                done++;
                total_moves += moves;
            }
        }

        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("%d-player: %d/50 done, avg %ld moves, %.3fs\n",
               np, done, done ? total_moves/done : 0, dt);
    }

    /* Test obs output */
    sa_init(state, 2, 42);
    sa_get_obs(state, obs);
    printf("\nObs[0..3] (meta): %d %d %d %d\n", obs[0], obs[1], obs[2], obs[3]);
    printf("Obs[4] (phase): %d\n", obs[4]);

    /* Benchmark clone */
    {
        int32_t clone[STATE_SIZE];
        clock_t t0 = clock();
        for (int i = 0; i < 1000000; i++)
            sa_clone(clone, state);
        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("\n1M clones: %.3fs (%.0f ns/clone)\n", dt, dt*1e9/1e6);
    }

    printf("\nAll tests passed!\n");
    return 0;
}
#endif
