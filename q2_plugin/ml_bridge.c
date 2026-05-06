/*
 * ml_bridge.c — UDP bridge between game.so Bot_Think and Python harness.
 *
 * Each bot slot owns one UDP socket (non-blocking send, blocking recv with
 * timeout).  The Python process binds the server side of each socket.
 *
 * Thread safety: not needed — all calls happen on the single game thread.
 */

#include "ml_bridge.h"
#include "../merge_mod/lithium/g_local.h"   /* edict_t, gi, vec3_t */

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>

#define MAX_BOTS_ML     32
#define HARNESS_ADDR    "127.0.0.1"

typedef struct {
    int             fd;             /* UDP socket, -1 if unused */
    int             bot_slot;
    uint16_t        port;           /* ML_BASE_PORT + slot */
    struct sockaddr_in harness_addr;
    ml_action_t     last_action;    /* cached for timeout fallback */
} ml_bot_sock_t;

static ml_bot_sock_t g_socks[MAX_BOTS_ML];
static int           g_initialized = 0;

static void ml_global_init(void) {
    if (g_initialized) return;
    memset(g_socks, 0, sizeof(g_socks));
    for (int i = 0; i < MAX_BOTS_ML; i++) g_socks[i].fd = -1;
    g_initialized = 1;
}

int ML_BotInit(int bot_slot) {
    ml_global_init();
    if (bot_slot < 0 || bot_slot >= MAX_BOTS_ML) return -1;

    ml_bot_sock_t *s = &g_socks[bot_slot];
    if (s->fd >= 0) close(s->fd);

    s->fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (s->fd < 0) {
        gi.dprintf("ML_BotInit: socket() failed for slot %d: %s\n",
                   bot_slot, strerror(errno));
        return -1;
    }

    s->bot_slot = bot_slot;
    s->port     = (uint16_t)(ML_BASE_PORT + bot_slot);

    memset(&s->harness_addr, 0, sizeof(s->harness_addr));
    s->harness_addr.sin_family      = AF_INET;
    s->harness_addr.sin_port        = htons(s->port);
    inet_pton(AF_INET, HARNESS_ADDR, &s->harness_addr.sin_addr);

    /* default action: stand still */
    memset(&s->last_action, 0, sizeof(s->last_action));
    s->last_action.magic = ML_ACT_MAGIC;

    gi.dprintf("ML: bot slot %d → UDP port %d\n", bot_slot, s->port);
    return 0;
}

int ML_BotStep(int bot_slot, const ml_obs_t *obs, ml_action_t *act,
               int timeout_ms) {
    if (bot_slot < 0 || bot_slot >= MAX_BOTS_ML) return -1;
    ml_bot_sock_t *s = &g_socks[bot_slot];
    if (s->fd < 0) { *act = s->last_action; return -1; }

    /* send observation */
    ssize_t sent = sendto(s->fd, obs, sizeof(*obs), 0,
                          (struct sockaddr *)&s->harness_addr,
                          sizeof(s->harness_addr));
    if (sent != sizeof(*obs)) {
        *act = s->last_action;
        return -1;
    }

    /* wait for action with timeout */
    struct timeval tv = { 0, timeout_ms * 1000 };
    setsockopt(s->fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    ml_action_t incoming;
    ssize_t n = recv(s->fd, &incoming, sizeof(incoming), 0);
    if (n == sizeof(incoming) && incoming.magic == ML_ACT_MAGIC
        && incoming.tick == obs->tick) {
        s->last_action = incoming;
        *act = incoming;
        return 0;
    }

    /* timeout or bad packet — reuse last action */
    *act = s->last_action;
    return -1;
}

void ML_BotShutdown(int bot_slot) {
    if (bot_slot < 0 || bot_slot >= MAX_BOTS_ML) return;
    ml_bot_sock_t *s = &g_socks[bot_slot];
    if (s->fd >= 0) { close(s->fd); s->fd = -1; }
}

/* ── Ray casting ────────────────────────────────────────────────────── */

void ML_FillRays(edict_t *ent, ml_obs_t *obs) {
    static const float angles[ML_RAY_COUNT] = {
        0, 22.5f, 45, 67.5f, 90, 112.5f, 135, 157.5f,
        180, 202.5f, 225, 247.5f, 270, 292.5f, 315, 337.5f
    };

    vec3_t origin;
    VectorCopy(ent->s.origin, origin);
    origin[2] += ent->viewheight;

    float yaw_rad = ent->s.angles[YAW] * (M_PI / 180.0f);

    for (int i = 0; i < ML_RAY_COUNT; i++) {
        float a = (angles[i] + ent->s.angles[YAW]) * (M_PI / 180.0f);
        vec3_t end;
        end[0] = origin[0] + cosf(a) * 2048.0f;
        end[1] = origin[1] + sinf(a) * 2048.0f;
        end[2] = origin[2];

        trace_t tr = gi.trace(origin, NULL, NULL, end, ent,
                              CONTENTS_SOLID | CONTENTS_WINDOW);

        obs->rays[i].direction[0] = cosf(a);
        obs->rays[i].direction[1] = sinf(a);
        obs->rays[i].direction[2] = 0;
        obs->rays[i].distance = (tr.fraction < 1.0f)
                                ? tr.fraction * 2048.0f
                                : -1.0f;
    }
    (void)yaw_rad;
}

/* ── Hook zone lookup ───────────────────────────────────────────────── */
/* Populated later when nav annotation sidecar is implemented.
   For now fills zeros so the obs is structurally complete. */
void ML_FillHookZones(edict_t *ent, ml_obs_t *obs) {
    obs->hook_zone_count = 0;
    memset(obs->hook_zones, 0, sizeof(obs->hook_zones));
    (void)ent;
}
