#!/bin/bash
# start_train_env.sh — launch one Q2 training instance, spectator-ready.
#
# Usage:  start_train_env.sh [port] [bot_count] [map]
#
# Sets cheats 1 so a connecting spectator can use noclip / give all.
# Reserves slot 0 for a human spectator; bots fill slots 1..bot_count.

PORT=${1:-27910}
BOTCOUNT=${2:-3}
MAP=${3:-q2dm1}
TIMEDEMO=${TIMEDEMO:-0}    # set TIMEDEMO=1 to run server uncapped (training)

Q2_ROOT="${Q2_ROOT:-/home/raymond/q2_lithium_merge}"

# botlist by count: 1v1sk1 (1 bot), 2v2sk1 (4 bots), 4v4sk1 (8 bots)
case "$BOTCOUNT" in
    1) BOTLIST=1v1sk1 ;;
    2|3|4) BOTLIST=2v2sk1 ;;
    *) BOTLIST=4v4sk1 ;;
esac

cd "$Q2_ROOT"
exec stdbuf -oL -eL ./q2ded \
    +set game        lithium \
    +set dedicated   1 \
    +set ip          127.0.0.1 \
    +set port        "$PORT" \
    +set deathmatch  1 \
    +set cheats      1 \
    +set autospawn   1 \
    +set botlist     "$BOTLIST" \
    +set ml_enabled  1 \
    +set ml_bot_slot 7 \
    +set maxclients  8 \
    +set timelimit   0 \
    +set timedemo "$TIMEDEMO" \
    +map             "$MAP"
