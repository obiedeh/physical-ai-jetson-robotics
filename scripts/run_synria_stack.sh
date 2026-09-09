#!/usr/bin/env bash
# Synria stack launcher — three Alicia-D tabletop games × robot on/off.
#
# Interactive menu (six options) or CLI subcommand. The 'view' action
# opens the scene USD in Isaac Sim's GUI (with the Synria arm mounted
# at -0.1524 m on world X — 6 inches left of table centre — when robot
# is on). The 'train' action runs scripts/linux_rtx/train_isaaclab_
# synria.sh with the matching TRAIN_TASK; training implies robot on.
#
# Usage:
#   bash scripts/run_synria_stack.sh                       # interactive menu
#   bash scripts/run_synria_stack.sh view ludo             # ludo scene, no robot
#   bash scripts/run_synria_stack.sh view ludo robot       # ludo + robot
#   bash scripts/run_synria_stack.sh train chess           # full training run
#
# Robot mount frame (kept in sync with view_synria_scene.py and
# isaac/isaaclab_tasks/synria_pickplace/env_cfg.py):
#   pos = (0.1524, 0.0, 0.8382)   # 6" from centre on world +X
#   rot = 90° about world +Z      # arm forward (+X URDF) → world +Y (FrontCamera)
# The /World/Game group is mirrored to the +X table end to match.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/.venv/isaacsim5/bin/python}"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/linux_rtx/train_isaaclab_synria.sh"
VIEW_SCRIPT="${REPO_ROOT}/scripts/view_synria_scene.py"

usage() {
    cat <<EOF
Synria stack launcher

Usage:
  bash $0                              interactive menu
  bash $0 view {ludo|chess|checkers} [robot]   open scene in Isaac Sim viewer
  bash $0 train {ludo|chess|checkers}          run full Isaac Lab training

Game maps to:
  ludo     → Synria-Ludo-PickPlace-v0
  chess    → Synria-Chess-PickPlace-v0
  checkers → Synria-Checkers-PickPlace-v0
EOF
}

task_for_game() {
    case "$1" in
        ludo)     echo "Synria-Ludo-PickPlace-v0" ;;
        chess)    echo "Synria-Chess-PickPlace-v0" ;;
        checkers) echo "Synria-Checkers-PickPlace-v0" ;;
        *) echo "" ;;
    esac
}

choose_interactive() {
    echo ""
    echo "Synria stack — pick a mode"
    echo "  1) Ludo     — viewer, no robot"
    echo "  2) Ludo     — viewer with robot (6\" left of centre)"
    echo "  3) Chess    — viewer, no robot"
    echo "  4) Chess    — viewer with robot"
    echo "  5) Checkers — viewer, no robot"
    echo "  6) Checkers — viewer with robot"
    echo "  7) Ludo     — train (Isaac Lab, robot on)"
    echo "  8) Chess    — train"
    echo "  9) Checkers — train"
    echo ""
    while true; do
        read -r -p "Pick [1-9]: " choice
        case "$choice" in
            1) echo "view ludo";          return ;;
            2) echo "view ludo robot";    return ;;
            3) echo "view chess";         return ;;
            4) echo "view chess robot";   return ;;
            5) echo "view checkers";      return ;;
            6) echo "view checkers robot";return ;;
            7) echo "train ludo";         return ;;
            8) echo "train chess";        return ;;
            9) echo "train checkers";     return ;;
            *) echo "Invalid — type 1-9." 1>&2 ;;
        esac
    done
}

if [ $# -eq 0 ]; then
    # shellcheck disable=SC2046
    set -- $(choose_interactive)
fi

case "${1:-}" in
    view)
        GAME="${2:-}"
        ROBOT_FLAG=""
        if [ "${3:-}" = "robot" ] || [ "${3:-}" = "with-robot" ]; then
            ROBOT_FLAG="--with-robot"
        fi
        if [ -z "$(task_for_game "$GAME")" ]; then
            echo "Unknown game '$GAME'. Use ludo|chess|checkers." 1>&2
            usage 1>&2
            exit 2
        fi
        if [ ! -x "$ISAAC_PYTHON" ]; then
            echo "Isaac Sim Python not found at $ISAAC_PYTHON" 1>&2
            echo "Override with ISAAC_PYTHON=/path/to/python" 1>&2
            exit 1
        fi
        echo "Launching Isaac Sim viewer: $GAME (robot: $([ -n "$ROBOT_FLAG" ] && echo on || echo off))"
        exec "$ISAAC_PYTHON" "$VIEW_SCRIPT" --game "$GAME" $ROBOT_FLAG
        ;;
    train)
        GAME="${2:-}"
        TASK="$(task_for_game "$GAME")"
        if [ -z "$TASK" ]; then
            echo "Unknown game '$GAME'. Use ludo|chess|checkers." 1>&2
            usage 1>&2
            exit 2
        fi
        echo "Launching Isaac Lab training: TRAIN_TASK=$TASK"
        TRAIN_TASK="$TASK" exec bash "$TRAIN_SCRIPT"
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        usage 1>&2
        exit 2
        ;;
esac
