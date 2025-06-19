#!/bin/bash

API_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source "$API_DIR/docker/utils.sh"

declare -a PIDS=()

[ ! -f .env ] || export $(grep -v '^#' .env | xargs)

get_child_pids() {
    local all_pids=()

    for pid in $(pgrep -P "$1" 2>/dev/null); do
        all_pids+=("$pid")
        local grandchild_pids=$(get_child_pids "$pid")
        for gchild in $grandchild_pids; do
            all_pids+=("$gchild")
        done
    done

    echo "${all_pids[@]}"
}

cleanup_pids() {
    local parent_pids=($1)
    local services="$2"
    local psw="$3"

    local use_sudo=0
    if [ -n "$psw" ]; then
        use_sudo=1
    fi

    color_echo blue "Shutting down processes..."
    local all_pids=()
    for pid in "${parent_pids[@]}"; do
        all_pids+=("$pid")
        for child in $(get_child_pids "$pid"); do
            all_pids+=("$child")
        done
    done
    # remove duplicates
    all_pids=($(echo "${all_pids[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

    for pid in "${all_pids[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -TERM "$pid" 2>/dev/null
        fi
    done

    sleep 2

    for pid in "${all_pids[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            if [ "$use_sudo" -eq 1 ]; then
                echo "$psw" | sudo -S kill -9 "$pid" 2>/dev/null
            else
                kill -9 "$pid" 2>/dev/null
            fi
        fi
    done

    local pid_still_running=0
    for pid in "${all_pids[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            pid_still_running=1
            color_echo red "⚠️ Process $pid is still running!"
        fi
    done

    if [ -n "$services" ]; then
        local remaining=$(ps aux | grep -E "$services" | grep -v grep | wc -l)
        if [ "$remaining" -gt 0 ]; then
            color_echo red "⚠️ $remaining processes might still be running. You may need to manually kill them."
            ps aux | grep -E "$services" | grep -v grep
        elif [ "$pid_still_running" -eq 0 ]; then
            color_echo blue "All processes successfully terminated."
        fi
    else
        if [ "$pid_still_running" -eq 0 ]; then
            color_echo blue "All tracked processes successfully terminated."
        fi
    fi

    return 0
}

# Cleanup function if running standalone
if [ "$START_MODE" != "CHILD" ]; then
    cleanup() {
        cleanup_pids "${PIDS[*]}" "flask|dramatiq|multiprocessing"
        exit 0
    }

    trap cleanup INT TERM HUP
fi

export CUDA_VISIBLE_DEVICES=${DEVICE_NB:-0}

venv/bin/flask --app app.main run --debug -p "${API_PORT:-5000}" &
FLASK_PID=$!
PIDS+=($FLASK_PID)

venv/bin/dramatiq app.main -t 1 -p 1 &
DRAMATIQ_PID=$!
PIDS+=($DRAMATIQ_PID)

color_echo cyan "Flask server PID       $FLASK_PID"
color_echo cyan "Dramatiq worker PID    $DRAMATIQ_PID"

# If running as standalone, wait for processes
if [ "$START_MODE" != "CHILD" ]; then
    color_echo magenta "Press Ctrl+C to stop all API processes"
    wait
else
    (tail -f /dev/null >/dev/null 2>&1) &
    TAIL_PID=$!
    PIDS+=($TAIL_PID)
    wait $TAIL_PID
fi
