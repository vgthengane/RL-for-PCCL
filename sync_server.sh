#!/bin/bash


# ============================================================
# Sync Server Script
# ------------------------------------------------------------
# Purpose:
#   This script continuously synchronizes the project directory
#   between the local research storage and the fast scratch
#   storage used for running experiments.
#
# Sync behaviour:
#   1) LOCAL  → REMOTE : Sync source code and project files
#   2) REMOTE → LOCAL  : Sync experiment outputs only
#
# Typical workflow:
#   - Code editing happens in the LOCAL project directory
#   - Training and heavy computation run from the REMOTE path
#     on fast storage
#   - Experiment outputs are written to `_experiments/`
#   - Those experiment results are synced back automatically
#
# IMPORTANT:
#   This script must be executed from the **datamove server**
#   because both LOCAL and REMOTE paths are mounted there.
#
#   Example:
#       ssh vt00262@datamove1.surrey.ac.uk
#       bash /vol/research/Vishal_Thengane/projects/RL-for-PCCL/sync_server.sh
#
# Behaviour:
#   - Automatically daemonizes itself using `nohup`
#   - Writes logs to:  sync_server.log
#   - Prevents multiple instances using a file lock
#   - Runs sync every 10 seconds
#
# Author notes:
#   Designed for ML research workflow where:
#     - code changes frequently
#     - experiments run on faster storage
#     - experiment results must sync back automatically
# ============================================================


LOCAL="/vol/research/Vishal_Thengane/projects/RL-for-PCCL/"
REMOTE="/mnt/fast/nobackup/users/vt00262/projects/RL-for-PCCL/"

# Determine script directory and fixed log file location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/sync_server.log"


# ============================================================
# Automatic background execution (daemon mode)
# ------------------------------------------------------------
# You normally do NOT need to run this script with `nohup`
# manually. Simply execute:
#
#   bash sync_server.sh
#
# The script will automatically restart itself using:
#
#   nohup bash sync_server.sh > sync_server.log 2>&1 &
#
# and continue running in the background.
#
# Log file location:
#   /vol/research/Vishal_Thengane/projects/RL-for-PCCL/sync_server.log
#
# Useful commands
# ------------------------------------------------------------
# Check if the sync daemon is running:
#   pgrep -af sync_server.sh
#
# Monitor sync activity:
#   tail -f /vol/research/Vishal_Thengane/projects/RL-for-PCCL/sync_server.log
#
# Check currently running rsync processes (optional):
#   pgrep -af rsync
# ============================================================

if [ -z "$SYNC_SERVER_DAEMON" ]; then
    export SYNC_SERVER_DAEMON=1
    echo "Starting sync_server.sh in background (nohup)..."

    # Do NOT truncate the log file here. Truncating while another
    # process still holds the file descriptor can produce NULL
    # byte (\0000) in the log. Instead we simply append to it.
    nohup bash "$0" "$@" >> "$LOG_FILE" 2>&1 &

    echo "Background sync started. Log: $LOG_FILE"
    exit 0
fi

# ============================================================
# Prevent multiple sync scripts running simultaneously
# Using flock (safer than PID detection)
# ============================================================

LOCK_FILE="$SCRIPT_DIR/sync_server.lock"

exec 200>"$LOCK_FILE"

# Try to acquire lock
if ! flock -n 200; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another sync_server.sh instance is already running."
  echo "Exiting."
  exit 0
fi


# ============================================================
# Main Sync Loop
# ============================================================

while true; do

  # --------------------------------------------------------
  # Sync code from LOCAL → SERVER
  # --------------------------------------------------------
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync LOCAL → SERVER (code)"

  rsync -avzu --delete \
    --exclude "./_experiments" \
    --exclude "sync_server.lock" \
    --exclude "sync_server.log" \
    --exclude "sync_server.sh" \
    --exclude "*nohup.out" \
    --exclude "*_archive" \
    "$LOCAL" "$REMOTE"


  # --------------------------------------------------------
  # Sync experiment outputs from SERVER → LOCAL
  # --------------------------------------------------------
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync SERVER → LOCAL (experiments)"

  rsync -avzu \
  "$REMOTE/_experiments/" "$LOCAL/_experiments/"


  # --------------------------------------------------------
  # Sleep interval
  # --------------------------------------------------------
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sleeping 10 seconds..."
  sleep 10

done
