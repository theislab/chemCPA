#!/bin/bash

# Name of the tmux session
SESSION_NAME="chemcpa_training"

# Path to your Python script
SCRIPT_PATH="chemCPA/train_hydra.py"

# Check if the session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Attaching to it."
    tmux attach-session -t $SESSION_NAME
else
    # Create a new session
    tmux new-session -d -s $SESSION_NAME

    # Rename the first window
    tmux rename-window -t $SESSION_NAME:0 'training'

    # Send the command to run the Python script
    tmux send-keys -t $SESSION_NAME:0 "python $SCRIPT_PATH" C-m

    # Attach to the session
    tmux attach-session -t $SESSION_NAME
fi