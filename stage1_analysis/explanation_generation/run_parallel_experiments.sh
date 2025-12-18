#!/bin/bash

# Parallel Experiment Runner
# This script runs all 6 experiment settings in parallel across different terminals/processes
# Each setting will use ALL available models

echo "Starting parallel experiments..."
echo "Each setting will run ALL models on the full dataset"
echo "============================================"

# Run each setting in the background
python run_experiments.py --setting none &
PID1=$!
echo "Started: none (PID: $PID1)"

python run_experiments.py --setting none_description &
PID2=$!
echo "Started: none_description (PID: $PID2)"

python run_experiments.py --setting unpaired_properties &
PID3=$!
echo "Started: unpaired_properties (PID: $PID3)"

python run_experiments.py --setting unpaired_properties_description &
PID4=$!
echo "Started: unpaired_properties_description (PID: $PID4)"

python run_experiments.py --setting paired_properties &
PID5=$!
echo "Started: paired_properties (PID: $PID5)"

python run_experiments.py --setting paired_properties_description &
PID6=$!
echo "Started: paired_properties_description (PID: $PID6)"

echo "============================================"
echo "All experiments started!"
echo "Waiting for all to complete..."

# Wait for all background jobs to complete
wait $PID1
echo "✅ Completed: none"

wait $PID2
echo "✅ Completed: none_description"

wait $PID3
echo "✅ Completed: unpaired_properties"

wait $PID4
echo "✅ Completed: unpaired_properties_description"

wait $PID5
echo "✅ Completed: paired_properties"

wait $PID6
echo "✅ Completed: paired_properties_description"

echo "============================================"
echo "🎉 All experiments completed!"
echo "Results saved in: checkpoints/explanation_generation/"

