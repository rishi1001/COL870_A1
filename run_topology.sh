#!/bin/bash

echo "Running the script"

echo "Running the MLP"
python run_experiment.py --dataset "Cora" --k 3 --model_type "linear"

# for loop 
echo "Running the GNN classifier for multiple k"
for i in 1 2 3 4 5
do
    echo "Value of k = $i"
    python run_experiment.py --dataset "Cora" --k $i --model_type "gnn"
done
