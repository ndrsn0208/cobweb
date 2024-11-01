#!/bin/bash

# Loop from 1 to 20
for i in {1..50}
do
    # Run the Python script with the current value of i as an argument
    python example-tabular-fast.py $i 0
done