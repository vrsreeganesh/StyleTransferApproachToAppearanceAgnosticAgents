#!/bin/bash

# Infinite loop
while true
do
    # Your command to be executed
    qstat -u vrs

    # Print empty lines
    echo -e "\n"

    # Pause for 5 seconds
    sleep 5
done
