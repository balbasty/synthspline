#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 50G -t 1-00:00:00 -c 32 -G 1 -o logs/vesselsynth.log python3 scripts/vessels_oct.py;
watch -n 0.1 "squeue | grep $USER"