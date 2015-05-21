#!/bin/bash
for ny in `seq 22 3 1200`; do grep -E "[0-9]* $ny $1 [0-9]*$" < $2 |sort -g | tail -n1; done
