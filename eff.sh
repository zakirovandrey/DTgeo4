#!/bin/bash
for nz in `seq 32 32 1024`; do grep -E "[0-9]* $1 $nz [0-9]*$" < $2 |sort -g | tail -n1; done
