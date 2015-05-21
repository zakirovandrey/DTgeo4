#!/bin/bash
ratefile=rate750ti_pmlNz_Lvec.dat
for ny in `seq 3 1 6` `seq 7 5 1200`; do 
#  for nx in 6 60 120 600 1200 2400 4092; do 
  for nx in 80; do 
#    ntmax=$(($nx/3-3)); for nts in `seq 6 1 7`; do 
    ntmax=$(($nx/3-3)); for nts in `seq 6 1 6`; do 
#      nt=$((2**$nts)); #if [ $nt -gt $ntmax ]; then break; fi
      nt=50; #if [ $nt -gt $ntmax ]; then break; fi
      for nz in `seq 32 32 864`; do
        if [ $((nx*ny*nz*24)) -gt $((123*1024*1024*1024)) ]; then break; fi
        make clean; NV=$nz NS=$nx NA=$ny NTIME=$nt make -j4;
#        echo -n "$nx $ny $nz $nt " >> $ratefile;
        ./DTgeo --test |grep Step |cut -d"|" -f2 |cut -d" " -f2,3,4,5,6 |sort -g |tail -n1 >> $ratefile;
#        if ! [ -f DFmxw ]; then echo >> $ratefile; fi
      done; 
      echo -e "\n" >> $ratefile; 
    done;
  done;
done &> tmp.log
