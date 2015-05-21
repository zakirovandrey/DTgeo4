#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import *
import sys
import os

import DTgeo
GridNx = DTgeo.cvar.GridNx
GridNy = DTgeo.cvar.GridNy
GridNz = DTgeo.cvar.GridNz
dx=DTgeo.cvar.ds
dy=DTgeo.cvar.dv
dz=DTgeo.cvar.da
dt=DTgeo.cvar.dt

SrcCoords_LOC  = [ GridNx/2*dx, GridNy/2*dy, 0]

class SM: pass
SM.Vp=3.0; SM.Vs=1.5; SM.sigma=1.0

SS = DTgeo.cvar.shotpoint
SS.F0=0.03;
SS.gauss_waist=0.2;
SS.srcXs, SS.srcXv, SS.srcXa = SrcCoords_LOC[0],SrcCoords_LOC[1],SrcCoords_LOC[2];
SS.BoxMs, SS.BoxPs = SS.srcXs-4.1*dx, SS.srcXs+4.1*dx; 
SS.BoxMa, SS.BoxPa = SS.srcXa-5.1*dz, SS.srcXa+5.1*dz; 
SS.BoxMv, SS.BoxPv = SS.srcXv-5.1*dy, SS.srcXv+5.1*dy; 
boxDiagLength=sqrt((SS.BoxPs-SS.BoxMs)**2+(SS.BoxPa-SS.BoxMa)**2+(SS.BoxMv-SS.BoxPv)**2)
SS.tStop = 0#boxDiagLength/2/min(SM.Vp,0.0001+SM.Vs)+8/(pi*SS.F0)+10*dt # 5000*dt; # ((BoxPs-BoxMs)+(BoxPa-BoxMa)+(BoxMv-BoxPv))/c+2*M_PI/Omega;
SS.V_max = 7.0;

SS.set(SM.Vp, SM.Vs, SM.sigma)

DTgeo.cvar.Tsteps=20000
DTgeo._main(sys.argv)
