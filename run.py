#!/usr/bin/python
# -*- coding: utf-8 -*-

#from ctypes import *
#mpi = CDLL('libmpi.so.0', RTLD_GLOBAL)

from math import *
import sys
import os
sys.path.append("./spacemodel")

from aivlib.vctr2 import *
from aivlib.vctr3 import *
from spacemodel import *

import DTgeo

GridNx = DTgeo.cvar.GridNx
GridNy = DTgeo.cvar.GridNy
GridNz = DTgeo.cvar.GridNz
dx=DTgeo.cvar.ds
dy=DTgeo.cvar.dv
dz=DTgeo.cvar.da
dt=DTgeo.cvar.dt

SM = SpaceModel()
#model = './spacemodel/Linevskaya_nVs_ext/0/'
model = './spacemodel/model-B/0/'

plasts = []
for f in os.listdir(model):
    if f.endswith('.plst'):
        plasts.append(GridPlast())
        plasts[-1].load(Ifile(model+f))
        plasts[-1].main_plast = True
        SM.add_plast(plasts[-1])

#model size:
Xmin=102500; Ymin=378600; Xmax=137700; Ymax=412900

P = GridPlast()
P.init( Indx(2, 2), # Nx, Ny
        Vctr(Xmin,Ymin), # offset xy
        Vctr(Xmax-Xmin, Ymax-Ymin)) # step 
P.set(Indx(0, 0), # cell 
      Vctr(+500, -50000), # z_t, z_b 
      Vctr(5200., 2300, 2.2),# Vp, Vs, sigma 
      Vctr(0., 0., 0.)) # gradient
P.set(Indx(1, 0), Vctr(+500, -50000), Vctr(5200., 2.3e3, 2.2), Vctr(0., 0., 0.))
P.set(Indx(0, 1), Vctr(-1500, -50000), Vctr(5200., 2.3e3, 2.2), Vctr(0., 0., 0.))
P.set(Indx(1, 1), Vctr(-1500, -50000), Vctr(5200., 2.3e3, 2.2), Vctr(0., 0., 0.))
P.main_plast = False
#SM.add_plast(P)

#P2 = GridPlast()
#P2.init( Indx(2, 2), Vctr(Xmin,Ymin), Vctr(Xmax-Xmin, Ymax-Ymin))
#P2.set(Indx(0, 0), Vctr(-500, -50000), Vctr(10.*5200., 2300., 1.), Vctr(0., 0., 0.))
#P2.set(Indx(0, 1), Vctr(-500, -50000), Vctr(10.*5200., 2.3e3, 1.), Vctr(0., 0., 0.))
#P2.set(Indx(1, 0), Vctr(-500, -50000), Vctr(10.*5200., 2.3e3, 1.), Vctr(0., 0., 0.))
#P2.set(Indx(1, 1), Vctr(-500, -50000), Vctr(10.*5200., 2.3e3, 1.), Vctr(0., 0., 0.))
#P2.main_plast = False

#SM.add_plast(P2)

print 'load OK'

SrcCoords_LOC  = [ GridNx/2*dx+0.5*dx, GridNy/2*dy+0.5*dy, 50.0]
SrcCoords_GLOB = [ (Xmax+Xmin)/2., (Ymax+Ymin)/2., 0 ]

boom = SM.get_par(SrcCoords_GLOB[0], SrcCoords_GLOB[1], SrcCoords_GLOB[2])
SM.Vp, SM.Vs, SM.sigma = boom.Vp, boom.Vs, boom.sigma
print "Phys_params at shotpoint %g %g %g\n"%(SM.Vp,SM.Vs,SM.sigma)

SS = DTgeo.cvar.shotpoint
SS.Ampl = 0.0;
SS.F0=0.03;
SS.gauss_waist=0.5;
SS.srcXs, SS.srcXv, SS.srcXa = SrcCoords_LOC[0],SrcCoords_LOC[1],SrcCoords_LOC[2];
SS.BoxMs, SS.BoxPs = SS.srcXs-4.1*dx, SS.srcXs+4.1*dx; 
SS.BoxMa, SS.BoxPa = SS.srcXa-4.1*dz, SS.srcXa+4.1*dz; 
SS.BoxMv, SS.BoxPv = SS.srcXv-5.1*dy, SS.srcXv+5.1*dy; 
SS.sphR = 50-2*dz; SS.BoxMs, SS.BoxPs = SS.srcXs-SS.sphR, SS.srcXs+SS.sphR;
boxDiagLength=sqrt((SS.BoxPs-SS.BoxMs)**2+(SS.BoxPa-SS.BoxMa)**2+(SS.BoxMv-SS.BoxPv)**2)
SS.tStop = boxDiagLength/2/min(SM.Vp,0.0001+SM.Vs)+8/(pi*SS.F0)+10*dt # 5000*dt; # ((BoxPs-BoxMs)+(BoxPa-BoxMa)+(BoxMv-BoxPv))/c+2*M_PI/Omega;
SS.V_max = 7.0;
SS.start = 0;

SS.set(SM.Vp, SM.Vs, SM.sigma)
MM = MiddleModel( SM, # исходная модель среды
                  Vctr(SrcCoords_GLOB[0], SrcCoords_GLOB[1], SrcCoords_GLOB[2]), # координаты ПВ в глобальной системе, [м]
                  0,                 # поворот вокруг ПВ, [рад]
                  Vctr(SrcCoords_LOC[0], SrcCoords_LOC[1]),         # координаты ПВ относительно левого нижнего угла счетной области, [м]
                  Vctr(dx/2,dy/2),   # размер полуячейки счетной области, [м]
                  Indx(GridNx*2+12,GridNy*2+16),   # размер счетной области по латерали в полуячейках 
                  Indx(12,16),         # размер ячейки текстуры, в полуячейках                            
                  dz*1.,                # интервал cглаживания границы слоев, [м]
                  -1e5                # нижняя граница модели, [м]
                 );

print 'Middle model initialization'
init_MM(MM)

DTgeo.cvar.Tsteps=5000
DTgeo._main(sys.argv)
