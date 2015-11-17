#!/usr/bin/python
import sys
import itertools
import genAlg as gen

gen.order=4

defout = sys.stdout

gen.PMLS=0
gen.AsyncType='D'
#gen.make_DTorre(typus=0, vpml=0, atype='D', spml=0, LargeNV=1); exit(0)
#for Yt in itertools.product("DSB", repeat=3):
#for Yt in ('D','S','DS','SD','Ic','Ib','Xc','Xb','DD','TFSFts','TFSFst'):
for Yt in ('B','S','D','I','X','TFSF','ITFSF'):
#for Yt in ('I','ITFSF'):
  for spml in 0,1:
    gen.PMLS=spml
    fl = open("ker%s%s.inc.cu"%(Yt,("","_pmls")[spml]), 'w')
    print >>fl, '#include "params.h"'
    if Yt[-4:]=='TFSF' or Yt[0]=="I" or Yt[0]=="B": print >>fl, '#include "signal.h"'
    for Dcase in 0,1:
      print >>fl, '__global__ void __launch_bounds__(Nz) %storre%s%d (int ix, int y0, int Nt, int t0) {'%(("","PMLS")[spml],Yt,Dcase)
      print >>fl, '  REG_DEC(%d)'%Dcase
      if Yt[-4:]=='TFSF': print >>fl, '  #include "%s%d.inc.cu"   \n}'%(Yt,Dcase)
      else         : print >>fl, '  #include "%s%d%s.inc.cu"\n}'%(Yt,Dcase,("","_pmls")[spml])
      #if Yt=='TFSF': print >>fl, '  if(inPMLv){\n    #include "D%d_pmlv.inc.cu"   \n  }else{\n    #include "TFSF%d.inc.cu"\n  }\n}'%(Dcase,Dcase)
      #else         : print >>fl, '  if(inPMLv){\n    #include "%s%d%s_pmlv.inc.cu"\n  }else{\n    #include "%s%d%s.inc.cu"\n  }\n}'%(Yt,Dcase,("","_pmls")[spml],Yt,Dcase,("","_pmls")[spml])
    fl.close()
    if Yt[-4:]=='TFSF' and spml==1: continue
    for zpml in 0,:
      for Dcase in 0,1:
        LargeNV=0
        fname = "%s%d%s%s.inc.cu"%(''.join(Yt), Dcase, ("","_pmls")[spml], ("","_pmlv")[zpml])
        defout.write("generating %s\n"%fname)
        Yt=''.join(Yt); gen.AsyncType=Yt
        sys.stdout = open(fname, 'w')

        gen.make_DTorre(typus=Dcase, vpml=zpml, atype=Yt, spml=spml, LargeNV=LargeNV)
        sys.stdout.flush()
        sys.stdout.close()
