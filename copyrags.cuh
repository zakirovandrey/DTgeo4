#define RAG_TRI(FLD)       rag->FLD[idom].trifld
#define RAG_TRI_PML(FLD,I) rag->FLD[idom].fldPML[I][pml_iz]
#define BUF_FLD_ONE(EVEN) chunk->pd##EVEN[ifld].trifld.one[iz]
#define BUF_FLD_TWO(EVEN) chunk->pd##EVEN[ifld].trifld.two[iz]
#define BUF_FLD_PML(EVEN,I) chunk->pd##EVEN[ifld].fldPML[I][pml_iz]
#define BUF_TO_RAG_ONE(EVEN,FLD1)   RAG_TRI(FLD1 ).one[iz] = BUF_FLD_ONE(EVEN); if(inPMLv)  RAG_TRI_PML(FLD1 ,0)=BUF_FLD_PML(EVEN ,0);
#define BUF_TO_RAG_TWO(EVEN,FLD23)  RAG_TRI(FLD23).two[iz] = BUF_FLD_TWO(EVEN); if(inPMLv) {RAG_TRI_PML(FLD23,1)=BUF_FLD_PML(EVEN ,1);\
                                                                                            RAG_TRI_PML(FLD23,2)=BUF_FLD_PML(EVEN ,2);}
#define RAG_TO_BUF_ONE(EVEN,FLD1)   BUF_FLD_ONE(EVEN) = RAG_TRI(FLD1 ).one[iz]; if(inPMLv)  BUF_FLD_PML(EVEN ,0)=RAG_TRI_PML(FLD1 ,0); 
#define RAG_TO_BUF_TWO(EVEN,FLD23)  BUF_FLD_TWO(EVEN) = RAG_TRI(FLD23).two[iz]; if(inPMLv) {BUF_FLD_PML(EVEN ,1)=RAG_TRI_PML(FLD23,1);\
                                                                                            BUF_FLD_PML(EVEN ,2)=RAG_TRI_PML(FLD23,2);}
#define BUF_TO_RAG_TRI_1(FLD1,FLD23) { BUF_TO_RAG_ONE(1,FLD1)         BUF_TO_RAG_TWO(1,FLD23) ifld++; }
#define BUF_TO_RAG_TRI_0(FLD1,FLD23) { BUF_TO_RAG_ONE(0,FLD1) ifld++; BUF_TO_RAG_TWO(0,FLD23)         }
#define RAG_TO_BUF_TRI_1(FLD1,FLD23) { RAG_TO_BUF_ONE(1,FLD1)         RAG_TO_BUF_TWO(1,FLD23) ifld++; }
#define RAG_TO_BUF_TRI_0(FLD1,FLD23) { RAG_TO_BUF_ONE(0,FLD1) ifld++; RAG_TO_BUF_TWO(0,FLD23)         }


#define BUF_TO_RAG_Si_0 {\
rag->Si[idom].duofld[0][iz].x  = chunk->fld[ifld][iz]; ifld++;\
rag->Si[idom].duofld[0][iz].y  = chunk->fld[ifld][iz]; ifld++;\
rag->Si[idom].duofld[1][iz].x  = chunk->fld[ifld][iz]; ifld++;\
rag->Si[idom].duofld[1][iz].y  = chunk->fld[ifld][iz]; ifld++;\
if(inPMLv) rag->Si[idom].fldPML[0][pml_iz] = chunk->fldPML[ifld_pml][pml_iz]; ifld_pml++;\
if(inPMLv) rag->Si[idom].fldPML[1][pml_iz] = chunk->fldPML[ifld_pml][pml_iz]; ifld_pml++;\
if(inPMLv) rag->Si[idom].fldPML[2][pml_iz] = chunk->fldPML[ifld_pml][pml_iz]; ifld_pml++;\
if(inPMLv) rag->Si[idom].fldPML[3][pml_iz] = chunk->fldPML[ifld_pml][pml_iz]; ifld_pml++; }
#define BUF_TO_RAG_Si_1 {\
rag->Si[idom].duofld[2][iz].x  = chunk->fld[ifld][iz]; ifld++;\
rag->Si[idom].duofld[2][iz].y  = chunk->fld[ifld][iz]; ifld++;\
if(inPMLv) rag->Si[idom].fldPML[4][pml_iz] = chunk->fldPML[ifld_pml][pml_iz]; ifld_pml++;}

#define BUF_TO_RAG_Vi_0 { \
rag->Vi[idom].trifld.one[iz]   = chunk->fld[ifld][iz]; ifld++;\
if(inPMLv) rag->Vi[idom].fldPML[0][pml_iz] = chunk->fldPML[ifld_pml][pml_iz]; ifld_pml++;}
#define BUF_TO_RAG_Vi_1 { \
rag->Vi[idom].trifld.two[iz].x = chunk->fld[ifld][iz]; ifld++;\
rag->Vi[idom].trifld.two[iz].y = chunk->fld[ifld][iz]; ifld++;\
if(inPMLv) rag->Vi[idom].fldPML[1][pml_iz] = chunk->fldPML[ifld_pml][pml_iz]; ifld_pml++; \
if(inPMLv) rag->Vi[idom].fldPML[2][pml_iz] = chunk->fldPML[ifld_pml][pml_iz]; ifld_pml++;}

#define RAG_TO_BUF_Si_0 {\
chunk->fld[ifld][iz] = rag->Si[idom].duofld[0][iz].x  ; ifld++;\
chunk->fld[ifld][iz] = rag->Si[idom].duofld[0][iz].y  ; ifld++;\
chunk->fld[ifld][iz] = rag->Si[idom].duofld[1][iz].x  ; ifld++;\
chunk->fld[ifld][iz] = rag->Si[idom].duofld[1][iz].y  ; ifld++;\
if(inPMLv) chunk->fldPML[ifld_pml][pml_iz] = rag->Si[idom].fldPML[0][pml_iz]; ifld_pml++;\
if(inPMLv) chunk->fldPML[ifld_pml][pml_iz] = rag->Si[idom].fldPML[1][pml_iz]; ifld_pml++;\
if(inPMLv) chunk->fldPML[ifld_pml][pml_iz] = rag->Si[idom].fldPML[2][pml_iz]; ifld_pml++;\
if(inPMLv) chunk->fldPML[ifld_pml][pml_iz] = rag->Si[idom].fldPML[3][pml_iz]; ifld_pml++; }
#define RAG_TO_BUF_Si_1 {\
chunk->fld[ifld][iz] = rag->Si[idom].duofld[2][iz].x ; ifld++;\
chunk->fld[ifld][iz] = rag->Si[idom].duofld[2][iz].y ; ifld++;\
if(inPMLv) chunk->fldPML[ifld_pml][pml_iz] = rag->Si[idom].fldPML[4][pml_iz]; ifld_pml++;}

#define RAG_TO_BUF_Vi_0 { \
chunk->fld[ifld][iz] = rag->Vi[idom].trifld.one[iz] ; ifld++;\
if(inPMLv) chunk->fldPML[ifld_pml][pml_iz] = rag->Vi[idom].fldPML[0][pml_iz]; ifld_pml++;}
#define RAG_TO_BUF_Vi_1 { \
chunk->fld[ifld][iz] = rag->Vi[idom].trifld.two[iz].x; ifld++;\
chunk->fld[ifld][iz] = rag->Vi[idom].trifld.two[iz].y; ifld++;\
if(inPMLv) chunk->fldPML[ifld_pml][pml_iz] = rag->Vi[idom].fldPML[1][pml_iz]; ifld_pml++; \
if(inPMLv) chunk->fldPML[ifld_pml][pml_iz] = rag->Vi[idom].fldPML[2][pml_iz]; ifld_pml++;}

template<const int even> __device__ inline void load_buffer(DiamondRag* rag0, halfRag* buffer, int ix, const int x0buf, const int xNbuf, const int idev, const int iz, const int pml_iz, const bool inPMLv){
  const int StepY=NStripe(idev);
  for(int xbuf=x0buf; xbuf<xNbuf; xbuf++, ix=(ix+1)%Ns) {
    const int ixm = (ix-1+Ns)%Ns;
    const int ixp = (ix+1   )%Ns;
    halfRag* chunk = buffer+xbuf;
    int idom; int ifld=0, ifld_pml=0;
    if(even==0) {
      DiamondRag* rag = rag0+ixp*StepY;
      for(idom=0; idom<NDT*NDT/2; idom++) { BUF_TO_RAG_Si_0; BUF_TO_RAG_Si_1; }
                                            BUF_TO_RAG_Si_0; BUF_TO_RAG_Vi_1;
      for(idom++; idom<NDT*NDT  ; idom++) { BUF_TO_RAG_Vi_0; BUF_TO_RAG_Vi_1; }
    }
    if(even==1) {
      DiamondRag* rag = rag0+ix*StepY;
      idom=NDT*NDT/2;                       BUF_TO_RAG_Si_1;
      for(idom++; idom<NDT*NDT  ; idom++) { BUF_TO_RAG_Si_0; BUF_TO_RAG_Si_1; }
      rag = rag0+ixp*StepY;
      for(idom=0; idom<NDT*NDT/2; idom++) { BUF_TO_RAG_Vi_0; BUF_TO_RAG_Vi_1; }
                                            BUF_TO_RAG_Vi_0;
    }
  }
}
template<const int even> __device__ inline void save_buffer(DiamondRag* rag0, halfRag* buffer, int ix, const int x0buf, const int xNbuf, const int idev, const int iz, const int pml_iz, const bool inPMLv){
  const int StepY=NStripe(idev);
  for(int xbuf=x0buf; xbuf<xNbuf; xbuf++, ix=(ix+1)%Ns) {
    const int ixm = (ix-1+Ns)%Ns;
    const int ixp = (ix+1   )%Ns;
    halfRag* chunk = buffer+xbuf;
    int ifld=0, ifld_pml=0; int idom;
    if(even==0) {
      DiamondRag* rag = rag0+ix*StepY;
      idom=NDT*NDT/2;                       RAG_TO_BUF_Si_1;
      for(idom++; idom<NDT*NDT  ; idom++) { RAG_TO_BUF_Si_0; RAG_TO_BUF_Si_1; }
      rag = rag0+ixp*StepY;                                                    
      for(idom=0; idom<NDT*NDT/2; idom++) { RAG_TO_BUF_Vi_0; RAG_TO_BUF_Vi_1; }
                                            RAG_TO_BUF_Vi_0;                   
    }
    if(even==1) {
      DiamondRag* rag = rag0+ix*StepY;
      for(idom=0; idom<NDT*NDT/2; idom++) { RAG_TO_BUF_Si_0; RAG_TO_BUF_Si_1; }
                                            RAG_TO_BUF_Si_0; RAG_TO_BUF_Vi_1;
      for(idom++; idom<NDT*NDT  ; idom++) { RAG_TO_BUF_Vi_0; RAG_TO_BUF_Vi_1; }
    }
  }
}
__device__ inline int get_dev(const int iy, int& ym) {
  int idev=0; ym=0;
  while(iy>=ym && idev<NDev) { ym+=NStripe(idev); idev++; }
  ym-=NStripe(idev-1);
  return idev-1;
}
template<const int even> __global__ void bufsave (int ix, int y0, int Nt, int t0) {
  const int iy=y0;
  const int iz=threadIdx.x+blockIdx.x*blockDim.x; const int pml_iz=iz;
  const bool inPMLv = (iz<Npmlz);
  if(iz<0 || iz>=Nv) return;
  int ymC=0;
  const int curDev=get_dev(iy, ymC); 
  DiamondRag      * __restrict__ RAG0       = &pars.rags[curDev][iy  -ymC];
  int xstart=ix;
  const bool isTopStripe = (curDev==NDev-1 && pars.subnode==NasyncNodes-1);
  const bool isBotStripe = (curDev==0      && pars.subnode==0            );

  if(iy==ymC+NStripe(curDev)-1 && !isTopStripe) if(even==0) save_buffer<0>(RAG0, pars.p2pBufP[curDev], xstart, t0, Nt, curDev, iz,pml_iz,inPMLv);
  if(iy==ymC                   && !isBotStripe) if(even==1) save_buffer<1>(RAG0, pars.p2pBufM[curDev], xstart, t0, Nt, curDev, iz,pml_iz,inPMLv);
}
