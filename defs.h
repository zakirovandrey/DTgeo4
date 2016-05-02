#ifndef DEFS_H
#define DEFS_H
const ftype p9d8 = 9./8.;
const ftype m1d24=-1./24.;
const ftype p27= 27.;
const ftype dtdxd24 = dt/dx/24.;
const ftype dtdyd24 = dt/dy/24.;
const ftype dtdzd24 = dt/dz/24.;
#if SquareGrid==1
const ftype dtdrd24 = dt/dr/24.;
#else
const ftype dtdrd24 = 1.0;
#endif
const ftype drdt24 = dr/dt*24.;

extern ftype* __restrict__ hostKpmlx1; extern ftype* __restrict__ hostKpmlx2;
extern ftype* __restrict__ hostKpmly1; extern ftype* __restrict__ hostKpmly2;
extern ftype* __restrict__ hostKpmlz1; extern ftype* __restrict__ hostKpmlz2;
extern __constant__ ftype Kpmlx1[(KNpmlx==0)?1:KNpmlx];
extern __constant__ ftype Kpmlx2[(KNpmlx==0)?1:KNpmlx];
extern __constant__ ftype Kpmly1[(KNpmly==0)?1:KNpmly];
extern __constant__ ftype Kpmly2[(KNpmly==0)?1:KNpmly];
extern __constant__ ftype Kpmlz1[(KNpmlz==0)?1:KNpmlz];
extern __constant__ ftype Kpmlz2[(KNpmlz==0)?1:KNpmlz];

template<class Ph, class Pd> static void copy2dev(Ph &hostP, Pd &devP) {
  for(int i=0; i<NDev; i++) {
    CHECK_ERROR( cudaSetDevice(i) );
    CHECK_ERROR( cudaMemcpyToSymbol(devP, &hostP, sizeof(Pd)) );
  }
  CHECK_ERROR( cudaSetDevice(0) );
}
//__device__ __forceinline__ static bool isOutS(const int x) { return (x<0+NDT || x>=Ns*2*NDT-2*NDT); }
__device__ __forceinline__ static bool isOutS(const int x) { return (x<=0+NDT || x>=Ns*2*NDT-NDT); }
//__device__ __forceinline__ static bool isOutS(const int x) { return false; }
#ifdef NOPMLS
__device__ __forceinline__ static bool inPMLsync(const int x) { return 0; }
#else
__device__ __forceinline__ static bool inPMLsync(const int x) { return (x<Npmlx/2*2*NDT-NDT && x>=0 || x<Ns*2*NDT && x>=Ns*2*NDT-Npmlx/2*2*NDT+NDT); }
#endif

#ifdef USE_TEX_REFS
#define TEX_MODEL_TYPE 0
#endif

//#define GLOBAL(x) ( (x+2*NDT*pars.GPUx0+2*NDT*NS-2*NDT*pars.wleft)%(2*NDT*NS)+pars.wleft*2*NDT )
//#define GLOBAL(x) ( (x+2*NDT*(NS+ix+pars.GPUx0-pars.wleft))%(2*NDT*NS)+2*NDT*pars.wleft )
#define GLOBAL(x) ( x+2*NDT*glob_ix )
//#define TEXCOFFV(xt,y,z,I,h) if(threadIdx.x==0 && blockIdx.x==0) printf("get tex value x=%g\n", GLOBAL(xt)*texStretch[I]); 
//#define TEXCOFFT(xt,y,z,I,h) if(threadIdx.x==0 && blockIdx.x==0) printf("get tex value x=%g\n", GLOBAL(xt)*texStretch[I]); 
//#define TEXCOFFS(xt,y,z,I,h) if(threadIdx.x==0 && blockIdx.x==0) printf("get tex value x=%g\n", GLOBAL(xt)*texStretch[I]);
#if TEX_MODEL_TYPE==1
#define TEXCOFFV(nind,xt,yt,z,I,h) ArrcoffV[nind] = tex3D<float >(pars.texs.layerV[curDev][I], h*texStretchH,(z)*texStretchY,  GLOBAL(xt)*texStretch[I]); /*if(threadIdx.x==Nz/2) printf("coffV(%d %d %d)=%g\n", xt,z,h, coffV);*/
#define TEXCOFFT(nind,xt,yt,z,I,h) ArrcoffT[nind] = tex3D<float >(pars.texs.layerT[curDev][I], h*texStretchH,(z)*texStretchY,  GLOBAL(xt)*texStretch[I]); /*if(threadIdx.x==Nz/2) printf("coffT(%d %d %d)=%g\n", xt,z,h, coffT);*/
#define TEXCOFFS(nind,xt,yt,z,I,h) ArrcoffS[nind] = tex3D<float2>(pars.texs.layerS[curDev][I], h*texStretchH,(z)*texStretchY,  GLOBAL(xt)*texStretch[I]); /*if(threadIdx.x==Nz/2) printf("coffS(%d %d %d)=%g,%g\n", xt,z,h, coffS.x, coffS.y);*/
#elif TEX_MODEL_TYPE==2
#define TEXCOFFV(nind,xt,yt,z,I,h) ArrcoffV[nind] = tex3D<float >(pars.texs.TexlayerV[curDev], h*texStretchH, (z)*texStretchY, GLOBAL(xt)*texStretch[0]); /*if(threadIdx.x==Nz/2) printf("coffV(%d %d %d)=%g\n", xt,z,h, coffV);*/
#define TEXCOFFT(nind,xt,yt,z,I,h) ArrcoffT[nind] = tex3D<float >(pars.texs.TexlayerT[curDev], h*texStretchH, (z)*texStretchY, GLOBAL(xt)*texStretch[0]); /*if(threadIdx.x==Nz/2) printf("coffT(%d %d %d)=%g\n", xt,z,h, coffT);*/
#define TEXCOFFS(nind,xt,yt,z,I,h) ArrcoffS[nind] = tex3D<float2>(pars.texs.TexlayerS[curDev], h*texStretchH, (z)*texStretchY, GLOBAL(xt)*texStretch[0]); /*if(threadIdx.x==Nz/2) printf("coffS(%d %d %d)=%g,%g\n", xt,z,h, coffS.x, coffS.y);*/
#endif //TEX_MODEL_TYPE
#ifdef USE_TEX_REFS

#ifdef CUDA_TEX_INTERP
#define GET_TEX_INTERP(text,z,h,xt) tex3D(text, (z)*texStretch[0].y+texShift[0].y, h*texStretchH, GLOBAL(xt)*texStretch[0].x+texShift[0].x)
#else //CUDA_TEX_INTERP not def
#define GET_TEX_INTERP(text,z,h,xt) \
  tex3D(text, (z)*texStretch[0].y+texShift[0].y, afloor+0.5f, bfloor+0.5f)*(1.0f-alpha)*(1.0f-beta)+\
  tex3D(text, (z)*texStretch[0].y+texShift[0].y, afloor+1.5f, bfloor+0.5f)*alpha*(1.0f-beta)+\
  tex3D(text, (z)*texStretch[0].y+texShift[0].y, afloor+0.5f, bceil +0.5f)*(1.0f-alpha)*beta+\
  tex3D(text, (z)*texStretch[0].y+texShift[0].y, afloor+1.5f, bceil +0.5f)*alpha*beta
/*  tex3D(text, (z)*texStretch[0].y+texShift[0].y, int(h*texStretchH-0.5f)+0.5f, int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)+0.5f)*\
  (1.0f-(h*texStretchH-0.5f-int(h*texStretchH-0.5f)))*(1.0f-(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f-int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)))+\
  tex3D(text, (z)*texStretch[0].y+texShift[0].y, int(h*texStretchH-0.5f)+1.5f, int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)+0.5f)*\
  (     (h*texStretchH-0.5f-int(h*texStretchH-0.5f)))*(1.0f-(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f-int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)))+\
  tex3D(text, (z)*texStretch[0].y+texShift[0].y, int(h*texStretchH-0.5f)+0.5f, int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)+1.5f)*\
  (1.0f-(h*texStretchH-0.5f-int(h*texStretchH-0.5f)))*(     (GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f-int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)))+\
  tex3D(text, (z)*texStretch[0].y+texShift[0].y, int(h*texStretchH-0.5f)+1.5f, int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)+1.5f)*\
  (     (h*texStretchH-0.5f-int(h*texStretchH-0.5f)))*(     (GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f-int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)))*/
#endif //CUDA_TEX_INTERP

// tex3D(text, (z)*texStretch[0].y+texShift[0].y, h*texStretchH, GLOBAL(xt)*texStretch[0].x+texShift[0].x)

//tex3D(layerRefS, (z)*texStretch[0].y+texShift[0].y, h*texStretchH, int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)+0.5f)*(1.0f-(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f-int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)))+tex3D(layerRefS, (z)*texStretch[0].y+texShift[0].y, h*texStretchH, int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f)+1.5f)*(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f-int(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f));

#ifdef CUDA_TEX_INTERP
#define CALC_A_B(h,xt) ;
#else 
#define CALC_A_B(h,xt) \
afloor=floorf(h*texStretchH-0.5f); alpha = h*texStretchH-0.5f-afloor; \
bfloor=floorf(GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f); beta = GLOBAL(xt)*texStretch[0].x+texShift[0].x-0.5f-bfloor; \
bfloor = int(bfloor)%texNwindow; bceil = int(bfloor+1)%texNwindow;
#endif //CUDA_TEX_INTERP
#define TEXCOFFS(nind,xt,yt,z,I,h)  CALC_A_B(h,xt);\
 ArrcoffS[nind] = GET_TEX_INTERP(layerRefS, z,h,xt); 
//if(threadIdx.x==0 && blockIdx.x==0) printf("S at %d using texture X-coord %g and %g /////%d and %d// alpha=%g beta=%g h=%d,iy=%d ArrcoffS=%g %g\n", GLOBAL(xt), bfloor, bceil, int(bfloor)%5, int(bfloor+1)%5,alpha,beta,h,iy, ArrcoffS[nind].x,ArrcoffS[nind].y );
#define TEXCOFFV(nind,xt,yt,z,I,h)  CALC_A_B(h,xt);\
 ArrcoffV[nind] = GET_TEX_INTERP(layerRefV, z,h,xt); 
//if(threadIdx.x==0 && blockIdx.x==0) printf("V at %d using texture X-coord %g and %g /////%d and %d// alpha=%g beta=%g\n", GLOBAL(xt), bfloor, bceil, int(bfloor)%5, int(bfloor+1)%5,alpha,beta);
#ifndef ANISO_TR
#define TEXCOFFTx(nind,xt,yt,z,I,h) CALC_A_B(h,xt); \
ArrcoffT[nind] = GET_TEX_INTERP(layerRefT, z,h,xt);
//if(threadIdx.x==0 && blockIdx.x==0) printf("T at %d using texture X-coord %g and %g /////%d and %d// alpha=%g beta=%g\n", GLOBAL(xt), bfloor, bceil, int(bfloor)%5, int(bfloor+1)%5,alpha,beta);
#define TEXCOFFTy(nind,xt,yt,z,I,h) TEXCOFFTx(nind,xt,yt,z,I,h)
#define TEXCOFFTz(nind,xt,yt,z,I,h) TEXCOFFTx(nind,xt,yt,z,I,h)
#elif ANISO_TR==1
#define TEXCOFFTx(nind,xt,yt,z,I,h) CALC_A_B(h,xt); ArrcoffT[nind] = GET_TEX_INTERP(layerRefTa, z,h,xt);
#define TEXCOFFTy(nind,xt,yt,z,I,h) CALC_A_B(h,xt); ArrcoffT[nind] = GET_TEX_INTERP(layerRefTi, z,h,xt);
#define TEXCOFFTz(nind,xt,yt,z,I,h) CALC_A_B(h,xt); ArrcoffT[nind] = GET_TEX_INTERP(layerRefTi, z,h,xt);
#elif ANISO_TR==2
#define TEXCOFFTx(nind,xt,yt,z,I,h) CALC_A_B(h,xt); ArrcoffT[nind] = GET_TEX_INTERP(layerRefTi, z,h,xt);
#define TEXCOFFTy(nind,xt,yt,z,I,h) CALC_A_B(h,xt); ArrcoffT[nind] = GET_TEX_INTERP(layerRefTa, z,h,xt);
#define TEXCOFFTz(nind,xt,yt,z,I,h) CALC_A_B(h,xt); ArrcoffT[nind] = GET_TEX_INTERP(layerRefTi, z,h,xt);
#elif ANISO_TR==3
#define TEXCOFFTx(nind,xt,yt,z,I,h) CALC_A_B(h,xt); ArrcoffT[nind] = GET_TEX_INTERP(layerRefTi, z,h,xt);
#define TEXCOFFTy(nind,xt,yt,z,I,h) CALC_A_B(h,xt); ArrcoffT[nind] = GET_TEX_INTERP(layerRefTi, z,h,xt);
#define TEXCOFFTz(nind,xt,yt,z,I,h) CALC_A_B(h,xt); ArrcoffT[nind] = GET_TEX_INTERP(layerRefTa, z,h,xt);
#endif//ANISO_TR
#endif//USE_TEX_REFS
//#define TEXCOFFV(xt,yt,z,I,h) coffV = tex3D<float >(pars.texs.TexlayerV[curDev], h*texStretchH, (z)*texStretchY, GLOBAL(xt)*texStretch[0]); /*if(threadIdx.x==Nz/2) printf("coffV(%d %d %d)=%g\n", xt,z,h, coffV);*/
//#define TEXCOFFT(xt,yt,z,I,h) coffT = tex3D<float >(pars.texs.TexlayerT[curDev], h*texStretchH, (z)*texStretchY, GLOBAL(xt)*texStretch[0]); /*if(threadIdx.x==Nz/2) printf("coffT(%d %d %d)=%g\n", xt,z,h, coffT);*/
//#define TEXCOFFS(xt,yt,z,I,h) coffS = tex3D<float2>(pars.texs.TexlayerS[curDev], h*texStretchH, (z)*texStretchY, GLOBAL(xt)*texStretch[0]); /*if(threadIdx.x==Nz/2) printf("coffS(%d %d %d)=%g,%g\n", xt,z,h, coffS.x, coffS.y);*/
#ifdef COFFS_DEFAULT
#define TEXCOFFV(nind,xt,yt,z,I,h) ArrcoffV[nind] = coffV;
#define TEXCOFFT(nind,xt,yt,z,I,h) ArrcoffT[nind] = coffT;
#define TEXCOFFS(nind,xt,yt,z,I,h) ArrcoffS[nind] = coffS;
#endif//COFFS_DEFAULT
__device__ inline int get_iz(const int nth) {
  int iz=nth;
  if(nth<Npmlz && nth>=Npmlz/2) iz+=Nv-Npmlz;
  else if(nth>=Npmlz) iz-=Npmlz/2;
  return iz;
}
__host__ __device__ inline int get_pml_iy(const int iy) {
  const int diy=(iy+Na)%Na;
  return (diy<Npmly/2)?diy:(diy-Na+Npmly);
  //const int pmliy=iy-Na+Npmly;
  //return pmliy;
}
__device__ inline int get_pml_ix(const int ix) {
  if (ix<Npmlx/2*2*NDT) return (ix+KNpmlx)%KNpmlx;
  else return (ix-Ns*2*NDT+Npmlx*2*NDT)%KNpmlx;
}

__device__ inline int get_idev(const int iy, int& ym) {
  int idev=0; ym=0;
  while(iy>=ym && idev<NDev) { ym+=NStripe(idev); idev++; }
  ym-=NStripe(idev-1);
  return idev-1;
}
__device__ inline int Vrefl(const int iz, const int incell=0) {
  if(iz>=Nv) return Nv+Nv-iz-1-incell;
  if(iz< 0 ) return -iz-incell;
  return iz;
}

struct __align__(16) ftype8 { ftype4 u, v; };
//extern __shared__ ftypr2 shared_fld[2][7][Nv];
//extern __shared__ ftype2 shared_fld[(FTYPESIZE*Nv*28>0xc000)?7:14][Nv];
extern __shared__ ftype2 shared_fld[SHARED_SIZE][NzMax];
#ifdef DROP_ONLY_V
#define DEC_CHUNKS_0 const int chunkSi[]={ 0,2}, chunkTx[]={0,0}, chunkTy[]={0,0}, chunkTz[]={0,0}, chunkVx[]={0,0}, chunkVy[]={0,0}, chunkVz[]={0,0};
#define DEC_CHUNKS_1 const int chunkSi[]={-1,0}, chunkTx[]={0,0}, chunkTy[]={0,0}, chunkTz[]={0,0}, chunkVx[]={0,0}, chunkVy[]={0,0}, chunkVz[]={0,0};
#define DEC_CHANNEL_PTR\
  channelSx = pars.drop.channelAddr[0], channelSy = pars.drop.channelAddr[1], channelSz = pars.drop.channelAddr[2];
#else 
#define DEC_CHUNKS_0 const int chunkSi[]={ 0,2}, chunkTx[]={0,3}, chunkTy[]={1,3}, chunkTz[]={0,3}, chunkVx[]={2,4}, chunkVy[]={1,4}, chunkVz[]={2,4};
#define DEC_CHUNKS_1 const int chunkSi[]={-1,0}, chunkTx[]={0,0}, chunkTy[]={0,1}, chunkTz[]={0,0}, chunkVx[]={1,2}, chunkVy[]={0,0}, chunkVz[]={1,2};
#define DEC_CHANNEL_PTR\
  channelSx = pars.drop.channelAddr[0]; channelSy = pars.drop.channelAddr[1]; channelSz = pars.drop.channelAddr[2];\
  channelTx = pars.drop.channelAddr[3]; channelTy = pars.drop.channelAddr[4]; channelTz = pars.drop.channelAddr[5];\
  channelVx = pars.drop.channelAddr[6]; channelVy = pars.drop.channelAddr[7]; channelVz = pars.drop.channelAddr[8];
#endif
#define tshift_coeff Ntime
#ifdef TEST_RATE
#define BLOCK_SPACING TEST_RATE
#else
#define BLOCK_SPACING 1
#endif

#ifdef SPLIT_ZFORM
#define ZFROMTHREAD izBeg-5+threadIdx.x
#else 
#define ZFROMTHREAD get_iz(threadIdx.x)
#endif

#define REG_DEC(EVENTYPE) \
  const int iy=(y0+BLOCK_SPACING*blockIdx.x)%Na;\
  const int tshift=tshift_coeff*pars.iStep;\
  const bool inPMLv = (threadIdx.x<Npmlz);\
  const int iz=ZFROMTHREAD; const int pml_iz=threadIdx.x, Kpml_iz=2*threadIdx.x;\
  if(iz<0 || iz>=Nv) return;\
  const int eventype=EVENTYPE;\
  /*const int izP0=iz, izP1 = (iz+1)%Nv, izP2 = (iz+2)%Nv, izM1 = (iz-1+Nv)%Nv, izM2 = (iz-2+Nv)%Nv;*/\
  const int izP0=threadIdx.x, izP0m=izP0, izP1m = Vrefl(iz+1,0)-iz+izP0, izP2m = Vrefl(iz+2,0)-iz+izP0, izM1m = Vrefl(iz-1,0)-iz+izP0, izM2m = Vrefl(iz-2,0)-iz+izP0;\
  const int                   izP0c=izP0, izP1c = Vrefl(iz+1,1)-iz+izP0, izP2c = Vrefl(iz+2,1)-iz+izP0, izM1c = Vrefl(iz-1,1)-iz+izP0, izM2c = Vrefl(iz-2,1)-iz+izP0;\
  const int Kpml_iy=get_pml_iy(iy)*NDT*2; int Kpml_ix=0;\
  int it=t0; ftype difx[100],dify[100],difz[100]; ftype zerov=0.;\
  register ftype coffV = defCoff::drho*dtdrd24, coffT=defCoff::Vs*defCoff::Vs*defCoff::rho*dtdrd24; \
  register coffS_t coffS = DEF_COFF_S;\
  register ftype ArrcoffV[100], ArrcoffT[100];\
  register coffS_t ArrcoffS[100];\
  float alpha,beta,afloor,bfloor,bceil;\
  const int texNwindow = int(ceil(Ns*NDT*2.0*texStretch[0].x)+2);\
  int I; register htype h[100];\
  ftype2 regPml; ftype regPml2; \
  ftype2 reg_fldV[250], reg_fldS[250]; ftype reg_R;\
  const int iy_p0=iy, iy_p1=iy+1, iy_p2=iy+2, iy_p3=iy+3;\
  const int iy_m1=iy-1, iy_m2=iy-2;\
  const int dStepT=1, dStepX=1, dStepRag=Na, dStepRagPML=Npmly; \
      ftype src0x,src1x,src2x,src3x;\
      ftype src0y,src1y,src2y,src3y;\
      ftype src0z,src1z,src2z,src3z;\
      bool upd_inSF, neigh_inSF;\
\
  int glob_ix = (ix+pars.GPUx0+NS-pars.wleft)%NS+pars.wleft;\
  DEC_CHUNKS_##EVENTYPE;\
  ftype *channelSx=0, *channelSy=0, *channelSz=0, *channelTx=0, *channelTy=0, *channelTz=0, *channelVx=0,*channelVy=0,*channelVz=0;\
  int ymC=0,ymM=0,ymP=0;\
  const int idevC=get_idev(iy  ,ymC); \
  const int idevM=get_idev(iy-1,ymM); \
  const int idevP=get_idev(iy+1,ymP); \
  /*if(idevC==0) { DEC_CHANNEL_PTR }*/\
  int y_tmp=0; const int curDev=get_idev(y0, y_tmp); \
  const int dStepRagC=NStripe(idevC);\
  const int dStepRagM=NStripe(idevM);\
  const int dStepRagP=NStripe(idevP);\
  DiamondRag      * __restrict__ RAG0       = &pars.rags[curDev][iy  -ymC];\
  DiamondRag      * __restrict__ RAGcc      = RAG0+ ix           *dStepRagC;\
  DiamondRag      * __restrict__ RAGmc      = RAG0+((ix-1+Ns)%Ns)*dStepRagC;\
  ModelRag * __restrict__ modelRag = &pars.ragsInd[curDev][iy-ymC];\
  int xstart=ix;\
  const bool isTopStripe = (curDev==NDev-1 && pars.subnode==NasyncNodes-1);\
  const bool isBotStripe = (curDev==0      && pars.subnode==0            );\
  if(iy==ymC+NStripe(curDev)-1 && !isTopStripe) if(EVENTYPE==0) load_buffer<EVENTYPE>(RAG0, pars.p2pBufP[curDev], xstart, t0, Nt, curDev, iz,pml_iz,inPMLv);\
  if(iy==ymC                   && !isBotStripe) if(EVENTYPE==1) load_buffer<EVENTYPE>(RAG0, pars.p2pBufM[curDev], xstart, t0, Nt, curDev, iz,pml_iz,inPMLv);

#define POSTEND(EVENTYPE)\
  //if(iy==ymC+NStripe(curDev)-1 && !isTopStripe) if(EVENTYPE==0) save_buffer<EVENTYPE>(RAG0, pars.p2pBufP[curDev], xstart, t0, Nt, curDev, iz,pml_iz,inPMLv);\
  if(iy==ymC                   && !isBotStripe) if(EVENTYPE==1) save_buffer<EVENTYPE>(RAG0, pars.p2pBufM[curDev], xstart, t0, Nt, curDev, iz,pml_iz,inPMLv);

#define PTR_DEC \
  const int ixm=(ix-1+Ns)%Ns, ixp=(ix+1)%Ns;\
                                 RAGcc      = RAG0+ix *dStepRagC;\
  DiamondRag      * __restrict__ RAGcm      = RAGcc-1;\
  DiamondRag      * __restrict__ RAGcp      = RAGcc+1;\
                                 RAGmc      = RAG0+ixm*dStepRagC;\
  DiamondRag      * __restrict__ RAGmm      = RAGmc-1;\
  DiamondRag      * __restrict__ RAGmp      = RAGmc+1;\
  DiamondRag      * __restrict__ RAGpc      = RAG0+ixp*dStepRagC;\
  DiamondRag      * __restrict__ RAGpm      = RAGpc-1;\
  DiamondRag      * __restrict__ RAGpp      = RAGpc+1;\
  DiamondRagPML   * __restrict__ ApmlRAGcc  = &pars.ragsPMLa[idevC][ix *Npmly+get_pml_iy(iy)%(Npmly)];\
  DiamondRagPML   * __restrict__ ApmlRAGmc  = &pars.ragsPMLa[idevC][ixm*Npmly+get_pml_iy(iy)%(Npmly)];\
  DiamondRagPML   * __restrict__ ApmlRAGpc  = &pars.ragsPMLa[idevC][ixp*Npmly+get_pml_iy(iy)%(Npmly)];\
  DiamondRagPML   * __restrict__ SpmlRAGcc;/*  = &pars.ragsPMLs[idevC][((ix  <Npmlx/2)? ix   :(ix  -Ns+Npmlx))*dStepRagC   +iy-ymC];*/\
  DiamondRagPML   * __restrict__ SpmlRAGmc;/*  = &pars.ragsPMLs[idevC][((ix-1<Npmlx/2)?(ix-1):(ix-1-Ns+Npmlx))*dStepRagC   +iy-ymC];*/\
  DiamondRagPML   * __restrict__ SpmlRAGpc;/*  = &pars.ragsPMLs[idevC][((ix+1<Npmlx/2)?(ix+1):(ix+1-Ns+Npmlx))*dStepRagC   +iy-ymC];*/\
  if(ix  <Npmlx/2) SpmlRAGcc  = &pars.ragsPMLsL[idevC][  ix              *dStepRagC   +iy-ymC];\
  else             SpmlRAGcc  = &pars.ragsPMLsR[idevC][ (ix  -Ns+Npmlx/2)*dStepRagC   +iy-ymC];\
  if(ix-1<Npmlx/2) SpmlRAGmc  = &pars.ragsPMLsL[idevC][ (ix-1           )*dStepRagC   +iy-ymC];\
  else             SpmlRAGmc  = &pars.ragsPMLsR[idevC][ (ix-1-Ns+Npmlx/2)*dStepRagC   +iy-ymC];\
  if(ix+1<Npmlx/2) SpmlRAGpc  = &pars.ragsPMLsL[idevC][ (ix+1           )*dStepRagC   +iy-ymC];\
  else             SpmlRAGpc  = &pars.ragsPMLsR[idevC][((ix+1-Ns+Npmlx/2)%(Npmlx/2))*dStepRagC   +iy-ymC];\
  /*if(iy-1<0  ) { RAGcm=0; RAGmm=0; RAGpm=0; }*/\
  /*if(iy+1>=Na) { RAGcp=0; RAGmp=0; RAGpp=0; }*/;\
  ModelRag * __restrict__ modelRagC = modelRag +ix *dStepRagC;\
  ModelRag * __restrict__ modelRagM = modelRag +ixm*dStepRagC;\
  ModelRag * __restrict__ modelRagP = modelRag +ixp*dStepRagC;\
  /*if(threadIdx.x==0 && ix==11) printf("sRAGcc=%p, sRAGmc=%p, sRAGpc=%p, sRAG0=%p\n", SpmlRAGcc, SpmlRAGmc, SpmlRAGpc, pars.ragsPMLs[idevC] );*/\
  /*if(threadIdx.x==0 && blockIdx.x==0) printf(" steps %d %d %d idev %d %d %d ymC,ymM,ymP=%d,%d,%d ix=%d iy=%d rcc=%p, rcm=%p, rcp=%p, rmc=%p, rmm=%p, rmp=%p, rpc=%p, rpm=%p, rpp=%p, rccPMLa=%p, rmcPMLa=%p, rpcPMLa=%p, rccPMLs=%p, rmcPMLs=%p, rpcPMLs=%p\n", dStepRagC, dStepRagM, dStepRagP, idevC, idevM, idevP, ymC, ymM, ymP, ix, iy,  RAGcc, RAGcm, RAGcp, RAGmc, RAGmm, RAGmp, RAGpc, RAGpm, RAGpp, ApmlRAGcc, ApmlRAGmc, ApmlRAGpc, SpmlRAGcc, SpmlRAGmc, SpmlRAGpc);*/

#ifdef SPLIT_ZFORM
#define isCONzT(xc,zc) \
2*iz+zc>=izBeg*2-xc     && 2*iz+zc<izEnd*2+xc     &&          zform==0 && eventype==0 || \
2*iz+zc>=izBeg*2+xc-6   && 2*iz+zc<izEnd*2-xc+6   &&          zform==1 && eventype==0 || \
2*iz+zc>=izBeg*2-xc-3   && 2*iz+zc<izEnd*2+xc+3   && xc<3  && zform==0 && eventype==1 || \
2*iz+zc>=izBeg*2-xc+3   && 2*iz+zc<izEnd*2+xc-3   && xc>=3 && zform==0 && eventype==1 || \
2*iz+zc>=izBeg*2+xc+3-6 && 2*iz+zc<izEnd*2-xc-3+6 && xc<3  && zform==1 && eventype==1 || \
2*iz+zc>=izBeg*2+xc-3-6 && 2*iz+zc<izEnd*2-xc+3+6 && xc>=3 && zform==1 && eventype==1

#define isCONzS(xc,zc) isCONzT(xc,zc)

#define isCONzV(xc,zc) \
2*iz+zc>=izBeg*2-xc     && 2*iz+zc<izEnd*2+xc     &&          zform==0 && eventype==1 || \
2*iz+zc>=izBeg*2+xc-6   && 2*iz+zc<izEnd*2-xc+6   &&          zform==1 && eventype==1 || \
2*iz+zc>=izBeg*2-xc-3   && 2*iz+zc<izEnd*2+xc+3   && xc<3  && zform==0 && eventype==0 || \
2*iz+zc>=izBeg*2-xc+3   && 2*iz+zc<izEnd*2+xc-3   && xc>=3 && zform==0 && eventype==0 || \
2*iz+zc>=izBeg*2+xc+3-6 && 2*iz+zc<izEnd*2-xc-3+6 && xc<3  && zform==1 && eventype==0 || \
2*iz+zc>=izBeg*2+xc-3-6 && 2*iz+zc<izEnd*2-xc+3+6 && xc>=3 && zform==1 && eventype==0
#else //if not def SPLIT_ZFORM
#define isCONzT(xc,zc) 1
#define isCONzS(xc,zc) 1
#define isCONzV(xc,zc) 1
#endif //SPLIT_ZFORM

#define RPOINT_CHUNK_HEAD ;

#define RPOINT_CHUNK_SHIFT ;

#define I01 1
#define I02 2
#define I03 3
#define I04 4
#define I05 5
#define I06 6
#define I07 7
#define I08 8
#define I09 9
#define I10 10
#define I11 11
#define I12 12
#define I13 13
#define I14 14
#define I15 15
#define I16 16
#define I17 17
#define I18 18
#define I19 19
#define I20 20
#define I21 21
#define I22 22
#define I23 23
#define I24 24
#define I25 25
#define I26 26
#define I27 27
#define I28 28
#define I29 29
#define I30 30
#define I31 31                                                                                                                                                                                                                              
#define I32 32                                                                                                                                                                                                                              
#define I33 33                                                                                                                                                                                                                              
#define I34 34                                                                                                                                                                                                                              
#define I35 35                                                                                                                                                                                                                              
#define I36 36                                                                                                                                                                                                                              
#define I37 37                                                                                                                                                                                                                              
#define I38 38                                                                                                                                                                                                                              
#define I39 39                                                                                                                                                                                                                              
#define I40 40                                                                                                                                                                                                                              
#define I41 41                                                                                                                                                                                                                              
#define I42 42                                                                                                                                                                                                                              
#define I43 43                                                                                                                                                                                                                              
#define I44 44                                                                                                                                                                                                                              
#define I45 45                                                                                                                                                                                                                              
#define I46 46                                                                                                                                                                                                                              
#define I47 47                                                                                                                                                                                                                              
#define I48 48                                                                                                                                                                                                                              
#define I49 49                                                                                                                                                                                                                              
#define I50 50                                                                                                                                                                                                                              
#define I51 51                                                                                                                                                                                                                              
#define I52 52                                                                                                                                                                                                                              
#define I53 53                                                                                                                                                                                                                              
#define I54 54                                                                                                                                                                                                                              
#define I55 55                                                                                                                                                                                                                              
#define I56 56                                                                                                                                                                                                                              
#define I57 57                                                                                                                                                                                                                              
#define I58 58                                                                                                                                                                                                                              
#define I59 59                                                                                                                                                                                                                              
#define I60 60                                                                                                                                                                                                                              
#define I61 61                                                                                                                                                                                                                              
#define I62 62                                                                                                                                                                                                                              
#define I63 63                                                                                                                                                                                                                              
#define I64 64                                                                                                                                                                                                                              
#define I65 65                                                                                                                                                                                                                              
#define I66 66                                                                                                                                                                                                                              
#define I67 67                                                                                                                                                                                                                              
#define I68 68                                                                                                                                                                                                                              
#define I69 69
#define I70 70
#define I71 71
#define I72 72
#define I73 73
#define I74 74
#define I75 75
#define I76 76
#define I77 77
#define I78 78
#define I79 79
#define I80 80
#define I81 81
#define I82 82
#define I83 83
#define I84 84
#define I85 85
#define I86 86
#define I87 87
#define I88 88
#define I89 89
#define I90 90
#define I91 91
#define I92 92
#define I93 93
#define I94 94
#define I95 95
#define I96 96
#define I97 97
#define I98 98
#define I99 99

#endif//DEFS_H
