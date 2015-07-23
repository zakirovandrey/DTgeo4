#include "cuda_math.h"
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef MPI_ON
#include <mpi.h>
#endif
#include "omp.h"
#include "params.h"
#include "init.h" 
#include "signal.hpp"
#include "diamond.cu"

__global__ void calc_limits(float* buf, float* fLims) {
  float2 fLim;
  float* pf=buf+blockIdx.x*Nz+threadIdx.x;
  fLim.x = fLim.y = *pf;

  for(int i=0; i<Nx; i++,pf+=Ny*Nz) {
    float v=*pf;
    if(v<fLim.x) fLim.x = v;
    if(v>fLim.y) fLim.y = v;
  }
  __shared__ float2 fLim_sh[Nz];
  fLim_sh[threadIdx.x] = fLim;
  if(threadIdx.x>warpSize) return;
  for(int i=threadIdx.x; i<Nz; i+=warpSize) {
    float2 v=fLim_sh[i];
    if(v.x<fLim.x) fLim.x = v.x;
    if(v.y>fLim.y) fLim.y = v.y;
  }
  fLim_sh[threadIdx.x] = fLim;
  if(threadIdx.x>0) return;
  for(int i=0; i<warpSize; i++) {
    float2 v=fLim_sh[i];
    if(v.x<fLim.x) fLim.x = v.x;
    if(v.y>fLim.y) fLim.y = v.y;
  }
  fLims[2*blockIdx.x  ] = fLim.x;
  fLims[2*blockIdx.x+1] = fLim.y;
}

#include "im2D.h"
#include "im3D.hpp"
int type_diag_flag=0;

im3D_pars im3DHost;

#ifdef USE_TEX_REFS
extern texture<coffS_t, cudaTextureType3D, cudaReadModeElementType> layerRefS;
extern texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefV;
extern texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefT;
#endif
texture<coffS_t, cudaTextureType3D, cudaReadModeElementType> ShowRefS;
texture<float , cudaTextureType3D, cudaReadModeElementType> ShowRefV;
texture<float , cudaTextureType3D, cudaReadModeElementType> ShowRefT;
void CreateShowTexModel(){
  if(parsHost.texs.ShowTexBinded) return;
  parsHost.texs.ShowTexBinded=1;
  printf("Creating Model texture for im3D showing (size=%.2fMb)\n", parsHost.texs.texN[0].y*parsHost.texs.texN[0].z*parsHost.texs.texN[0].x*sizeof(float)*4/1024./1024. );
  ShowRefS.addressMode[0] = layerRefS.addressMode[0]; ShowRefV.addressMode[0] = layerRefV.addressMode[0]; ShowRefT.addressMode[0] = layerRefT.addressMode[0];
  ShowRefS.addressMode[1] = layerRefS.addressMode[1]; ShowRefV.addressMode[1] = layerRefV.addressMode[1]; ShowRefT.addressMode[1] = layerRefT.addressMode[1];
  ShowRefS.addressMode[2] = layerRefS.addressMode[2]; ShowRefV.addressMode[2] = layerRefV.addressMode[2]; ShowRefT.addressMode[2] = layerRefT.addressMode[2];
  ShowRefS.filterMode = cudaFilterModeLinear; ShowRefV.filterMode = layerRefV.filterMode; ShowRefT.filterMode = layerRefT.filterMode;
  ShowRefS.normalized = layerRefS.normalized; ShowRefV.normalized = layerRefV.normalized; ShowRefT.normalized = layerRefT.normalized;

  cudaArray* DevModelS, *DevModelV, *DevModelT;
  cudaChannelFormatDesc channelDesc;
  channelDesc = cudaCreateChannelDesc<coffS_t>(); CHECK_ERROR( cudaMalloc3DArray(&DevModelS, &channelDesc, make_cudaExtent(parsHost.texs.texN[0].y,parsHost.texs.texN[0].z,parsHost.texs.texN[0].x)) );
  channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevModelV, &channelDesc, make_cudaExtent(parsHost.texs.texN[0].y,parsHost.texs.texN[0].z,parsHost.texs.texN[0].x)) );
  channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevModelT, &channelDesc, make_cudaExtent(parsHost.texs.texN[0].y,parsHost.texs.texN[0].z,parsHost.texs.texN[0].x)) );

  const int texNz = parsHost.texs.texN[0].y, texNy = parsHost.texs.texN[0].z;
  cudaMemcpy3DParms copyparms={0}; copyparms.srcPos=make_cudaPos(0,0,0); copyparms.dstPos=make_cudaPos(0,0,0);
  copyparms.kind=cudaMemcpyHostToDevice;
  copyparms.srcPtr = make_cudaPitchedPtr(&parsHost.texs.HostLayerS[0][0], texNz*sizeof(coffS_t), texNz, texNy);
  copyparms.dstArray = DevModelS;
  copyparms.extent = make_cudaExtent(texNz,texNy,parsHost.texs.texN[0].x);
  CHECK_ERROR( cudaMemcpy3D(&copyparms) );
  copyparms.srcPtr = make_cudaPitchedPtr(&parsHost.texs.HostLayerV[0][0], texNz*sizeof(float  ), texNz, texNy);
  copyparms.dstArray = DevModelV;
  copyparms.extent = make_cudaExtent(texNz,texNy,parsHost.texs.texN[0].x);
  CHECK_ERROR( cudaMemcpy3D(&copyparms) );
  copyparms.srcPtr = make_cudaPitchedPtr(&parsHost.texs.HostLayerT[0][0], texNz*sizeof(float  ), texNz, texNy);
  copyparms.dstArray = DevModelT;
  copyparms.extent = make_cudaExtent(texNz,texNy,parsHost.texs.texN[0].x);
  CHECK_ERROR( cudaMemcpy3D(&copyparms) );

  channelDesc = cudaCreateChannelDesc<coffS_t>(); CHECK_ERROR( cudaBindTextureToArray(ShowRefS, DevModelS, channelDesc) );
  channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(ShowRefV, DevModelV, channelDesc) );
  channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(ShowRefT, DevModelT, channelDesc) );
}
char* FuncStr[] = {"Sx","Sy","Sz","Tx","Ty","Tz", "Vx", "Vy", "Vz", "I1", "I2", "I3", "kappa=l+2m (rho*Vp^2)", "lambda (rho*(Vp^2-2*Vs^2))", 
"mu_Tx (rho*Vs^2)", "mu_Ty (rho*Vs^2)", "mu_Tz (rho*Vs^2)",
"1/rho_Vx", "1/rho_Vy", "1/rho_Vz",
"Index0", "Index1", 
"h_Si", "h_Tx", "h_Ty", "h_Tz", "h_Vx", "h_Vy", "h_Vz" };
__device__ float pow2(float v) { return v*v; }
#define MXW_DRAW_ANY(val) *pbuf = val;
__global__ void mxw_draw(float* buf) {
  const float d3=1./3; float val=0;
  int iz=threadIdx.x;
  DiamondRag* p=&pars.get_plaster(blockIdx.x,blockIdx.y);
  ModelRag* index=&pars.get_index(blockIdx.x,blockIdx.y);
  const int Npls=2*NDT*NDT;
  #define MODELCOFF_S(text,xv,yv,hv) tex3D(ShowRefS, (yv)*texStretch[0].y+texShift[0].y, (hv)*texStretchH, (xv)*texStretchShow.x+texShiftShow.x)
  #define MODELCOFF_V(text,xv,yv,hv) tex3D(ShowRefV, (yv)*texStretch[0].y+texShift[0].y, (hv)*texStretchH, (xv)*texStretchShow.x+texShiftShow.x)
  #define MODELCOFF_T(text,xv,yv,hv) tex3D(ShowRefT, (yv)*texStretch[0].y+texShift[0].y, (hv)*texStretchH, (xv)*texStretchShow.x+texShiftShow.x)
  for(int idom=0; idom<Npls; idom++) {
    int Ragdir=0;
    if(pars.nFunc==6 || pars.nFunc==7 || pars.nFunc==8 || pars.nFunc==17 || pars.nFunc==18 || pars.nFunc==19 || pars.nFunc==26 || pars.nFunc==27 || pars.nFunc==28 ) Ragdir=1;
    int shx=-NDT+idom/NDT+idom%NDT;
    int shy=+NDT-idom/NDT+idom%NDT;
    if(Ragdir) shy=0+idom/NDT-idom%NDT;
    int idev=0, nextY=NStripe(0);
    #if 1
      while(blockIdx.y>=nextY) nextY+=NStripe(++idev);
      shy-=idev*2*NDT;
      if(blockIdx.y==nextY-NStripe(idev) && idom< Npls/2 && Ragdir==1 && idev!=0     ) continue;
      if(blockIdx.y==nextY-NStripe(idev) && idom>=Npls/2 && Ragdir==0 && idev!=0     ) continue;
      if(blockIdx.y==nextY-1             && idom< Npls/2 && Ragdir==0 && idev!=NDev-1) continue;
      if(blockIdx.y==nextY-1             && idom>=Npls/2 && Ragdir==1 && idev!=NDev-1) continue;
    #endif
    int ix = blockIdx.x*2*NDT+shx+4;
    int iy = blockIdx.y*2*NDT+shy+2;
    float* pbuf=&buf[threadIdx.x+NT*(iy/2+Ny*(ix/2))];
    switch(pars.nFunc) {
      case 0 : if(idom%2==0) { pbuf+=0    ; MXW_DRAW_ANY(p->Si[idom/2].duofld[0][iz].x ); } break; //Sx
      case 1 : if(idom%2==0) { pbuf+=0    ; MXW_DRAW_ANY(p->Si[idom/2].duofld[0][iz].y ); } break; //Sy
      case 2 : if(idom%2==0) { pbuf+=0    ; MXW_DRAW_ANY(p->Si[idom/2].duofld[1][iz].x ); } break; //Sz
      case 4 : if(idom%2==0) { pbuf+=NT*Ny; MXW_DRAW_ANY(p->Si[idom/2].duofld[1][iz].y ); } break; //Ty
      case 5 : if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(p->Si[idom/2].duofld[2][iz].x ); } break; //Tz
      case 3 : if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(p->Si[idom/2].duofld[2][iz].y ); } break; //Tx
      case 7 : if(idom%2==0) { pbuf+=0    ; MXW_DRAW_ANY(p->Vi[idom/2].trifld.one[iz]  ); } break; //Vy
      case 6 : if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(p->Vi[idom/2].trifld.two[iz].x); } break; //Vx
      case 8 : if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(p->Vi[idom/2].trifld.two[iz].y); } break; //Vz
      case 9 : if(idom%2==0) { pbuf+=0    ; MXW_DRAW_ANY(d3*(p->Si[idom/2].duofld[0][iz].x+p->Si[idom/2].duofld[0][iz].y+p->Si[idom/2].duofld[1][iz].x)); } break; //Inv1
      case 10: if(idom%2==0) { pbuf+=0    ; ftype Sx = p->Si[idom/2].duofld[0][iz].x; ftype Sy = p->Si[idom/2].duofld[0][iz].y; ftype Sz = p->Si[idom/2].duofld[1][iz].x; atomicAdd(pbuf, Sx*Sy+Sx*Sz+Sy*Sz); }
               if(idom%2==1) { pbuf+=0    ; ftype Tx = p->Si[idom/2].duofld[2][iz].y; atomicAdd(pbuf,-0.25*Tx*Tx);if(iz  <Nz-1)atomicAdd(pbuf+1 ,-0.25*Tx*Tx);if(iy/2<Ny-1)atomicAdd(pbuf+NT   ,-0.25*Tx*Tx);if(iz  <Nz-1 && iy/2<Ny-1)atomicAdd(pbuf+NT   +1 ,-0.25*Tx*Tx); } 
               if(idom%2==0) { pbuf+=NT*Ny; ftype Ty = p->Si[idom/2].duofld[1][iz].y; atomicAdd(pbuf,-0.25*Ty*Ty);if(iz  <Nz-1)atomicAdd(pbuf+1 ,-0.25*Ty*Ty);if(ix/2<Nx-1)atomicAdd(pbuf+NT*Ny,-0.25*Ty*Ty);if(iz  <Nz-1 && ix/2<Nx-1)atomicAdd(pbuf+NT*Ny+1 ,-0.25*Ty*Ty); } 
               if(idom%2==1) { pbuf+=0    ; ftype Tz = p->Si[idom/2].duofld[2][iz].x; atomicAdd(pbuf,-0.25*Tz*Tz);if(iy/2<Ny-1)atomicAdd(pbuf+NT,-0.25*Tz*Tz);if(ix/2<Nx-1)atomicAdd(pbuf+NT*Ny,-0.25*Tz*Tz);if(iy/2<Ny-1 && ix/2<Nx-1)atomicAdd(pbuf+NT*Ny+NT,-0.25*Tz*Tz); } break; //inv2
      case 11: break; //inv3
      case 12: if(idom%2==0)              { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_S(ShowRefS, ix, (iz*2+1), index->h[ idom/2*2     ][iz].x).x ); } break; // Vp*Vp*rho
      case 13: if(idom%2==0)              { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_S(ShowRefS, ix, (iz*2+1), index->h[ idom/2*2     ][iz].x).y ); } break; // (Vp*Vp-2*Vs*Vs)*rho
      case 14: if(idom%2==1)              { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_T(ShowRefT, ix, (iz*2  ), index->h[ idom/2*2+1   ][iz].y)   ); } break; // Vs*Vs*rho for T_x
      case 15: if(idom%2==0)              { pbuf+=NT*Ny; MXW_DRAW_ANY(MODELCOFF_T(ShowRefT, ix, (iz*2  ), index->h[ idom/2*2     ][iz].y)   ); } break; // Vs*Vs*rho for T_y
      case 16: if(idom%2==1)              { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_T(ShowRefT, ix, (iz*2+1), index->h[ idom/2*2+1   ][iz].x)   ); } break; // Vs*Vs*rho for T_z
      case 17: if(idom%2==1 && (idom+1)%4==0) { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_V(ShowRefV, ix, (iz*2+1), index->h[Npls+(idom/2*3+1)/2][iz].x)   ); } 
          else if(idom%2==1 && (idom+1)%4!=0) { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_V(ShowRefV, ix, (iz*2+1), index->h[Npls+(idom/2*3+1)/2][iz].y)   ); } break; // 1/rho for V_x
      case 18: if(idom%2==0 && idom    %4==0) { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_V(ShowRefV, ix, (iz*2+1), index->h[Npls+ idom/2*3   /2][iz].x)   ); } 
          else if(idom%2==0 && idom    %4!=0) { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_V(ShowRefV, ix, (iz*2+1), index->h[Npls+ idom/2*3   /2][iz].y)   ); } break; // 1/rho for V_y
      case 19: if(idom%2==1 && (idom+1)%4!=0) { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_V(ShowRefV, ix, (iz*2  ), index->h[Npls+(idom/2*3+2)/2][iz].x)   ); } 
          else if(idom%2==1 && (idom+1)%4==0) { pbuf+=0    ; MXW_DRAW_ANY(MODELCOFF_V(ShowRefV, ix, (iz*2  ), index->h[Npls+(idom/2*3+2)/2][iz].y)   ); } break; // 1/rho for V_z
  //    case 20: if(idom%2==0) { pbuf+=0    ; MXW_DRAW_ANY(index->I[(idom<NDT*NDT)?0:1][iz]); } break; //Index0
  //    case 21: if(idom%2==0) { pbuf+=0    ; MXW_DRAW_ANY(index->I[(idom<NDT*NDT)?2:3][iz]); } break; //Index1
      case 22: if(idom%2==0) { pbuf+=0    ; MXW_DRAW_ANY(index->h[idom/2*2  ][iz].x); } break; //h_Si
      case 24: if(idom%2==0) { pbuf+=NT*Ny; MXW_DRAW_ANY(index->h[idom/2*2  ][iz].y); } break; //h_Ty
      case 25: if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(index->h[idom/2*2+1][iz].x); } break; //h_Tz
      case 23: if(idom%2==1) { pbuf+=0    ; MXW_DRAW_ANY(index->h[idom/2*2+1][iz].y); } break; //h_Tx
      case 27: if(idom%2==0 && idom%4==0    ) { pbuf+=0    ; MXW_DRAW_ANY(index->h[Npls+idom/2*3/2    ][iz].x); } 
          else if(idom%2==0 && idom%4!=0    ) { pbuf+=0    ; MXW_DRAW_ANY(index->h[Npls+idom/2*3/2    ][iz].y); }; break; //h_Vy
      case 26: if(idom%2==1 && (idom+1)%4==0) { pbuf+=0    ; MXW_DRAW_ANY(index->h[Npls+(idom/2*3+1)/2][iz].x); }
          else if(idom%2==1 && (idom+1)%4!=0) { pbuf+=0    ; MXW_DRAW_ANY(index->h[Npls+(idom/2*3+1)/2][iz].y); }; break; //h_Vx
      case 28: if(idom%2==1 && (idom+1)%4!=0) { pbuf+=0    ; MXW_DRAW_ANY(index->h[Npls+(idom/2*3+2)/2][iz].x); }
          else if(idom%2==1 && (idom+1)%4==0) { pbuf+=0    ; MXW_DRAW_ANY(index->h[Npls+(idom/2*3+2)/2][iz].y); }; break; //h_Vz
    }
    if(pars.nFunc<12 && pars.bgMat) if(idom%2==0) { pbuf+=0    ; atomicAdd(pbuf, 5*pow(0.1,double(7-pars.bgMat))*MODELCOFF_S(ShowRefS, ix, (iz*2+1), index->h[ idom/2*2     ][iz].x).y ); }
  }
  switch(pars.nFunc) {
//    case 10: val=1./3*(c->Sx[iz]*c->Sy[iz]+c->Sx[iz]*c->Sz[iz]+c->Sy[iz]*c->Sz[iz]-c->Tx[iz]*c->Tx[iz]-c->Ty[iz]*c->Ty[iz]-c->Tz[iz]*c->Tz[iz]); MXW_DRAW_ANY(val>0?sqrt(val):-sqrt(-val)); break;
//    case 11: val=1./3*(c->Sx[iz]*c->Sy[iz]*c->Sz[iz]+2*c->Tx[iz]*c->Ty[iz]*c->Tz[iz]-c->Sx[iz]*c->Tx[iz]*c->Tx[iz]-c->Sy[iz]*c->Ty[iz]*c->Ty[iz]-c->Sz[iz]*c->Tz[iz]*c->Tz[iz]); MXW_DRAW_ANY(pow(val, 1.f/3.f)); break;
//    case 12: MXW_DRAW_ANY(TEXVp (blockIdx.x,blockIdx.y,threadIdx.x )); break;
//    case 13: MXW_DRAW_ANY(TEXVs (blockIdx.x,blockIdx.y,threadIdx.x )); break;
//    case 14: MXW_DRAW_ANY(TEXRho(blockIdx.x,blockIdx.y,threadIdx.x )); break;
    default: break;
  }
}

struct any_idle_func_struct {
    virtual void step() {}
};
struct idle_func_calc: public any_idle_func_struct {
  float t;
  void step();
};
void idle_func_calc::step() {
  calcStep();
  CreateShowTexModel();
  CHECK_ERROR( cudaMemset(parsHost.arr4im.Arr3Dbuf,0,((long long int)Nx)*Ny*Nz*sizeof(ftype)) ); mxw_draw<<<dim3((USE_UVM==2)?Np:Ns,Na),NT>>>(parsHost.arr4im.Arr3Dbuf);
  im3DHost.initCuda(parsHost.arr4im);
  recalc_at_once=true;
}

unsigned char* read_config_file(int& n){
  n = 0; int c; 
  FILE* cfgfile;
  cfgfile = fopen("acts.cfg","r");
  if (cfgfile==NULL) return NULL;
  else {
    c = fgetc(cfgfile); if(c == EOF) {printf("config file is empty"); return NULL; } 
    n = 0;
    while(c != EOF) {
      c = fgetc(cfgfile);
      n++;
    }
    fclose(cfgfile);
  }
  unsigned char* actlist = NULL;
  cfgfile = fopen("acts.cfg","r");
  if (cfgfile==NULL) return NULL;
  else {
    actlist = new unsigned char[n];
    for(int i=0; i<n; i++) { 
      actlist[i] = (unsigned char)fgetc(cfgfile);
      if     (actlist[i]=='\n') actlist[i] = 13;
      else if(actlist[i]=='2' ) actlist[i] = 50;
      else if(actlist[i]=='3' ) actlist[i] = 51;
    }
    fclose(cfgfile);
  }
  return actlist; 
}
int iact = 0;
int nact = 0;
unsigned char* sequence_act = NULL; 
static void key_func(unsigned char key, int x, int y) {
  if(type_diag_flag>=2) printf("keyN=%d, coors=(%d,%d)\n", key, x, y);
  if(key == 'h') {
    printf("\
======= Управление mxw3D:\n\
  <¦>  \tИзменение функции для визуализации: WEH¦Sx¦Ez¦Ey¦Ex¦Hx¦Hy¦Hz¦Sy¦Sz¦eps\n\
«Enter»\tПересчёт одного большого шага\n\
   b   \tвключает пересчёт в динамике (см. «Управление динамикой»)\n\
"); im3DHost.print_help();
    return;
  }
  switch(key) {
  //case '>': if(parsHost.nFunc<parsHost.MaxFunc) parsHost.nFunc++; break;
  //case '<': if(parsHost.nFunc>0) parsHost.nFunc--; break;
  case '>': parsHost.nFunc = (parsHost.nFunc+1)%parsHost.MaxFunc; break;
  case '<': parsHost.nFunc = (parsHost.nFunc+parsHost.MaxFunc-1)%parsHost.MaxFunc; break;
  case 'B': parsHost.bgMat = (parsHost.bgMat+1)%7; break;
  case 13: calcStep(); break;
  //case  8: recalc_at_once=arr3D_list->prev_arr2gpu(); return;
  case 'c': 
    {
    printf("reading config file\n");
    sequence_act = read_config_file(nact);
    glutPostRedisplay();
    return; 
    }
  default: if(!im3DHost.key_func(key, x, y)) {
  if(type_diag_flag>=0) printf("По клавише %d в позиции (%d,%d) нет никакого действия\n", key, x, y);
  } return;
  }
  copy2dev( parsHost, pars );
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  CreateShowTexModel();
  CHECK_ERROR( cudaMemset(parsHost.arr4im.Arr3Dbuf,0,((long long int)Nx)*Ny*Nz*sizeof(ftype)) ); mxw_draw<<<dim3((USE_UVM==2)?Np:Ns,Na),NT>>>(parsHost.arr4im.Arr3Dbuf);
  im3DHost.initCuda(parsHost.arr4im);
  recalc_at_once=true;
}
static void draw_func() { 
  if (iact<nact) { 
    key_func(sequence_act[iact],0,0);
    iact++;
    glutPostRedisplay();
  }
  if(nact>0 && iact==nact) delete[] sequence_act;
  im3DHost.fName = FuncStr[parsHost.nFunc]; im2D.draw(im3DHost.reset_title()); 
}

//void (*idle_func_ptr)(float* );
static void idle_func() { im3DHost.recalc_func(); }
static void mouse_func(int button, int state, int x, int y) { im3DHost.mouse_func(button, state, x, y); }
static void motion_func(int x, int y) { im3DHost.motion_func(x, y); }

double PMLgamma_func(int i, int N, ftype dstep){ //return 0; 
  if(i>=N-3) return 0;
  N-=3;
  double attenuation_factor = 4;
  double sigma_max= shotpoint.V_max*log(10000)*( (attenuation_factor+1)/(2*(N*dstep*0.5)) );
  double x_max = pow(sigma_max, 1./attenuation_factor);
  double x = x_max-i*(x_max/N);
  return pow(x, attenuation_factor);
}
double PMLgamma_funcY(int i, int N, ftype dstep){ //return 0; 
  if(i>=N-3) return 0;
  N-=3;
  double attenuation_factor = 4;
  double sigma_max= shotpoint.V_max*log(10000)*( (attenuation_factor+1)/(2*(N*dstep*0.5)) );
  double x_max = pow(sigma_max, 1./attenuation_factor);
  double x = x_max-i*(x_max/N);
  return pow(x, attenuation_factor);
}
double PMLgamma_funcZ(int i, int N, ftype dstep){ //return 0; 
  if(i>=N-3) return 0;
  N-=3;
  double attenuation_factor = 4;
  double sigma_max= shotpoint.V_max*log(10000)*( (attenuation_factor+1)/(2*(N*dstep*0.5)) );
  double x_max = pow(sigma_max, 1./attenuation_factor);
  double x = x_max-i*(x_max/N);
  return pow(x, attenuation_factor);
}
void setPMLcoeffs(float* k1x, float* k2x, float* k1y, float* k2y, float* k1z, float* k2z) {
  for(int i=0; i<KNpmlx; i++){
    k2x[i] = 1.0/(1.0+0.5*dt*PMLgamma_func(KNpmlx/2-abs(i-KNpmlx/2)-3, KNpmlx/2-3, dx));
    k1x[i] = 2.0*k2x[i]-1;
  }
  for(int i=0; i<KNpmly; i++){
    k2y[i] = 1.0/(1.0+0.5*dt*PMLgamma_funcY(KNpmly-i, KNpmly, dy));
    k1y[i] = 2.0*k2y[i]-1;
  }
  for(int i=0; i<KNpmlz; i++){
    k2z[i] = 1.0/(1.0+0.5*dt*PMLgamma_funcZ(KNpmlz/2-abs(i-KNpmlz/2), KNpmlz/2, dz));
    k1z[i] = 2.0*k2z[i]-1;
  }
}
void GeoParamsHost::set(){
  
  #ifndef USE_WINDOW
  if(Np!=Ns) { printf("Error: if not defined USE_WINDOW Np must be equal Ns\n"); exit(-1); }
  #endif//USE_WINDOW

  int node=0, Nprocs=1;
  #ifdef MPI_ON
  MPI_Comm_rank (MPI_COMM_WORLD, &node);
  MPI_Comm_size (MPI_COMM_WORLD, &Nprocs);
  #endif
  mapNodeSize = new int[Nprocs];
  int accSizes=0;
  for(int i=0; i<Nprocs; i++) mapNodeSize[i] = Np/Nprocs+Ns; 
  int sums=0; for(int i=0; i<Nprocs-1; i++) sums+= mapNodeSize[i]-Ns; mapNodeSize[Nprocs-1]=Np-sums;
  for(int i=0; i<Nprocs; i++) {
    if(node==i) printf("X-size=%d rags on node %d\n", mapNodeSize[i], i);
    #ifdef MPI_ON
    if(mapNodeSize[i]<2*Ns) { printf("Data on node %d is too small\n", i); exit(-1); }
    #endif
    accSizes+= mapNodeSize[i];
  }
  if(accSizes-Ns*(Nprocs-1)!=Np) { printf("Error: sum (mapNodes) must be = Np+Ns*(Nprocs-1)\n"); exit(-1); }
  #ifdef MPI_ON
  if(mapNodeSize[0]       <=Npmlx/2+Ns+Ns) { printf("Error: mapNodeSize[0]<=Npmlx/2+Ns+Ns\n"); exit(-1); }
  if(mapNodeSize[Nprocs-1]<=Npmlx/2+Ns+Ns) { printf("Error: mapNodeSize[Nodes-1]<=Npmlx/2+Ns+Ns\n"); exit(-1); }
  #endif

  //dir= new string("/Run/zakirov/tmp/"); //ix=Nx+Nbase/2; Yshtype=0;
  dir= new std::string(im3DHost.drop_dir);
  drop.dir=dir;
  struct stat st = {0};

  if (stat(dir->c_str(), &st) == -1)  mkdir(dir->c_str(), 0700);

  if(node==0) printf("Grid size: %dx%d Rags /%dx%dx%d Yee_cells/, TorreH=%d\n", Np, Na, Np*NDT,Na*NDT,Nv, Ntime);
  if(node==0) printf("Window size: %d, copy-shift step %d \n", Ns, Window::NTorres );
  if(gridNx%NDT!=0) { printf("Error: gridNx must be dividable by %d\n", NDT); exit(-1); }
  if(gridNz%NDT!=0) { printf("Error: gridNz must be dividable by %d\n", NDT); exit(-1); }
  if(dt*sqrt(1/(dx*dx)+1/(dy*dy)+1/(dz*dz))>6./7.) { printf("Error: Courant condition is not satisfied\n"); exit(-1); }
//  if(sizeof(DiamondRag)!=sizeof(RagArray)) { printf("Error: sizeof(DiamondRag)=%d != sizeof(RagArray)\n", sizeof(DiamondRag),sizeof(RagArray)); exit(-1); }
  int NaStripe=0; for(int i=0;i<NDev;i++) NaStripe+=NStripe[i]; if(NaStripe!=Na) { printf("Error: sum(NStripes[i])!=NA\n"); exit(-1); }
  iStep = 0; isTFSF=true;
  Zcnt=0.5*Nz*dz;
  nFunc = 0; MaxFunc = sizeof(FuncStr)/sizeof(char*);
  size_t size_xz     = Ns   *sizeof(DiamondRag   );
  size_t size_xzModel= Ns   *sizeof(ModelRag     );
  size_t sz          = Na*size_xz;
  size_t szModel     = Na*size_xzModel;
  size_t szPMLa      = Ns*Npmly*sizeof(DiamondRagPML);
  size_t size_xzPMLs = Npmlx/2*sizeof(DiamondRagPML);
  size_t szPMLs      = Na*size_xzPMLs;
  if(node==0) {
  printf("GPU Cell's Array size     : %.2fM = %.2fM(Main)+%.2fM(Model)+%.2fM(PMLs)+%.2fM(PMLa)\n", 
           (sz+szModel+szPMLs+szPMLa)/(1024.*1024.),
           sz     /(1024.*1024.), 
           szModel/(1024.*1024.), 
           szPMLs /(1024.*1024.), 
           szPMLa /(1024.*1024.)  );
  for(int istrp=0; istrp<NDev-1; istrp++) printf( "                   Stripe%d: %.2fM = %.2fM+%.2fM+%.2fM\n", istrp, 
           (size_xz*NStripe[istrp]+size_xzModel*NStripe[istrp]+size_xzPMLs*NStripe[istrp])/(1024.*1024.),
           size_xz     *NStripe[istrp ]/(1024.*1024.), 
           size_xzModel*NStripe[istrp ]/(1024.*1024.),
           size_xzPMLs *NStripe[istrp ]/(1024.*1024.)  );
                                         printf( "                   Stripe%d: %.2fM = %.2fM+%.2fM+%.2fM+%.2fM\n", NDev-1, 
           (size_xz*NStripe[NDev-1]+size_xzModel*NStripe[NDev-1]+size_xzPMLs*NStripe[NDev-1]+szPMLa)/(1024.*1024.),
           size_xz     *NStripe[NDev-1]/(1024.*1024.), 
           size_xzModel*NStripe[NDev-1]/(1024.*1024.),
           size_xzPMLs *NStripe[NDev-1]/(1024.*1024.), 
           szPMLa                      /(1024.*1024.)  );
  }
  for(int idev=0; idev<NDev; idev++) {
    CHECK_ERROR( cudaSetDevice(idev) );
    CHECK_ERROR( cudaMalloc( (void**)&(ragsInd  [idev]), size_xzModel*NStripe[idev]) );
    CHECK_ERROR( cudaMalloc( (void**)&(rags     [idev]), size_xz     *NStripe[idev]) );
//    CHECK_ERROR( cudaMalloc( (void**)&(ragsPMLs[idev]), size_xzPMLs*NStripe[idev]    ) );
    #ifndef USE_WINDOW
    CHECK_ERROR( cudaMalloc( (void**)&(ragsPMLsL[idev]), size_xzPMLs*NStripe[idev]    ) );
    CHECK_ERROR( cudaMalloc( (void**)&(ragsPMLsR[idev]), size_xzPMLs*NStripe[idev]    ) );
    #endif
    if(idev==NDev-1)
    CHECK_ERROR( cudaMalloc( (void**)& ragsPMLa       , szPMLa ) );
    CHECK_ERROR( cudaMemset(rags   [idev], 0, size_xz     *NStripe[idev]) );
    CHECK_ERROR( cudaMemset(ragsInd[idev], 0, size_xzModel*NStripe[idev]) );
    #ifndef USE_WINDOW
    cudaMemset(ragsPMLsL[idev], 0,  size_xzPMLs*NStripe[idev]);
    cudaMemset(ragsPMLsR[idev], 0,  size_xzPMLs*NStripe[idev]);
    #endif
    if(idev==NDev-1)
    CHECK_ERROR( cudaMemset(ragsPMLa     , 0,                     szPMLa) );
  }
  CHECK_ERROR( cudaSetDevice(0) );

  const int Nn = mapNodeSize[node];
  #if 1//USE_WINDOW
  printf("Allocating RAM memory on node %d: %g Gb\n", node, (Nn*Na*sizeof(DiamondRag)+Nn*Na*sizeof(ModelRag)+Nn*Npmly*sizeof(DiamondRagPML)+Npmlx*Na*sizeof(DiamondRagPML))/(1024.*1024.*1024.));
  #if USE_UVM==2
  CHECK_ERROR( cudaMallocHost(&data     , Nn*Na     *sizeof(DiamondRag   )) ); memset(data     , 0, Nn*Na     *sizeof(DiamondRag   ));
  CHECK_ERROR( cudaMallocHost(&dataInd  , Nn*Na     *sizeof(ModelRag     )) ); memset(dataInd  , 0, Nn*Na     *sizeof(ModelRag     ));
  CHECK_ERROR( cudaMallocHost(&dataPMLa , Nn*Npmly  *sizeof(DiamondRagPML)) ); memset(dataPMLa , 0, Nn*Npmly  *sizeof(DiamondRagPML));
  CHECK_ERROR( cudaMallocHost(&dataPMLsL, Npmlx/2*Na*sizeof(DiamondRagPML)) ); memset(dataPMLsL, 0, Npmlx/2*Na*sizeof(DiamondRagPML));
  CHECK_ERROR( cudaMallocHost(&dataPMLsR, Npmlx/2*Na*sizeof(DiamondRagPML)) ); memset(dataPMLsR, 0, Npmlx/2*Na*sizeof(DiamondRagPML));
  if (node==1) printf("data allocated, pointer to %p\n", data);
  for(int i=0; i<node; i++) data    -= mapNodeSize[i]*Na   ; data    +=node*Ns*Na;
  for(int i=0; i<node; i++) dataInd -= mapNodeSize[i]*Na   ; dataInd +=node*Ns*Na;
  for(int i=0; i<node; i++) dataPMLa-= mapNodeSize[i]*Npmly; dataPMLa+=node*Ns*Npmly;
  if (node==1) printf("now data points to %p\n", data);
  #else
  data     = new DiamondRag   [Nn*Na   ]; memset(data    , 0, Nn*Na   *sizeof(DiamondRag   ));
  dataPMLa = new DiamondRagPML[Nn*Npmly]; memset(dataPMLa, 0, Nn*Npmly*sizeof(DiamondRagPML));
  dataPMLs = new DiamondRagPML[Npmlx*Na]; memset(dataPMLs, 0, Npmlx*Na*sizeof(DiamondRagPML));
  #endif
  #endif

  drop.init();
  texs.init();
  cuTimer t0;
  int xL=0; for(int inode=0; inode<node; inode++) xL+= mapNodeSize[inode]; xL-= Ns*node;
  int xR = xL+mapNodeSize[node];
  omp_set_num_threads(4);
  for(int x=0;x<Np;x++) {
    printf("Initializing h-parameter %.2f%%      \r",100*double(x+1)/Np); fflush(stdout);
    if(x>=xL && x<xR) { 
      #pragma omp parallel for
      for(int y=0;y<Na;y++) dataInd[x*Na+y].set(x,y);
    }
  }
//  printf("t0=%g\n",t0.gettime());
  
  sensors = new std::vector<Sensor>();
}

void init_index() {
  //-------Set PML coeffs----------------------------//
  hostKpmlx1 = new float[KNpmlx]; hostKpmlx2 = new float[KNpmlx];
  hostKpmly1 = new float[KNpmly]; hostKpmly2 = new float[KNpmly];
  hostKpmlz1 = new float[KNpmlz]; hostKpmlz2 = new float[KNpmlz];
  setPMLcoeffs(hostKpmlx1, hostKpmlx2, hostKpmly1, hostKpmly2, hostKpmlz1, hostKpmlz2);
  for(int idev=0; idev<NDev; idev++) {
    CHECK_ERROR( cudaSetDevice(idev) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmlx1, hostKpmlx1, sizeof(float)*KNpmlx) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmlx2, hostKpmlx2, sizeof(float)*KNpmlx) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmly1, hostKpmly1, sizeof(float)*KNpmly) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmly2, hostKpmly2, sizeof(float)*KNpmly) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmlz1, hostKpmlz1, sizeof(float)*KNpmlz) );
    CHECK_ERROR( cudaMemcpyToSymbol(Kpmlz2, hostKpmlz2, sizeof(float)*KNpmlz) );
  }
  CHECK_ERROR( cudaSetDevice(0) );
  //-----------------------------------------------------------------------------------//

/*  parsHost.sensors->push_back(Sensor("Ez",(X0+Rreson)/dx+1,Y0/dy,Z0/dz));
  parsHost.sensors->push_back(Sensor("Ez",(X0-Rreson)/dx-1,Y0/dy,Z0/dz));
  parsHost.sensors->push_back(Sensor("Ez",X0/dx,(Y0+Rreson)/dy+1,Z0/dz));
  parsHost.sensors->push_back(Sensor("Ez",X0/dx,(Y0-Rreson)/dy-1,Z0/dz));*/

//  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
//  CHECK_ERROR(cudaMalloc3DArray(&index_texArray, &channelDesc, make_cudaExtent(parsHost.IndNz,parsHost.IndNy,parsHost.IndNx)));
  cudaChannelFormatDesc channelDesc;
}
void set_texture(const int ix=0){
}

int print_help() {
  printf("using: ./DFmxw [--help] [--zoom \"1. 1. 1.\"] [--step \"1. 1. 1.\"] [--box \"1. 1. 1.\"] [--mesh \"200. 200. 200.\"] [--Dmesh 5.] [--drop_dir \".\"] [--bkgr_col \"0.1 0.1 0.1\"] [--mesh_col \"0.8 0.8 0.2\"] [--box_col \"1. 1. 1.\"] [--sensor \"1 1 1\"]\n");
  printf("  --zoom\tмасштабный фактор, действует на 2D режим и размер окна, [1. 1. 1.];\n");
  printf("  --box \tкоррекция пропорций размера бокса в 3D режиме, [1. 1. 1.];\n");
  printf("  --step \tшаги между точками, действует только на тики, [1. 1. 1.];\n");
  printf("  --mesh\tрасстояние между линиями сетки в боксе по координатам в ячейках (до коррекции), [200. 200. 200.];\n");
  printf("  --Dmesh\tширина линии сетки в пикселях (со сглаживанием выглядит несколько уже), [5.];\n");
  printf("  --drop_dir\tимя директории, в которую будут сохраняться различные файлы, [.];\n");
  printf("  --bkgr_col\tцвет фона, [0.1 0.1 0.1];\n");
  printf("  --mesh_col\tцвет линий сетки, [0.8 0.8 0.2];\n");
  printf("  --box_col\tцвет линий бокса, [1.0 1.0 1.0];\n");
  printf("  --sensor\tкоординаты сенсора, можно задавать несколько сенсоров;\n");
  return 0;
}
void read_float3(float* v, char* str) {
  for(int i=0; i<3; i++) { v[i] = strtof(str, &str); str++; }
}
float read_float(char* str) {
  return atof(str);
}
void add_sensor(int ix, int iy, int iz);

bool help_only=false, test_only=false;
int Tsteps=10*Ntime;
int _main(int argc, char** argv) {
  #ifdef MPI_ON
  MPI_Init(&argc,&argv);
  #endif
  argv ++; argc --;
  im3DHost.reset();
  while(argc>0 && strncmp(*argv,"--",2)==0) {
    if(strncmp(*argv,"--help",6)==0) return print_help();
    else if(strcmp(*argv,"--test")==0) { test_only = true; argv ++; argc --; continue; }
    if(strcmp(*argv,"--box")==0) read_float3(im3DHost.BoxFactor, argv[1]);
    else if(strcmp(*argv,"--test")==0) test_only = true;
    else if(strcmp(*argv,"--mesh")==0) read_float3(im3DHost.MeshBox, argv[1]);
    else if(strcmp(*argv,"--Dmesh")==0) im3DHost.Dmesh=read_float(argv[1]);
    else if(strcmp(*argv,"--zoom")==0) read_float3(im3DHost.Dzoom, argv[1]);
    else if(strcmp(*argv,"--step")==0) read_float3(im3DHost.step, argv[1]);
    else if(strcmp(*argv,"--bkgr_col")==0) read_float3(im3DHost.bkgr_col, argv[1]);
    else if(strcmp(*argv,"--mesh_col")==0) read_float3(im3DHost.mesh_col, argv[1]);
    else if(strcmp(*argv,"--box_col")==0) read_float3(im3DHost.box_col, argv[1]);
    else if(strcmp(*argv,"--drop_dir")==0) strcpy(im3DHost.drop_dir,argv[1]);
    else if(strcmp(*argv,"--sensor")==0) { float v[3]; read_float3(v, argv[1]); add_sensor(v[0], v[1], v[2]); }
    else { printf("Illegal parameters' syntax notation\n"); return print_help(); }
    //else if(strcmp(*argv,"--")==0) read_float3(im3DHost., argv[1]);
    printf("par: %s; vals: %s\n", argv[0], argv[1]);
    argv +=2; argc -=2;
  };
  im2D.get_device(3,0);
  if(test_only) printf("No GL\n");
  else printf("With GL\n");
try {
  if(type_diag_flag>=1) printf("Настройка опций визуализации по умолчанию\n");
  //imHost.reset();
  cudaTimer tm; tm.start();
  parsHost.set();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  copy2dev( parsHost, pars );
  copy2dev( shotpoint, src );
  shotpoint.check();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  init();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  init_index();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  set_texture();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  copy2dev( parsHost, pars );
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );

  if(test_only) {
    for(int i=0; i<Tsteps/Ntime; i++) {
//    while(true) {
      tm.start();
      calcStep();
//      double tCpu=tm.stop();
//      printf("run time: %.2f msec, %.2f Gcells/sec\n", tCpu, 1.e-6*Ntime*Nx*Ny*Nz/tCpu);
//return 0;
    }
    return 0;
  }

  tm.start();
  parsHost.reset_im();
  im3DHost.reset(parsHost.arr4im);
  copy2dev( parsHost, pars );
  CreateShowTexModel();
  CHECK_ERROR( cudaMemset(parsHost.arr4im.Arr3Dbuf,0,((long long int)Nx)*Ny*Nz*sizeof(ftype)) ); mxw_draw<<<dim3((USE_UVM==2)?Np:Ns,Na),NT>>>(parsHost.arr4im.Arr3Dbuf);
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  im2D.get_device(3,0);
  im2D.init_image(argc,argv, im3DHost.bNx, im3DHost.bNy, "im3D");
  im3DHost.init3D(parsHost.arr4im); im3DHost.iz0=Nx-1; im3DHost.key_func('b',0,0);

  if(type_diag_flag>=1) printf("Настройка GLUT и запуск интерфейса\n");
  glutIdleFunc(idle_func);
  glutKeyboardFunc(key_func);
  glutMouseFunc(mouse_func);
  glutMotionFunc(motion_func);
  glutDisplayFunc(draw_func);
  if(type_diag_flag>=0) printf("Init cuda device: %.1f msec\n", tm.stop());
  glutMainLoop();
} catch(...) {
  printf("Возникла какая-то ошибка.\n");
}
  parsHost.clear();
  return -1;
}
int main(int argc, char** argv) {
  return _main(argc,argv);
}

float get_val_from_arr3D(int ix, int iy, int iz) {
  Arr3D_pars& arr=parsHost.arr4im;
  if(arr.inCPUmem) return arr.Arr3Dbuf[arr.get_ind(ix,iy,iz)];
  float res=0.0;
  if(arr.inGPUmem) exit_if_ERR(cudaMemcpy(&res, arr.get_ptr(ix,iy,iz), sizeof(float), cudaMemcpyDeviceToHost));
  return res;
}
Arr3D_pars& set_lim_from_arr3D() {
  Arr3D_pars& arr=parsHost.arr4im;
  if(arr.inCPUmem) arr.reset_min_max();
  if(arr.inGPUmem) {
    float* fLims=0,* fLimsD=0;
    exit_if_ERR(cudaMalloc((void**) &fLimsD, 2*Ny*sizeof(float)));
    calc_limits<<<Ny,Nz>>>(arr.Arr3Dbuf, fLimsD);
    fLims=new float[2*Ny];
    exit_if_ERR(cudaMemcpy(fLims, fLimsD, 2*Ny*sizeof(float), cudaMemcpyDeviceToHost));
    exit_if_ERR(cudaFree(fLimsD));
    arr.fMin = fLims[0]; arr.fMax = fLims[1];
    for(int i=0; i<Ny; i++) {
      if(fLims[2*i  ]<arr.fMin) arr.fMin = fLims[2*i  ];
      if(fLims[2*i+1]>arr.fMax) arr.fMax = fLims[2*i+1];
    }
    delete fLims;
  }
  return arr;
}
