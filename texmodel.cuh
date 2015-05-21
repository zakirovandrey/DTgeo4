#ifndef TEXMODEL_CU
#define TEXMODEL_CU
#include <stdio.h>
#include "cuda_math.h"
#ifdef USE_AIVLIB_MODEL
#include "spacemodel/include/access2model.hpp"
#endif

#define USE_TEX_REFS
#define MAX_TEXS 10

//1 -- many textures, 2 -- one texture, 0 -- one texture reference
#define TEX_MODEL_TYPE 0
#define H_MAX_SIZE (USHRT_MAX-1)
typedef ushort2 htype;

//const float texStretchX=1.0/(2*Ns*NDT);
//const float texStretchY=1.0/(2*Nz);
const float texStretchH=1.0/H_MAX_SIZE;
extern __constant__ float2 texStretch[MAX_TEXS];
extern __constant__ float2 texShift[MAX_TEXS];
extern __constant__ float2 texStretchShow;
extern __constant__ float2 texShiftShow;
struct __align__(16) ModelRag{
  #if TEX_MODEL_TYPE==1
  int I[4][Nz];
  #endif
  //htype h[NDT*NDT*7+1][Nz];
  htype h[32][Nz];
  int3 check_bounds(const int3 &v) {
    int3 ret=v;
    if(v.x<0) ret.x=0; else if(v.x>=Np*NDT*2) ret.x=Np*NDT*2-1;
    if(v.y<0) ret.y=0; else if(v.y>=Nz*2    ) ret.y=Nz*2-1;
    if(v.z<0) ret.z=0; else if(v.z>=Na*NDT*2) ret.z=Na*NDT*2-1;
    return ret;
  }
  void set(int x, int y);
};

#ifdef USE_TEX_REFS
extern texture<float2, cudaTextureType3D, cudaReadModeElementType> layerRefS;
extern texture<float , cudaTextureType3D, cudaReadModeElementType> layerRefV;
extern texture<float , cudaTextureType3D, cudaReadModeElementType> layerRefT;
#endif
struct ModelTexs{
  bool ShowTexBinded;
  int Ntexs;
  int3* texN;
  int* tex0;
  float* texStep;
  //cudaTextureObject_t layerS[MAX_TEXS], layerV[MAX_TEXS], layerT[MAX_TEXS];
  cudaTextureObject_t TexlayerS[NDev], TexlayerV[NDev], TexlayerT[NDev];
  cudaTextureObject_t *layerS[NDev], *layerV[NDev], *layerT[NDev];
  cudaTextureObject_t *layerS_host[NDev], *layerV_host[NDev], *layerT_host[NDev];
  float2** HostLayerS; float **HostLayerV, **HostLayerT;
  cudaArray** DevLayerS[NDev], **DevLayerV[NDev], **DevLayerT[NDev];
  void init();
  void copyTexs(const int x1dev, const int x2dev, const int x1host, const int x2host, cudaStream_t& streamCopy);
  void copyTexs(const int xdev, const int xhost, cudaStream_t& streamCopy);
};

#include "signal.h"
namespace defCoff {
//  const float Vp=TFSF::Vp_, Vs=TFSF::Vs_, rho=TFSF::Rho,drho=TFSF::dRho;
  const float Vp=3.0, Vs=1.5, rho=1.0,drho=1/rho;
};

#endif
