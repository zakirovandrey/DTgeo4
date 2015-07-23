#ifndef SIGNAL_H
#define SIGNAL_H
__device__ __noinline__ float SrcTFSF_Sx(const int s, const int v, const int a,  const float tt);
__device__ __noinline__ float SrcTFSF_Sy(const int s, const int v, const int a,  const float tt);
__device__ __noinline__ float SrcTFSF_Sz(const int s, const int v, const int a,  const float tt);
__device__ __noinline__ float SrcTFSF_Tx(const int s, const int v, const int a,  const float tt);
__device__ __noinline__ float SrcTFSF_Ty(const int s, const int v, const int a,  const float tt);
__device__ __noinline__ float SrcTFSF_Tz(const int s, const int v, const int a,  const float tt);
__device__ __noinline__ float SrcTFSF_Vx(const int s, const int v, const int a,  const float tt);
__device__ __noinline__ float SrcTFSF_Vy(const int s, const int v, const int a,  const float tt);
__device__ __noinline__ float SrcTFSF_Vz(const int s, const int v, const int a,  const float tt);
__device__ __noinline__ bool inSF(const int _s, const int _a, const int _v) ;

__device__ __forceinline__ float SrcSurf_Sx(const int s, const int v, const int a,  const float tt) {return 0;};
__device__ __forceinline__ float SrcSurf_Sy(const int s, const int v, const int a,  const float tt) {return 0;};
__device__ __forceinline__ float SrcSurf_Sz(const int s, const int v, const int a,  const float tt) {return 0;};
__device__ __forceinline__ float SrcSurf_Tx(const int s, const int v, const int a,  const float tt) {return 0;};
__device__ __forceinline__ float SrcSurf_Ty(const int s, const int v, const int a,  const float tt) {return 0;};
__device__ __forceinline__ float SrcSurf_Tz(const int s, const int v, const int a,  const float tt) {return 0;};
__device__ __forceinline__ float SrcSurf_Vx(const int s, const int v, const int a,  const float tt) {return 0;}
__device__ float SrcSurf_Vy(const int s, const int v, const int a,  const float tt) ; 
__device__ __forceinline__ float SrcSurf_Vz(const int s, const int v, const int a,  const float tt) {return 0;}


extern __constant__ TFSFsrc src;
extern  TFSFsrc shotpoint;

#endif
