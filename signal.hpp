#ifndef SIGNAL_HPP
#define SIGNAL_HPP
#ifdef MPI_ON
#include <mpi.h>
#endif

#include "lambda_func.hpp"
#define S L7
#include "signal.h"
namespace TFSF{
//  const float kappa = Vp_*Vp_*Rho, lambda=(Vp_*Vp_-2*Vs_*Vs_)*Rho, mu=Vs_*Vs_*Rho, r0src=0, dvp=1./Vp_;
};

__constant__ TFSFsrc src;
TFSFsrc shotpoint;

void TFSFsrc::set(const double _Vp, const double _Vs, const double _Rho) {
    Vp = _Vp; Vs = _Vs; Rho=_Rho;
    kappa = _Vp*_Vp*_Rho; lambda=(_Vp*_Vp-2*_Vs*_Vs)*_Rho; mu=_Vs*_Vs*_Rho; r0src=0; dvp=1./_Vp;
    dRho=1.0/_Rho;
    
    w0 = gauss_waist*Vp/F0;
    Rh=srcXv*0.95;
    NastyaF0=F0*1;//M_PI;
    Ampl=1.0;
    r0=1.0*Vp/F0;
    r1=r0+da;
    rstart=r0;
    delay = (sqrt(r1*r1+Rh*Rh)-sqrt(r0*r0+Rh*Rh))/Vp;
}
void TFSFsrc::check(){
    int node=0, Nprocs=1;
    #ifdef MPI_ON
    MPI_Comm_rank (MPI_COMM_WORLD, &node);
    MPI_Comm_size (MPI_COMM_WORLD, &Nprocs);
    #endif
    if (node!=0) return;
    printf("TF/SF source: Box(s,v,a) |%g-%g|x|%g-%g|x|%g-%g|\n", BoxMs,BoxPs, BoxMv,BoxPv, BoxMa,BoxPa);
    printf("TF/SF source: Shotpoint(s,v,a) = %g,%g,%g\n", srcXs, srcXv, srcXv);
    printf("TF/SF source: Vp,Vs,Rho = %g,%g,%g\n", Vp,Vs,Rho);
    printf("TF/SF source: F0 = %g, tStop=%g\n", F0,tStop);
    printf("TF/SF stops after %d steps\n", int(tStop/dt));
    printf("Maximal velocity=%g\n", V_max);
    if(dt*sqrt(1/(dx*dx)+1/(dy*dy)+1/(dz*dz))>6./7./V_max) { printf("Error: Courant condition is not satisfied (V_max=%g)\n", V_max); exit(-1); }
}

struct SphereTFSF{
  float x,y,z, d1r, phase, sk, v_r;
  static const float A=1.0e5;
  __device__ SphereTFSF(const float t, const int _x, const int _y, const int _z) {
    using namespace TFSF;
    const float dxs=dx*0.5,dxa=dy*0.5,dxv=dz*0.5;
    
    const int ks=2*NDT;
    const int realx = _x;
    //const int realx = (_x+ks*pars.GPUx0+ks*Ns-ks*pars.wleft)%(ks*Ns)+pars.wleft*ks;
    x = dxs*realx-src.srcXs; y=dxa*_y-src.srcXa; z = dxv*_z-src.srcXv;
    float r = radius(x,y,z); phase =src.F0*((t-src.start)*dt-(r-src.r0src)*src.dvp); d1r = 1./r;
  }
  __device__ inline void set_sk_vr() { v_r = Vr(); sk = Vrs(); }
  __device__ inline float Vrs() { //skobka (dpr/dr-pr/r) int po t
    using namespace TFSF;
    return A*d1r*( S<2>(phase)*src.F0*src.F0*src.dvp*src.dvp + 3.0*S<1>(phase)*src.F0*src.dvp*d1r + 3.0*S<0>(phase)*d1r*d1r)/src.F0;
  }
  __device__ inline float vrs() { //skobka (dpr/dr-pr/r)
    using namespace TFSF;
    return A*d1r*( S<3>(phase)*src.F0*src.F0*src.dvp*src.dvp + 3.0*S<2>(phase)*src.F0*src.dvp*d1r + 3.0*S<1>(phase)*d1r*d1r);
  }
  __device__ inline float  vr() {
    using namespace TFSF;
    return A*d1r*(S<2>(phase)*src.F0*src.dvp+S<1>(phase)*d1r);
  }
  __device__ inline float  Vr() {
    using namespace TFSF;
    return -A*d1r*(S<1>(phase)*src.F0*src.dvp+S<0>(phase)*d1r)/src.F0;
  }
  __device__ inline float getVx() { return x*vr()*d1r; }
  __device__ inline float getVy() { return y*vr()*d1r; }
  __device__ inline float getVz() { return z*vr()*d1r; }
  __device__ inline float getSx() { set_sk_vr(); return (src.kappa *(v_r+sk*x*x*d1r) + src.lambda*(v_r+sk*y*y*d1r) + src.lambda*(v_r+sk*z*z*d1r))*d1r; }
  __device__ inline float getSy() { set_sk_vr(); return (src.lambda*(v_r+sk*x*x*d1r) + src.kappa *(v_r+sk*y*y*d1r) + src.lambda*(v_r+sk*z*z*d1r))*d1r; }
  __device__ inline float getSz() { set_sk_vr(); return (src.lambda*(v_r+sk*x*x*d1r) + src.lambda*(v_r+sk*y*y*d1r) + src.kappa *(v_r+sk*z*z*d1r))*d1r; }
  __device__ inline float getTx() { return src.mu*(2.0*Vrs()*y*z*d1r*d1r); }
  __device__ inline float getTy() { return src.mu*(2.0*Vrs()*z*x*d1r*d1r); }
  __device__ inline float getTz() { return src.mu*(2.0*Vrs()*x*y*d1r*d1r); }
};
__device__ __noinline__ float SrcTFSF_Sx(const int s, const int v, const int a,  const float tt){
  SphereTFSF src(tt, s,a,v); return src.getSx();
}
__device__ __noinline__ float SrcTFSF_Sy(const int s, const int v, const int a,  const float tt){
  SphereTFSF src(tt, s,a,v); return src.getSy();
  if(fabsf(s-Nx/2)*dx>=30*dx || fabsf(v-Nz/2)*dz>=30*dz) return 0;
//  if(int(a)==0 && int(s)==Nx/2 && int(v)==Nz/2) printf("it=%d, val=%g\n",it,sinf(a*dy*2*M_PI+it*dt*2*M_PI));
  return (1.0+cosf((s-Nx/2)/30.*M_PI))*(1.0+cosf((v-Nz/2)/30.*M_PI))*sinf(a*dy*2*M_PI-tt*dt*2*M_PI);
}
__device__ __noinline__ float SrcTFSF_Sz(const int s, const int v, const int a,  const float tt){
  SphereTFSF src(tt, s,a,v); return src.getSz();
}
__device__ __noinline__ float SrcTFSF_Tx(const int s, const int v, const int a,  const float tt){
  SphereTFSF src(tt, s,a,v); return src.getTx();
}
__device__ __noinline__ float SrcTFSF_Ty(const int s, const int v, const int a,  const float tt){
  SphereTFSF src(tt, s,a,v); return src.getTy();
}
__device__ __noinline__ float SrcTFSF_Tz(const int s, const int v, const int a,  const float tt){
  SphereTFSF src(tt, s,a,v); return src.getTz();
}
__device__ __noinline__ float SrcTFSF_Vx(const int s, const int v, const int a,  const float tt){
  SphereTFSF src(tt, s,a,v); return src.getVx();
  return 0;
}
__device__ __noinline__ float SrcTFSF_Vy(const int s, const int v, const int a,  const float tt){
  SphereTFSF src(tt, s,a,v); return src.getVy();
  if(fabsf(s-Nx/2)*dx>=30*dx || fabsf(v-Nz/2)*dz>=30*dz) return 0;
  return (1.0+cosf((s-Nx/2)/30.*M_PI))*(1.0+cosf((v-Nz/2)/30.*M_PI))*sinf(a*dy*2*M_PI-tt*dt*2*M_PI);
  return 0;
}
__device__ __noinline__ float SrcTFSF_Vz(const int s, const int v, const int a,  const float tt){
  SphereTFSF src(tt, s,a,v); return src.getVz();
}

__device__ __noinline__ bool inSF(const int _s, const int _a, const int _v) {
  using namespace TFSF;
//  if(v==Nz/2 && a==Ny/2) printf("%g   %g   %g\n",tfsfSm, s, tfsfSp);
  const int ks=2*NDT;
  const int reals = _s;
  //const int reals = (_s+ks*pars.GPUx0+ks*Ns-ks*pars.wleft)%(ks*Ns)+pars.wleft*ks;
//  if(_v==128 && _a==10) printf("checking inSF _s=%d|%d reals=%d|%d\n", _s, _s/(2*NDT), reals, reals/(2*NDT));
  float s = reals*0.5*dx, a=_a*0.5*dy, v=_v*0.5*dz;
//  if(_v==128) printf("checking inSF s/dx/NDT=%d\n", int(s/dx/NDT));
  return (s>src.BoxMs && s<src.BoxPs && a>src.BoxMa && a<src.BoxPa && v>src.BoxMv && v<src.BoxPv); 
}


__device__ inline float EnvelopeR(float x, float y) {
  float r2=x*x+y*y; if(r2>=src.Rh*src.Rh) return 0.0;
  /*Cosine,Cosine/2*/
  const double Kx=0.5*M_PI/src.Rh, Ky=0.5*M_PI/src.Rh; return
  //cos(Kx*x)*cos(Ky*y);
  0.5*(1.0+cosf(2.0*Kx*sqrtf(x*x+y*y)));
  /*Gauss*/ //const double Kr=1.0/(Rh*Rh); double r2=Kr*(x*x+y*y); return exp(-r2)-sqrt(r2)*(exp(-9.)/3.0);
  /*Boom*/  //double v=getBoomDistance()/getBoomDistance(x,y); return v*v*v*0.5*(1.0+cos((M_PI/Rh)*sqrt(r2)));
}
__device__ inline float Boom(float x, float y, float z, float t, bool S) {
  float rr = radius(x,y,z);
  float arg = t-(rr-src.rstart)/src.Vp;
  float Prs;
  if(S) Prs = (src.Ampl/rr)*( L7<3>(arg)*src.NastyaF0*src.NastyaF0 / (src.Vp*src.Vp) + 3*L7<2>(arg)*src.NastyaF0 / (src.Vp*rr) +3*L7<1>(arg)              /(rr*rr));   
  else  Prs = (src.Ampl/rr)*( L7<2>(arg)*src.NastyaF0              / (src.Vp*src.Vp) + 3*L7<1>(arg)              / (src.Vp*rr) +3*L7<0>(arg)/src.NastyaF0 /(rr*rr));   
  float Pd2r = -(src.Ampl/rr)*( 6*L7<0>(arg) /(rr*rr*rr)+6*L7<1>(arg)*src.NastyaF0 / (src.Vp*rr*rr)+3*L7<2>(arg)*src.NastyaF0*src.NastyaF0 / (src.Vp*src.Vp*rr)+L7<3>(arg)*src.NastyaF0*src.NastyaF0*src.NastyaF0 / (src.Vp*src.Vp*src.Vp) )/src.NastyaF0;

  if(S) return dt*src.Vs*src.Vs*Prs * z / (rr*rr);  
  else return dt*( (-src.Ampl*z*src.NastyaF0)*(src.NastyaF0*L7<3>(arg)/src.Vp+L7<2>(arg)/rr)/(rr*rr)-2*src.Vs*src.Vs*(z/(rr*rr*rr))*((rr*rr-z*z)*Pd2r+Prs*(3*z*z-rr*rr)/rr) );  
}

__device__ __noinline__ float SrcSurf_Vy(const int s, const int v, const int a,  const float tt){
  const float dxs=dx*0.5,dxa=dy*0.5,dxv=dz*0.5;
  float x = dxs*s-src.srcXs, z=dxa*a-src.srcXa, y = dxv*v-src.srcXv;
  //gauss pulse
  float arg = (tt-src.start)*dt; 
  float Th=3./src.F0;
  if (arg>2*Th) return 0;
  return dt*src.Ampl/(src.w0*src.w0)*src.Vp*src.Vp/(src.F0*src.F0)*__expf(-(x*x+y*y)/(src.w0*src.w0))* //L7shtsht(arg);
        //Ricker wavelet
        (1-2*M_PI*M_PI*src.F0*src.F0*(arg-Th*0.5)*(arg-Th*0.5))*__expf(-M_PI*M_PI*src.F0*src.F0*(arg-Th*0.5)*(arg-Th*0.5));
       //  else return EnvelopeR(x,y)*( sqrt((r0*r0+Rh*Rh)/(r1*r1+Rh*Rh))*r1/r0*Boom(x,y,r0,(it-0.5)*dt-delay,false) - Boom(x,y,r1,(it-0.5)*dt,false) )/Rho;
}


#undef S
#endif
