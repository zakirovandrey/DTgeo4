const int NDT=3;
typedef float ftype;

#define gridNx 1350
#define gridNy 416
#define gridNz 984//1152

//#define ANISO_TR 2

#define NDev 2

//#define USE_AIVLIB_MODEL
//#define MPI_ON
//#define TEST_RATE
#define USE_WINDOW
//#define COFFS_DEFAULT
//#define CLICK_BOOM

#define DROP_ONLY_V

#ifndef NS
#define NS 25
const int Np=gridNx/3;
const int GridNx=gridNx;
#else
const int Np=NS*5;
const int GridNx=3*Np;
#endif
#ifndef NA
const int GridNy=gridNy;
#else 
const int GridNy=NV;
#endif
#ifndef NV
const int GridNz=gridNz;
#else
const int GridNz=3*NA;
#endif

const int Npmlx=2*5;//2*24;
const int Npmly=5;//24;
const int Npmlz=2*16;//128;

const ftype ds=25./2, da=6.95, dv=25./2, dt=2./3.;

extern struct TFSFsrc{
  ftype Vp, Vs, Rho;
  ftype kappa, lambda, mu, r0src, dvp;
  ftype dRho;
  ftype F0;
  ftype tStop;
  ftype srcXs, srcXa, srcXv;
  ftype BoxMs, BoxPs; 
  ftype BoxMa, BoxPa; 
  ftype BoxMv, BoxPv;
  ftype sphR;
  ftype V_max;
  int start;
  //----------surface source parameters-----------//
  ftype NastyaF0,Ampl, w0, gauss_waist;
  ftype r0,r1,rstart;
  ftype delay;
  ftype Rh;
  //----------------------------------------------//
  void set(const double, const double, const double);
  void check();
} shotpoint;

