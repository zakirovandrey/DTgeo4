const int NDT=3;
typedef float ftype;

#define gridNx 1500
#define gridNy 256
#define gridNz 600
const int GridNx=gridNx;
const int GridNy=gridNy;
const int GridNz=gridNz;

#define NDev 2

//#define USE_AIVLIB_MODEL
//#define MPI_ON
//#define TEST_RATE
#define USE_WINDOW
//#define COFFS_DEFAULT
//#define CLICK_BOOM

const int Npmlx=2*5;//2*24;
const int Npmly=5;//24;
const int Npmlz=2*16;//128;

const ftype ds=25./3, da=25./3, dv=25./3, dt=0.5;

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

