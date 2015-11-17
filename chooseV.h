//#define MINBLOCKS (FTYPESIZE*Nv*20>0xc000)?1:2
#define MINBLOCKS 1
__global__ void __launch_bounds__(Nz,MINBLOCKS) torreD0 (int ix, int y0, int Nt, int t0); 
__global__ void __launch_bounds__(Nz,MINBLOCKS) torreD1 (int ix, int y0, int Nt, int t0); 
__global__ void __launch_bounds__(Nz) torreS0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreS1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreI0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreI1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreX0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreX1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreB0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreB1 (int ix, int y0, int Nt, int t0);
/*__global__ void __launch_bounds__(Nz) torreDD0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreDD1(int ix, int y0, int Nt, int t0);
*/
__global__ void __launch_bounds__(Nz) torreTFSF0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreTFSF1(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreITFSF0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) torreITFSF1(int ix, int y0, int Nt, int t0);

__global__ void __launch_bounds__(Nz) PMLStorreD0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreD1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreS0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreS1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreI0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreI1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreX0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreX1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreB0 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreB1 (int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreTFSF0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreTFSF1(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreITFSF0(int ix, int y0, int Nt, int t0);
__global__ void __launch_bounds__(Nz) PMLStorreITFSF1(int ix, int y0, int Nt, int t0);

