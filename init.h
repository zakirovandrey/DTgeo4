#ifndef _INIT_H
#define _INIT_H
#include "params.h"

__global__ void __launch_bounds__(Nz) set_data() {
  int iz = threadIdx.x; 
  DiamondRag* p=&pars.get_plaster(blockIdx.x,blockIdx.y);
  const int Npls=2*NDT*NDT;
  for (int Ragdir=0; Ragdir<2; Ragdir++) {
    for(int idom=0; idom<Npls; idom++) {
      int shx=-NDT+idom/NDT+idom%NDT;
      int shy=+NDT-idom/NDT+idom%NDT;
      if(Ragdir) shy=0+idom/NDT-idom%NDT;
      int ix = blockIdx.x*2*NDT+shx+4;
      int iy = blockIdx.y*2*NDT+shy+2;
    
      if(Ragdir==0) {
        if(idom%2==0)         p->Si[idom/2].duofld[0][iz].x=0;   //Sx
        if(idom%2==0)         p->Si[idom/2].duofld[0][iz].y=0;   //Sy
        if(idom%2==0)         p->Si[idom/2].duofld[1][iz].x=0;   //Sz
        if(idom%2==0) { ix++; p->Si[idom/2].duofld[1][iz].y=0; } //Ty
        if(idom%2==1)         p->Si[idom/2].duofld[2][iz].x=0;   //Tz
        if(idom%2==1)         p->Si[idom/2].duofld[2][iz].y=0;   //Tx
      } else {
        if(idom%2==0) {
          const int Lenx=Nx*2, Leny=Ny*2, Lenz=Nv;
          ftype env = (1.0+cos((ix-Lenx/2)/50.*M_PI))*(1.0+cos((iy-Leny/3)/50.*M_PI))*(1.0+cos((iz-Lenz/2)/50.*M_PI));
          if(abs(ix-Lenx/2)>=50 || abs(iy-Leny/3)>=50 || abs(iz-Lenz/2)>=50) env=0.0f;
          p->Vi[idom/2].trifld.one[iz]   = 0*env*cos((iy-Leny/3)*0.06*M_PI); //Vy
        }
        if(idom%2==1) {      
          const int Lenx=Nx*2, Leny=Ny*2, Lenz=Nv;
          ftype env = (1.0+cos((ix-Lenx/2)/50.*M_PI))*(1.0+cos((iy-Leny/3)/50.*M_PI))*(1.0+cos((iz-Lenz/2)/50.*M_PI));
          if(abs(ix-Lenx/2)>=50 || abs(iy-Leny/3)>=50 || abs(iz-Lenz/2)>=50) env=0.0f;
          p->Vi[idom/2].trifld.two[iz].x = 0*env*cos((ix-Lenx/3)*0.06*M_PI);   //Vx
        }
        if(idom%2==1) {
          const int Lenx=Nx*2, Leny=Ny*2, Lenz=Nv;
          ftype env = (1.0+cos((ix-Lenx/2)/50.*M_PI))*(1.0+cos((iy-Leny/3)/50.*M_PI))*(1.0+cos((iz-Lenz/2)/50.*M_PI));
          if(abs(ix-Lenx/2)>=50 || abs(iy-Leny/3)>=50 || abs(iz-Lenz/2)>=50) env=0.0f;
          p->Vi[idom/2].trifld.two[iz].y = env*cos((iz-Lenz/3)*0.06*M_PI); //Vz 
        }
      }
    }
  }
}

void init(){
//  set_data<<<dim3(Ns,Na),Nv>>>();
  cudaDeviceSynchronize();
  CHECK_ERROR( cudaGetLastError() );
}

void drop(){
/*  host_cells = new Cell[Nx*Ny];
  CHECK_ERROR( cudaMemcpy( host_cells, parsHost.cells, Nx*Ny*sizeof(Cell), cudaMemcpyDeviceToHost ) );
  FILE* file = fopen("res.arr", "w");
  if(file==NULL) perror("Cannot open file res.arr\n");
  int nil=0, dim=3, szT=4;
  fwrite(&nil, 4, 1, file);
  fwrite(&dim, 4, 1, file);
  fwrite(&szT, 4, 1, file);
  fwrite(&Nx , 4, 1, file);
  fwrite(&Ny , 4, 1, file);
  fwrite(&Nz , 4, 1, file);
//  for(int i=0; i<Nx*Ny; i++) fwrite(host_cells[i].Ex, sizeof(float), Nz, file);
  fclose(file);*/
}

#endif //_INIT_H
