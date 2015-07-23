#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string>
#include <omp.h>
#include <aivlib/sphereT.hpp>
#include <aivlib/mystream.hpp>
#include <aivlib/vectorD.hpp>
#include <aivlib/multiarrayT.hpp>
#include <aivlib/arrayTD.hpp>

using namespace aiv;
const int tSkip=4;
const int Nt = 4000;
const int xStep=3, yStep=1, yShift=0;
int Nx,Nw; 
float t,x1,x2;
int main(int argc, char** argv){
//  int fs=atoi(argv[1]);
  float val;
  std::string dir="./";
  for(int nfld=1; nfld<argc; nfld++) {
    char* fname;
    //sprintf(fname, "%s/%s_node0.arr", dir.c_str(), fields[nfld].c_str()); FILE* fraw = fopen (fname, "rb"); if(fraw==NULL) perror("File open error\n");
    fname = argv[nfld]; FILE* fraw = fopen (fname, "rb"); if(fraw==NULL) perror("File open error\n");
    printf("converting file %s", fname);
    uint32_t *drop_cells;
    if( fread(&Nx, sizeof(Nx), 1, fraw)!=1 ) perror("Nx Reading error\n");
    if( fread(&Nw, sizeof(Nw), 1, fraw)!=1 ) perror("Nw Reading error\n");
    printf("  Nx=%d Nw=%d\n",Nx,Nw);
    drop_cells = new uint32_t[Nx*Nw];
    if( fread(drop_cells, sizeof(uint32_t), Nx*Nw, fraw)!=Nx*Nw ) perror("drop_cells Reading error\n");

    array<float,3> arr; 
    indx<3> nc; nc[0]=Nt/tSkip; nc[1]=Nw*32/yStep; nc[2]=Nx/xStep; arr.init(nc);
    printf("x,y,z=%d %d %d\n", nc[0], nc[1], nc[2]);
    for(int z=0;z<nc[2];z++) for(int y=0;y<nc[1];y++) for(int x=0;x<nc[0];x++) arr[Indx(x,y,z)] = 0;
    printf("ok\n");
    int cur_pos = ftell(fraw); fseek(fraw,0,SEEK_END); int end_pos=ftell(fraw); fseek(fraw,cur_pos,SEEK_SET);
    while(ftell(fraw)!=end_pos) {
      if( fread(&t , sizeof(float), 1, fraw)!=1 ) { printf("(pos %d) ", ftell(fraw)); perror("drop_cells Reading error t \n"); }
      if( fread(&x1, sizeof(float), 1, fraw)!=1 ) { printf("(pos %d) ", ftell(fraw)); perror("drop_cells Reading error x1\n"); }
      if( fread(&x2, sizeof(float), 1, fraw)!=1 ) { printf("(pos %d) ", ftell(fraw)); perror("drop_cells Reading error x2\n"); }
      //printf("t=%g x1=%g x2=%g\n",t,x1,x2);
      for(int ix=x1; ix<x2; ix++) for(int iwarp=0; iwarp<Nw; iwarp++) for(int ith=0; ith<32; ith++) {
        //printf("ix=%d iwarp=%d ith=%d drop_cells=%d\n",ix,iwarp,ith,drop_cells[(ix+Nx)%Nx*Nw+iwarp]);
        if(ix<0) continue;
        if(drop_cells[(ix+Nx)%Nx*Nw+iwarp] & 1<<ith) {
          //printf("ix =%d iwarp=%d ith=%d t=%d val=%g\n", ix,iwarp,ith,int(t),val);
          if( fread(&val , sizeof(float), 1, fraw)!=1 ) { printf("(pos %d) ", ftell(fraw)); perror("drop_cells Reading error val\n"); }
          if(int(t)%tSkip!=0) continue;
          if(int(t/tSkip)>=Nt/tSkip)  { continue; printf("Nt exceeded\n"); /*exit(-1);*/ }
          //printf("ix=%d v=%d, tt=%d\n",(ix+Nx)%Nx,iwarp*32+ith, int(t/tSkip));
          arr[Indx(int(t/tSkip),(iwarp*32+ith+yShift)/yStep,(ix/xStep+Nx/xStep)%(Nx/xStep))] = val;
        }
      }
    }
    char savefname[256];
    sprintf(savefname,"%s/seismo%s", dir.c_str(), fname); printf("saving %s...\n",savefname); arr.dump(Ofile(savefname).self); printf("ok\n");
    delete[] drop_cells;
    arr.clean();
  }
  return 0;
}

