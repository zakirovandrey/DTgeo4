#define WSIZE 32
const int Nwarps=Nz/WSIZE;
extern __constant__ uint32_t drop_cells[Ns*NDT*Nwarps];
struct SeismoDrops {
  ftype* channel[9]; // SxyzTxyzVxyz
  ftype* channelHost[9]; // SxyzTxyzVxyz
  ftype** channelAddr;
  ftype* channelAddrHost[9];
  ptrdiff_t offset[9];
  uint32_t *drop_cellsHost;
  #ifdef MPI_ON
  MPI_File file[9]; MPI_Status status; MPI_Info info;
  #else
  FILE* file[9];
  #endif
  static const int channelHostLength=Ntime*Np*NDT*100; // future think about size
  static const int channelDevLength=Ntime*Ns*NDT*100; // future think about size
  int node,Nprocs;
  std::string* dir;
  void init() {
    node=0; Nprocs=1;
    #ifdef MPI_ON
    MPI_Comm_rank (MPI_COMM_WORLD, &node);
    MPI_Comm_size (MPI_COMM_WORLD, &Nprocs);
    #endif
    printf("Size of drop_cells =%.2fKB\n", sizeof(uint32_t)*Ns*NDT*Nwarps/1024.);
    
    //get drop_cells from aivView, streamSys
    drop_cellsHost = new uint32_t[Np*NDT*Nwarps]; memset(drop_cellsHost, 0, sizeof(uint32_t)*Np*NDT*Nwarps);
    //     ____!!!___first warp is for PML___!!!___   NOPE, IT IS WRONG?? CHECK LATER
    //for(int ix=0; ix<NDT*Np; ix++) {drop_cellsHost[ix*Nwarps+Nz/32/2]=1/*65*/; drop_cellsHost[ix*Nwarps+Nz/32/2-1]=1; drop_cellsHost[ix*Nwarps+Nz/32/2+1]=1;}
    #if 1 
    for(int ix=0; ix<NDT*Np; ix+=3) for(int sline=0; sline<Nz/2-Npmlz/2; sline+=6) {
      int line=Nz/2+sline; int th_id=line; drop_cellsHost[ix*Nwarps+(th_id)/32]|=((uint64_t)1)<<(th_id%32); 
          line=Nz/2-sline;     th_id=line; drop_cellsHost[ix*Nwarps+(th_id)/32]|=((uint64_t)1)<<(th_id%32);
    } 
    for(int iw=0; iw<Nwarps; iw++) drop_cellsHost[Np*NDT/2*Nwarps+iw]=(((uint64_t)1)<<32)-1;
    #endif
    
    //CHECK_ERROR( cudaMalloc((void**)&drop_cells, sizeof(uint32_t)*Np*Nwarps) );
    //CHECK_ERROR( cudaMemcpy(drop_cells, drop_cellsHost, sizeof(uint32_t)*Np*Nwarps, cudaMemcpyHostToDevice) );
    
    CHECK_ERROR( cudaMalloc((void**)&channelAddr, sizeof(ftype*)*9) );
    for(int fld=0; fld<9; fld++){
      channelHost[fld] = new ftype[channelHostLength];
      offset[fld]=0;
      CHECK_ERROR( cudaMalloc((void**)&channel[fld], sizeof(ftype)*channelDevLength) ); // future think about size
      channelAddrHost[fld] = channel[fld];
    }
    CHECK_ERROR( cudaMemcpy(channelAddr, channelAddrHost, sizeof(ftype*)*9, cudaMemcpyHostToDevice) ); 
    char sf[256];
    
    #ifdef MPI_ON
    MPI_Info_create( &info );
    //MPI_Info_set(info, "romio_ds_write", "disable");
    //MPI_Info_set(info, "romio_ds_read",  "disable");
    for(int fn=0; fn<9; fn++) {
      switch(fn) {
        case 0: sprintf(sf,"%s/Sx.arr",dir->c_str()); break;
        case 1: sprintf(sf,"%s/Sz.arr",dir->c_str()); break;
        case 2: sprintf(sf,"%s/Sy.arr",dir->c_str()); break;
        case 3: sprintf(sf,"%s/Tx.arr",dir->c_str()); break;
        case 4: sprintf(sf,"%s/Tz.arr",dir->c_str()); break;
        case 5: sprintf(sf,"%s/Ty.arr",dir->c_str()); break;
        case 6: sprintf(sf,"%s/Vx.arr",dir->c_str()); break;
        case 7: sprintf(sf,"%s/Vz.arr",dir->c_str()); break;
        case 8: sprintf(sf,"%s/Vy.arr",dir->c_str()); break;
        default: printf("Unknown file number\n"); exit(-1);
      }
      //MPI_File_open( MPI_COMM_WORLD, sf, MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file[fn] );
      //MPI_File_close( &file[fn] );
      if(node==0) remove(sf);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_File_open( MPI_COMM_WORLD, sf, MPI_MODE_CREATE | MPI_MODE_EXCL            | MPI_MODE_WRONLY, info         , &file[fn] );
      MPI_File_set_atomicity(file[fn], true);
    }
    if(node==0) for(int i=0;i<9;i++) {
      uint32_t val = Np*NDT; MPI_File_write_shared(file[i], &val, 1, MPI_UNSIGNED, &status);
               val = Nwarps; MPI_File_write_shared(file[i], &val, 1, MPI_UNSIGNED, &status);
      MPI_File_write_shared(file[i], drop_cellsHost, Np*NDT*Nwarps, MPI_UNSIGNED, &status);
    }
    #else
    sprintf(sf,"%s/Sx.arr",dir->c_str()); file[0] = fopen(sf, "w"); if(file[0]==NULL) perror("Cannot open file for dump\n");
    sprintf(sf,"%s/Sz.arr",dir->c_str()); file[1] = fopen(sf, "w"); if(file[1]==NULL) perror("Cannot open file for dump\n");
    sprintf(sf,"%s/Sy.arr",dir->c_str()); file[2] = fopen(sf, "w"); if(file[2]==NULL) perror("Cannot open file for dump\n");
    sprintf(sf,"%s/Tx.arr",dir->c_str()); file[3] = fopen(sf, "w"); if(file[3]==NULL) perror("Cannot open file for dump\n");
    sprintf(sf,"%s/Tz.arr",dir->c_str()); file[4] = fopen(sf, "w"); if(file[4]==NULL) perror("Cannot open file for dump\n");
    sprintf(sf,"%s/Ty.arr",dir->c_str()); file[5] = fopen(sf, "w"); if(file[5]==NULL) perror("Cannot open file for dump\n");
    sprintf(sf,"%s/Vx.arr",dir->c_str()); file[6] = fopen(sf, "w"); if(file[6]==NULL) perror("Cannot open file for dump\n");
    sprintf(sf,"%s/Vz.arr",dir->c_str()); file[7] = fopen(sf, "w"); if(file[7]==NULL) perror("Cannot open file for dump\n");
    sprintf(sf,"%s/Vy.arr",dir->c_str()); file[8] = fopen(sf, "w"); if(file[8]==NULL) perror("Cannot open file for dump\n");
    for(int i=0;i<9;i++) {
      uint32_t val = Np*NDT; fwrite(&val, sizeof(uint32_t), 1, file[i]);
               val = Nwarps; fwrite(&val, sizeof(uint32_t), 1, file[i]);
      fwrite(drop_cellsHost, sizeof(uint32_t), Np*NDT*Nwarps, file[i]);
    }
    #endif
  }
  void copy_drop_cells(const int ixdev, const int ixhost, cudaStream_t& stream){
    DEBUG_PRINT(("copy drop cells HtoD ixdev=%d ixhost=%d \\ yes %d\n", ixdev, ixhost, ixhost<Np && ixhost>=0));
    if(ixhost<Np && ixhost>=0) CHECK_ERROR( cudaMemcpyToSymbolAsync(drop_cells, &drop_cellsHost[ixhost*NDT*Nwarps], sizeof(uint32_t)*NDT*Nwarps, sizeof(uint32_t)*ixdev*NDT*Nwarps, cudaMemcpyHostToDevice, stream) );
  }
  void save(cudaStream_t& stream){
    DEBUG_PRINT(("save channel from device to host\n"));
    CHECK_ERROR( cudaMemcpyAsync(channelAddrHost, channelAddr, sizeof(ftype*)*9, cudaMemcpyDeviceToHost, stream) );
    CHECK_ERROR( cudaStreamSynchronize(stream) );
    for(int i=0;i<9;i++) if(offset[i]+channelAddrHost[i]-channel[i] >= channelHostLength) { printf("Length of Host buffer channel %d is exceeded (to fix increase channelHostLength value)\n", i); exit(-1); }
    for(int i=0;i<9;i++) CHECK_ERROR( cudaMemcpyAsync((channelHost[i]+offset[i]), channel[i], (channelAddrHost[i]-channel[i])*sizeof(ftype), cudaMemcpyDeviceToHost, stream) );
    CHECK_ERROR( cudaStreamSynchronize(stream) );
    for(int i=0;i<9;i++) DEBUG_PRINT(("channel%d = %p, channelHost%d = %p, channelAddrHost%d = %p\n", i,channel[i], i,channelHost[i], i,channelAddrHost[i]));
    for(int i=0;i<9;i++) offset[i]+= (channelAddrHost[i]-channel[i]);
    for(int i=0;i<9;i++) DEBUG_PRINT(("new offset%d = %d\n", i,offset[i]));
    //---reset---
    for(int fld=0; fld<9; fld++) channelAddrHost[fld] = channel[fld];
    CHECK_ERROR( cudaMemcpyAsync(channelAddr, channelAddrHost, sizeof(ftype*)*9, cudaMemcpyHostToDevice, stream) );
    //int i=0; for(ftype* p=channelHost[i]; p!= channelHost[i]+offset[i]; p++) printf("buffer data p=%p val=%g\n", p, *p);
  }
  void dump(){
    DEBUG_PRINT(("dump channel data to file node=%d\n",node));
    #ifdef MPI_ON
    for(int i=0;i<9;i++) MPI_File_write_shared(file[i], channelHost[i], offset[i], MPI_FLOAT, &status);
    #else
    for(int i=0;i<9;i++) fwrite(channelHost[i], sizeof(ftype), offset[i], file[i]);
    #endif
    for(int i=0;i<9;i++) offset[i] = 0;
    DEBUG_PRINT(("ok dump channel data to file node=%d\n",node));
  }
  void sync(){
    #ifdef MPI_ON
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i=0;i<9;i++) MPI_File_sync(file[i]);
    #else
//    for(int i=0;i<9;i++) fflush(file[i]);
    #endif
  }
};
