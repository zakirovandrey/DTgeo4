__device__ inline void dropPP(int ix, const int ixsh, const int iz, const int it, ftype* RPchannel, const ftype& val) { 
  ix = (ix+Ns*NDT)%(Ns*NDT);
  if(drop_cells[ix*Nwarps+iz/WSIZE] >> iz%WSIZE &1){
    int channel_shift=0;
    for(int xprev=0; xprev<ixsh; xprev++) for(int iwarp=0; iwarp<Nwarps; iwarp++) channel_shift+= __popc(drop_cells[(ix+Ns*NDT-ixsh+xprev)%(Ns*NDT)*Nwarps+iwarp]);
    for(int iwarp=0; iwarp<iz/WSIZE; iwarp++) channel_shift+= __popc(drop_cells[ix*Nwarps+iwarp]);
    channel_shift+= __popc(drop_cells[ix*Nwarps+iz/WSIZE]<<(32-iz%WSIZE));
    RPchannel[channel_shift] = val;
    //else              RPchannel[channel_shift] = 0.5625*val+0.5625*valP-0.0625*valM-0.0625*valPP;
  }
}
