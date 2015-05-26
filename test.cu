__global__ void kernel(float4* a, const cudaTextureObject_t* tex){
  a[0] = tex3D<float4>(tex[blockIdx.x], 0.1, 0.2, 0.3);
}

