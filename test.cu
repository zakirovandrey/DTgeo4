__global__ void kernel(float* a, const cudaTextureObject_t* tex){
  a[0] = tex3D<float>(tex[blockIdx.x], 0.1, 0.2, 0.3);
}

