#include <mpi.h>

#define CHECK_ERROR(err) CheckError( err, __FILE__,__LINE__)
static void CheckError( cudaError_t err, const char *file, int line) {
  if(err!=cudaSuccess){
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

__global__ void kernel(float* p){
  printf("%d   %g\n",threadIdx.x, p[threadIdx.x]);
}

int main(int argc, char** argv){
  printf("mpi_init\n");
  MPI_Init (&argc, &argv);
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  float* p1, *p2, *p3, *p4;
  if(rank==0) {
  printf("rank 0 malloc\n");
    CHECK_ERROR( cudaSetDevice(0) );
    CHECK_ERROR( cudaMalloc((void**)&p1, 100*sizeof(float)) );
    CHECK_ERROR( cudaSetDevice(1) );
    CHECK_ERROR( cudaMalloc((void**)&p2, 100*sizeof(float)) );
    for(int i=0;i<100; i++) { p1[i]=i; }
    for(int i=0;i<100; i++) { p2[i]=i+1000; }
  printf("rank 0 malloc ok\n");
  }
  if(rank==1) {
  printf("rank 1 malloc\n");
    CHECK_ERROR( cudaSetDevice(0) );
    CHECK_ERROR( cudaMalloc((void**)&p3, 100*sizeof(float)));
    CHECK_ERROR( cudaSetDevice(1) );
    CHECK_ERROR( cudaMalloc((void**)&p4, 100*sizeof(float)));
  printf("rank 1 malloc ok\n");
  }
  MPI_Request request1, request2;
  MPI_Status status;
  printf("start sendrev 1\n");
  if(rank==1) MPI_Irecv(p1, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request1);
  if(rank==0) MPI_Isend(p4, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request2);
  MPI_Wait(&request1, &status);
  MPI_Wait(&request2, &status);

  printf("start sendrev 2\n");
  if(rank==1) MPI_Irecv(p2, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request1);
  if(rank==0) MPI_Isend(p3, 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request2);
  MPI_Wait(&request1, &status);
  MPI_Wait(&request2, &status);

  printf("device0\n");
  CHECK_ERROR( cudaSetDevice(0) );
  kernel<<<1,100>>>(p3);
  printf("device1\n");
  CHECK_ERROR( cudaSetDevice(1) );
  kernel<<<1,100>>>(p4);
  
  MPI_Finalize();

  return 0;

}
