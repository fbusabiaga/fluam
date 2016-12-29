#include <cuda_runtime.h>
#include <stdint.h>

#define FLUAM_VERSION "00.02.00"
#define SINGLE_PRECISION
#define DIM 3

typedef uint32_t uint;
typedef unsigned long long int ullint;


#if defined SINGLE_PRECISION
typedef float real;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
#else
typedef double real;
typedef double2 real2;
typedef double3 real3;
typedef double4 real4;
#endif

#if DIM == 3
typedef real3 reald;
typedef int3 intd;
#else
typedef real2 reald;
typedef int2 intd;
#endif


#ifndef SYSTEMINFO_H
#define SYSTEMINFO_H
struct SystemInfo{
  int dev;       // ID of the device 
  int cuda_arch; // Cuda compute capability of the device 100*major+10*minor  
};
#endif

extern SystemInfo sysInfo;
