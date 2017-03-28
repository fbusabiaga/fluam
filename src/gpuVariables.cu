// Filename: gpuVariables.cu
//
// Copyright (c) 2010-2016, Florencio Balboa Usabiaga
//
// This file is part of Fluam
//
// Fluam is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Fluam is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Fluam. If not, see <http://www.gnu.org/licenses/>.



/*********************************************************/
/* CELL VARIABLES FOR GPU                                */
/*********************************************************/ 
//DATA FOR RANDOM: 90.000.000 INT = 343 MB
//DATA FOR EACH CELL: 26 INT + 10 DOUBLE = 144 B
//DATA FOR EACH BOUNDARY: 1 INT + 34 DOUBLE = 140 B

typedef struct{
  int* vecino0GPU;
  int* vecino1GPU;
  int* vecino2GPU;
  int* vecino3GPU;
  int* vecino4GPU;
  int* vecino5GPU;
  int* vecinopxpyGPU;
  int* vecinopxmyGPU;
  int* vecinopxpzGPU;
  int* vecinopxmzGPU;
  int* vecinomxpyGPU;
  int* vecinomxmyGPU;
  int* vecinomxpzGPU;
  int* vecinomxmzGPU;
  int* vecinopypzGPU;
  int* vecinopymzGPU;
  int* vecinomypzGPU;
  int* vecinomymzGPU;
  int* vecinopxpypzGPU;
  int* vecinopxpymzGPU;
  int* vecinopxmypzGPU;
  int* vecinopxmymzGPU;
  int* vecinomxpypzGPU;
  int* vecinomxpymzGPU;
  int* vecinomxmypzGPU;
  int* vecinomxmymzGPU;
} vecinos;

typedef struct{
  double* fcell;
  double* fvec0;
  double* fvec1;
  double* fvec2;
  double* fvec3;
  double* fvec4;
  double* fvec5;
  double* fvecpxpy;
  double* fvecpxmy;
  double* fvecpxpz;
  double* fvecpxmz;
  double* fvecmxpy;
  double* fvecmxmy;
  double* fvecmxpz;
  double* fvecmxmz;
  double* fvecpypz;
  double* fvecpymz;
  double* fvecmypz;
  double* fvecmymz;
  double* fvecpxpypz;
  double* fvecpxpymz;
  double* fvecpxmypz;
  double* fvecpxmymz;
  double* fvecmxpypz;
  double* fvecmxpymz;
  double* fvecmxmypz;
  double* fvecmxmymz;
  int* position;
} fvec;

typedef struct{
  int* countparticlesincellX;
  int* countparticlesincellY;
  int* countparticlesincellZ;
  int* partincellX;
  int* partincellY;
  int* partincellZ;
  int* countPartInCellNonBonded;
  int* partInCellNonBonded;
  int* countparticlesincell;
  int* partincell;
} particlesincell;

typedef struct{
  cufftDoubleComplex* gradKx;
  cufftDoubleComplex* gradKy;
  cufftDoubleComplex* gradKz;
  cufftDoubleComplex* expKx;
  cufftDoubleComplex* expKy;
  cufftDoubleComplex* expKz;
} prefactorsFourier;


__constant__ int mxGPU, myGPU, mzGPU;
__constant__ int mxtGPU, mytGPU, mztGPU, mxmytGPU;
__constant__ int ncellsGPU, ncellstGPU;
__constant__ bool thermostatGPU;
__constant__ double lxGPU, lyGPU, lzGPU;
__constant__ double velocityboundaryGPU;
__constant__ double dtGPU;
__constant__ int numberneighboursGPU;
__constant__ double volumeGPU;
__constant__ double exGPU[6], eyGPU[6], ezGPU[6];
__constant__ double dxGPU, dyGPU, dzGPU; 
__constant__ double invdxGPU, invdyGPU, invdzGPU; 
__constant__ double invlxGPU, invlyGPU, invlzGPU;
__constant__ double invdtGPU;

double *densityGPU;
double *densityPredictionGPU;
double *vxGPU, *vyGPU, *vzGPU;
double *vxPredictionGPU, *vyPredictionGPU, *vzPredictionGPU;
double *fxGPU, *fyGPU, *fzGPU;
double *dmGPU;
double *dpxGPU, *dpyGPU, *dpzGPU;
double *rxcellGPU, *rycellGPU, *rzcellGPU;
double *advXGPU, *advYGPU, *advZGPU;
double *omegaGPU;

//IMEXRK
double *vx2GPU, *vy2GPU, *vz2GPU;
double *vx3GPU, *vy3GPU, *vz3GPU;
double *rxboundary2GPU, *ryboundary2GPU, *rzboundary2GPU;
double *rxboundary3GPU, *ryboundary3GPU, *rzboundary3GPU;
double *vxboundary2GPU, *vyboundary2GPU, *vzboundary2GPU;
double *vxboundary3GPU, *vyboundary3GPU, *vzboundary3GPU;
double *fx2GPU, *fy2GPU, *fz2GPU;
double *fx3GPU, *fy3GPU, *fz3GPU;



//Binary Mixture
double *cGPU, *cPredictionGPU, *dcGPU;

__constant__ double cWall0GPU, cWall1GPU, densityWall0GPU, densityWall1GPU;
__constant__ double vxWall0GPU, vxWall1GPU;
__constant__ double vyWall0GPU, vyWall1GPU;
__constant__ double vzWall0GPU, vzWall1GPU;
__constant__ double diffusionGPU, massSpecies0GPU, massSpecies1GPU;
__constant__ double shearviscosityGPU;
__constant__ double bulkviscosityGPU;
__constant__ double temperatureGPU;
__constant__ double pressurea0GPU;
__constant__ double pressurea1GPU;
__constant__ double pressurea2GPU;
__constant__ double densfluidGPU;

__constant__ double fact1GPU, fact2GPU, fact3GPU, fact4GPU;
__constant__ double fact5GPU, fact6GPU, fact7GPU;
__constant__ double volumeboundaryconstGPU;

__constant__ double soretCoefficientGPU, gradTemperatureGPU;

__constant__ double extraMobilityGPU;
__constant__ bool setExtraMobilityGPU;

double *rxboundaryGPU, *ryboundaryGPU, *rzboundaryGPU;
double *vxboundaryGPU, *vyboundaryGPU, *vzboundaryGPU;
double *fxboundaryGPU, *fyboundaryGPU, *fzboundaryGPU;
double *fboundaryOmega;
double *volumeboundaryGPU;
double *fbcell;

__constant__ double massParticleGPU, volumeParticleGPU;
__constant__ int npGPU;
__constant__ bool setparticlesGPU, setboundaryGPU;
__constant__ double omega0GPU;

int *ghostIndexGPU, *realIndexGPU;
int *ghostToPIGPU, *ghostToGhostGPU;
int *vecino0GPU, *vecino1GPU, *vecino2GPU;
int *vecino3GPU, *vecino4GPU, *vecino5GPU;
int *vecinopxpyGPU, *vecinopxmyGPU, *vecinopxpzGPU, *vecinopxmzGPU;
int *vecinomxpyGPU, *vecinomxmyGPU, *vecinomxpzGPU, *vecinomxmzGPU;
int *vecinopypzGPU, *vecinopymzGPU, *vecinomypzGPU, *vecinomymzGPU;
int *vecinopxpypzGPU, *vecinopxpymzGPU, *vecinopxmypzGPU, *vecinopxmymzGPU;
int *vecinomxpypzGPU, *vecinomxpymzGPU, *vecinomxmypzGPU, *vecinomxmymzGPU; 
__constant__ int nboundaryGPU;
__constant__ double vboundaryGPU;

int *neighbor0GPU, *neighbor1GPU, *neighbor2GPU;
int *neighbor3GPU, *neighbor4GPU, *neighbor5GPU;
int *neighborpxpyGPU, *neighborpxmyGPU, *neighborpxpzGPU, *neighborpxmzGPU;
int *neighbormxpyGPU, *neighbormxmyGPU, *neighbormxpzGPU, *neighbormxmzGPU;
int *neighborpypzGPU, *neighborpymzGPU, *neighbormypzGPU, *neighbormymzGPU;
int *neighborpxpypzGPU, *neighborpxpymzGPU, *neighborpxmypzGPU, *neighborpxmymzGPU;
int *neighbormxpypzGPU, *neighbormxpymzGPU, *neighbormxmypzGPU, *neighbormxmymzGPU;
__constant__ int mxNeighborsGPU, myNeighborsGPU, mzNeighborsGPU, mNeighborsGPU;


int *partincellX, *partincellY, *partincellZ;
int *countparticlesincellX, *countparticlesincellY, *countparticlesincellZ;
int *partInCellNonBonded, *countPartInCellNonBonded;
int *countparticlesincell, *partincell;
int *errorKernel;
double *saveForceX, *saveForceY, *saveForceZ;

__constant__ int maxNumberPartInCellGPU, maxNumberPartInCellNonBondedGPU;
__constant__ double cutoffGPU, invcutoffGPU, invcutoff2GPU;


//WAVE SOURCE
long long *stepGPU;
__constant__ double densityConstGPU, dDensityGPU;

vecinos *vec;
fvec *fb;
particlesincell *pc;

double *rxCheckGPU, *ryCheckGPU, *rzCheckGPU;
double *vxCheckGPU, *vyCheckGPU, *vzCheckGPU;

cudaArray *cuArrayDelta;
cudaArray *cuArrayDeltaDerived;
cudaArray *forceNonBonded1;

texture<int, 1> texvecino0GPU;
texture<int, 1> texvecino1GPU;
texture<int, 1> texvecino2GPU;
texture<int, 1> texvecino3GPU;
texture<int, 1> texvecino4GPU;
texture<int, 1> texvecino5GPU;
texture<int, 1> texvecinopxpyGPU;
texture<int, 1> texvecinopxmyGPU;
texture<int, 1> texvecinopxpzGPU;
texture<int, 1> texvecinopxmzGPU;
texture<int, 1> texvecinomxpyGPU;
texture<int, 1> texvecinomxmyGPU;
texture<int, 1> texvecinomxpzGPU;
texture<int, 1> texvecinomxmzGPU;
texture<int, 1> texvecinopypzGPU;
texture<int, 1> texvecinopymzGPU;
texture<int, 1> texvecinomypzGPU;
texture<int, 1> texvecinomymzGPU;
texture<int, 1> texvecinopxpypzGPU;
texture<int, 1> texvecinopxpymzGPU;
texture<int, 1> texvecinopxmypzGPU;
texture<int, 1> texvecinopxmymzGPU;
texture<int, 1> texvecinomxpypzGPU;
texture<int, 1> texvecinomxpymzGPU;
texture<int, 1> texvecinomxmypzGPU;
texture<int, 1> texvecinomxmymzGPU;

texture<int2, 1> texrxboundaryGPU;
texture<int2, 1> texryboundaryGPU;
texture<int2, 1> texrzboundaryGPU;
texture<int2, 1> texfxboundaryGPU;
texture<int2, 1> texfyboundaryGPU;
texture<int2, 1> texfzboundaryGPU;

cudaArray *cuArrayDeltaPBC;

texture<int, 1> texCountParticlesInCellX;
texture<int, 1> texCountParticlesInCellY;
texture<int, 1> texCountParticlesInCellZ;
texture<int, 1> texPartInCellX;
texture<int, 1> texPartInCellY;
texture<int, 1> texPartInCellZ;

texture<int2, 1> texVxGPU;
texture<int2, 1> texVyGPU;
texture<int2, 1> texVzGPU;

texture<int, 1> texCountParticlesInCellNonBonded;
texture<int, 1> texPartInCellNonBonded;

texture<float, 1, cudaReadModeElementType> texforceNonBonded1;

texture<int, 1> texneighbor0GPU;
texture<int, 1> texneighbor1GPU;
texture<int, 1> texneighbor2GPU;
texture<int, 1> texneighbor3GPU;
texture<int, 1> texneighbor4GPU;
texture<int, 1> texneighbor5GPU;
texture<int, 1> texneighborpxpyGPU;
texture<int, 1> texneighborpxmyGPU;
texture<int, 1> texneighborpxpzGPU;
texture<int, 1> texneighborpxmzGPU;
texture<int, 1> texneighbormxpyGPU;
texture<int, 1> texneighbormxmyGPU;
texture<int, 1> texneighbormxpzGPU;
texture<int, 1> texneighbormxmzGPU;
texture<int, 1> texneighborpypzGPU;
texture<int, 1> texneighborpymzGPU;
texture<int, 1> texneighbormypzGPU;
texture<int, 1> texneighbormymzGPU;
texture<int, 1> texneighborpxpypzGPU;
texture<int, 1> texneighborpxpymzGPU;
texture<int, 1> texneighborpxmypzGPU;
texture<int, 1> texneighborpxmymzGPU;
texture<int, 1> texneighbormxpypzGPU;
texture<int, 1> texneighbormxpymzGPU;
texture<int, 1> texneighbormxmypzGPU;
texture<int, 1> texneighbormxmymzGPU;


//Incompressible
cufftDoubleComplex *WxZ, *WyZ, *WzZ;
cufftDoubleComplex *vxZ, *vyZ, *vzZ;
cufftDoubleComplex *cZ;
cufftDoubleComplex *gradKx, *gradKy, *gradKz;
cufftDoubleComplex *expKx, *expKy, *expKz;
prefactorsFourier *pF;


//IncompressibleBoundaryRK2
double *rxboundaryPredictionGPU, *ryboundaryPredictionGPU, *rzboundaryPredictionGPU;
double *vxboundaryPredictionGPU, *vyboundaryPredictionGPU, *vzboundaryPredictionGPU;



//NEW Bonded forces
typedef struct{
  int *bondsParticleParticleGPU;
  int *bondsParticleParticleOffsetGPU;
  int *bondsIndexParticleParticleGPU;
  double *r0ParticleParticleGPU;
  double *kSpringParticleParticleGPU;


  int *bondsParticleFixedPointGPU;
  int *bondsParticleFixedPointOffsetGPU;
  double *r0ParticleFixedPointGPU;
  double *kSpringParticleFixedPointGPU;
  double *rxFixedPointGPU;
  double *ryFixedPointGPU;
  double *rzFixedPointGPU;
} bondedForcesVariables;

__constant__ bool bondedForcesGPU;
bondedForcesVariables *bFV;
int *bondsParticleParticleGPU;
int *bondsParticleParticleOffsetGPU;
int *bondsIndexParticleParticleGPU;
double *r0ParticleParticleGPU;
double *kSpringParticleParticleGPU;


int *bondsParticleFixedPointGPU;
int *bondsParticleFixedPointOffsetGPU;
double *r0ParticleFixedPointGPU;
double *kSpringParticleFixedPointGPU;
double *rxFixedPointGPU;
double *ryFixedPointGPU;
double *rzFixedPointGPU;
__constant__ bool particlesWallGPU;
__constant__ bool computeNonBondedForcesGPU;
__constant__ int kernelWidthGPU, nDriftGPU;
__constant__ double GaussianVarianceGPU, deltaRFDGPU;
