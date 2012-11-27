// Filename: createBondedForcesGPU.cu
//
// Copyright (c) 2010-2012, Florencio Balboa Usabiaga
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


__global__ void initBondedForcesVariables(bondedForcesVariables* bFV,
					  int* bondsParticleParticleGPU,
					  int* bondsParticleParticleOffsetGPU,
					  int* bondsIndexParticleParticleGPU,
					  double* r0ParticleParticleGPU,
					  double* kSpringParticleParticleGPU,
					  int* bondsParticleFixedPointGPU,
					  int* bondsParticleFixedPointOffsetGPU,
					  //int* bondsIndexParticleFixedParticleGPU,
					  double* r0ParticleFixedPointGPU,
					  double* kSpringParticleFixedPointGPU,
					  double* rxFixedPointGPU,
					  double* ryFixedPointGPU,
					  double* rzFixedPointGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>0) return;   

  bFV->bondsParticleParticleGPU       = bondsParticleParticleGPU;
  bFV->bondsParticleParticleOffsetGPU = bondsParticleParticleOffsetGPU;
  bFV->bondsIndexParticleParticleGPU  = bondsIndexParticleParticleGPU;
  bFV->r0ParticleParticleGPU          = r0ParticleParticleGPU;
  bFV->kSpringParticleParticleGPU     = kSpringParticleParticleGPU;


  bFV->bondsParticleFixedPointGPU       = bondsParticleFixedPointGPU;
  bFV->bondsParticleFixedPointOffsetGPU = bondsParticleFixedPointOffsetGPU;
  //bFV->bondsIndexParticleFixedPointGPU  = bondsIndexParticleFixedPointGPU;
  bFV->r0ParticleFixedPointGPU          = r0ParticleFixedPointGPU;
  bFV->kSpringParticleFixedPointGPU     = kSpringParticleFixedPointGPU;
  bFV->rxFixedPointGPU          = rxFixedPointGPU;
  bFV->ryFixedPointGPU          = ryFixedPointGPU;
  bFV->rzFixedPointGPU          = rzFixedPointGPU;

}




bool createBondedForcesGPU(){

  //Copy constant memory
  cudaMemcpyToSymbol(bondedForcesGPU,&bondedForces,sizeof(bool));

  /*int aux[np];
  for(int i=0;i<np;i++)
  aux[i]=0;*/

  //Allocate memory
  cudaMalloc((void**)&bFV,sizeof(bondedForcesVariables));
  cudaMalloc((void**)&bondsParticleParticleGPU,np*sizeof(int));
  cudaMalloc((void**)&bondsParticleParticleOffsetGPU,np*sizeof(int));
  cudaMalloc((void**)&bondsIndexParticleParticleGPU,nbondsParticleParticle*sizeof(int));
  cudaMalloc((void**)&r0ParticleParticleGPU,nbondsParticleParticle*sizeof(double));
  cudaMalloc((void**)&kSpringParticleParticleGPU,nbondsParticleParticle*sizeof(double));

  //Copy global memory
  cudaMemcpy(bondsParticleParticleGPU,bondsParticleParticle,
			   np*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bondsParticleParticleOffsetGPU,bondsParticleParticleOffset,
			   np*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bondsIndexParticleParticleGPU,bondsIndexParticleParticle,
			   nbondsParticleParticle*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(r0ParticleParticleGPU,r0ParticleParticle,
			   nbondsParticleParticle*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(kSpringParticleParticleGPU,kSpringParticleParticle,
			   nbondsParticleParticle*sizeof(double),cudaMemcpyHostToDevice);





  //Allocate memory
  cudaMalloc((void**)&bondsParticleFixedPointGPU,np*sizeof(int));
  cudaMalloc((void**)&bondsParticleFixedPointOffsetGPU,np*sizeof(int));
  //cudaMalloc((void**)&bondsIndexParticleFixedPointGPU,nbondsParticleFixedPoint*sizeof(int));
  cudaMalloc((void**)&r0ParticleFixedPointGPU,nbondsParticleFixedPoint*sizeof(double));
  cudaMalloc((void**)&kSpringParticleFixedPointGPU,nbondsParticleFixedPoint*sizeof(double));
  cudaMalloc((void**)&rxFixedPointGPU,nbondsParticleFixedPoint*sizeof(double));
  cudaMalloc((void**)&ryFixedPointGPU,nbondsParticleFixedPoint*sizeof(double));
  cudaMalloc((void**)&rzFixedPointGPU,nbondsParticleFixedPoint*sizeof(double));

  //Copy global memory
  cudaMemcpy(bondsParticleFixedPointGPU,bondsParticleFixedPoint,
			   np*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bondsParticleFixedPointOffsetGPU,bondsParticleFixedPointOffset,
			   np*sizeof(int),cudaMemcpyHostToDevice);
  //cudaMemcpy(bondsIndexParticleFixedPointGPU,bondsIndexParticleFixedPoint,
  //		   nbondsParticleFixedPoint*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(r0ParticleFixedPointGPU,r0ParticleFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(kSpringParticleFixedPointGPU,kSpringParticleFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(rxFixedPointGPU,rxFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(ryFixedPointGPU,ryFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(rzFixedPointGPU,rzFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice);




  /*cutilSafeCall(cudaMemcpy(bondsParticleParticleGPU,aux,np*sizeof(int),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(bondsParticleParticleOffsetGPU,aux,
			   np*sizeof(int),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(bondsIndexParticleParticleGPU,aux,
  nbondsParticleParticle*sizeof(int),cudaMemcpyHostToDevice));*/




  initBondedForcesVariables<<<1,1>>>(bFV,
				     bondsParticleParticleGPU,
				     bondsParticleParticleOffsetGPU,
				     bondsIndexParticleParticleGPU,
				     r0ParticleParticleGPU,
				     kSpringParticleParticleGPU,
				     bondsParticleFixedPointGPU,
				     bondsParticleFixedPointOffsetGPU,
				     //bondsIndexParticleFixedPointGPU,
				     r0ParticleFixedPointGPU,
				     kSpringParticleFixedPointGPU,
				     rxFixedPointGPU,
				     ryFixedPointGPU,
				     rzFixedPointGPU);




  return 1;
}
