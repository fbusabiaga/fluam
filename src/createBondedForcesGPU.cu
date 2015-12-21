// Filename: createBondedForcesGPU.cu
//
// Copyright (c) 2010-2015, Florencio Balboa Usabiaga
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
  bFV->r0ParticleFixedPointGPU          = r0ParticleFixedPointGPU;
  bFV->kSpringParticleFixedPointGPU     = kSpringParticleFixedPointGPU;
  bFV->rxFixedPointGPU          = rxFixedPointGPU;
  bFV->ryFixedPointGPU          = ryFixedPointGPU;
  bFV->rzFixedPointGPU          = rzFixedPointGPU;

}




bool createBondedForcesGPU(){

  //Copy constant memory
  cudaMemcpyToSymbol(bondedForcesGPU,&bondedForces,sizeof(bool));

  //Allocate memory
  cutilSafeCall(cudaMalloc((void**)&bFV,sizeof(bondedForcesVariables)));
  cutilSafeCall(cudaMalloc((void**)&bondsParticleParticleGPU,np*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&bondsParticleParticleOffsetGPU,np*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&bondsIndexParticleParticleGPU,nbondsParticleParticle*2*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&r0ParticleParticleGPU,nbondsParticleParticle*2*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&kSpringParticleParticleGPU,nbondsParticleParticle*2*sizeof(double)));

  //Copy global memory
  cutilSafeCall(cudaMemcpy(bondsParticleParticleGPU,bondsParticleParticle,
			   np*sizeof(int),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(bondsParticleParticleOffsetGPU,bondsParticleParticleOffset,
			   np*sizeof(int),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(bondsIndexParticleParticleGPU,bondsIndexParticleParticle,
			   nbondsParticleParticle*2*sizeof(int),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(r0ParticleParticleGPU,r0ParticleParticle,
			   nbondsParticleParticle*2*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(kSpringParticleParticleGPU,kSpringParticleParticle,
			   nbondsParticleParticle*2*sizeof(double),cudaMemcpyHostToDevice));





  //Allocate memory
  cutilSafeCall(cudaMalloc((void**)&bondsParticleFixedPointGPU,np*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&bondsParticleFixedPointOffsetGPU,np*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&r0ParticleFixedPointGPU,nbondsParticleFixedPoint*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&kSpringParticleFixedPointGPU,nbondsParticleFixedPoint*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rxFixedPointGPU,nbondsParticleFixedPoint*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&ryFixedPointGPU,nbondsParticleFixedPoint*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rzFixedPointGPU,nbondsParticleFixedPoint*sizeof(double)));

  //Copy global memory
  cutilSafeCall(cudaMemcpy(bondsParticleFixedPointGPU,bondsParticleFixedPoint,
			   np*sizeof(int),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(bondsParticleFixedPointOffsetGPU,bondsParticleFixedPointOffset,
			   np*sizeof(int),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(r0ParticleFixedPointGPU,r0ParticleFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(kSpringParticleFixedPointGPU,kSpringParticleFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(rxFixedPointGPU,rxFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(ryFixedPointGPU,ryFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(rzFixedPointGPU,rzFixedPoint,
			   nbondsParticleFixedPoint*sizeof(double),cudaMemcpyHostToDevice));




  initBondedForcesVariables<<<1,1>>>(bFV,
				     bondsParticleParticleGPU,
				     bondsParticleParticleOffsetGPU,
				     bondsIndexParticleParticleGPU,
				     r0ParticleParticleGPU,
				     kSpringParticleParticleGPU,
				     bondsParticleFixedPointGPU,
				     bondsParticleFixedPointOffsetGPU,
				     r0ParticleFixedPointGPU,
				     kSpringParticleFixedPointGPU,
				     rxFixedPointGPU,
				     ryFixedPointGPU,
				     rzFixedPointGPU);




  return 1;
}
