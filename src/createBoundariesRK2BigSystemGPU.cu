// Filename: createBoundariesRK2GPU.cu
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


bool createBoundariesRK2BigSystemGPU(){

  if(setparticles==0) np=0;
  
  cudaMemcpyToSymbol(nboundaryGPU,&nboundary,sizeof(int));
  cudaMemcpyToSymbol(npGPU,&np,sizeof(int));
  cutilSafeCall(cudaMemcpyToSymbol(maxNumberPartInCellGPU,&maxNumberPartInCell,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(maxNumberPartInCellNonBondedGPU,&maxNumberPartInCellNonBonded,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(particlesWallGPU,&particlesWall,sizeof(bool)));
  cudaMemcpyToSymbol(computeNonBondedForcesGPU,&computeNonBondedForces,sizeof(bool));

  //Create boundaries and particles variables
  cutilSafeCall(cudaMalloc((void**)&rxboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&ryboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rzboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rxboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&ryboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rzboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vxboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vxboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&fxboundaryGPU,27*(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&fyboundaryGPU,27*(nboundary+np)*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&fzboundaryGPU,27*(nboundary+np)*sizeof(double)));

  //Initialize boundaries variables
  cutilSafeCall(cudaMemcpy(rxboundaryGPU,rxboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(ryboundaryGPU,ryboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(rzboundaryGPU,rzboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(vxboundaryGPU,vxboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(vyboundaryGPU,vyboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(vzboundaryGPU,vzboundary,nboundary*sizeof(double),cudaMemcpyHostToDevice));

  //Initialize particles variables
  cutilSafeCall(cudaMemcpy(&rxboundaryGPU[nboundary],rxParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&ryboundaryGPU[nboundary],ryParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&rzboundaryGPU[nboundary],rzParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&vxboundaryGPU[nboundary],vxParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&vyboundaryGPU[nboundary],vyParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(&vzboundaryGPU[nboundary],vzParticle,np*sizeof(double),cudaMemcpyHostToDevice));
  
  //Copy some constants
  cutilSafeCall(cudaMemcpyToSymbol(volumeboundaryconstGPU,&volumeboundaryconst,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(massParticleGPU,&mass,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(volumeParticleGPU,&volumeParticle,sizeof(double)));
 
              
  cutilSafeCall(cudaMalloc((void**)&countparticlesincellX,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&countparticlesincellY,ncells*sizeof(int)));
  cutilSafeCall(cudaMalloc((void**)&countparticlesincellZ,ncells*sizeof(int)));

  int *aux;
  aux = new int [ncells];
  for(int i=0;i<ncells;i++) aux[i] = 0;
  cudaMemcpy(countparticlesincellX,aux,ncells*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(countparticlesincellY,aux,ncells*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(countparticlesincellZ,aux,ncells*sizeof(int),cudaMemcpyHostToDevice);
  delete[] aux;

  cudaMalloc((void**)&partincellX,maxNumberPartInCell*ncells*sizeof(int));
  cudaMalloc((void**)&partincellY,maxNumberPartInCell*ncells*sizeof(int));
  cudaMalloc((void**)&partincellZ,maxNumberPartInCell*ncells*sizeof(int));
  

  //texrxboundaryGPU
  texrxboundaryGPU.normalized = false;
  texrxboundaryGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texrxboundaryGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryGPU,(nboundary+np)*sizeof(double)));
  //texryboundaryGPU
  texryboundaryGPU.normalized = false;
  texryboundaryGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texryboundaryGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryGPU,(nboundary+np)*sizeof(double)));
  //texrzboundaryGPU
  texrzboundaryGPU.normalized = false;
  texrzboundaryGPU.addressMode[0] = cudaAddressModeClamp;//Wrap and Clamp
  texrzboundaryGPU.filterMode = cudaFilterModePoint;//cudaFilterModeLinear and cudaFilterModePoint
  cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryGPU,(nboundary+np)*sizeof(double)));

  if(setparticles){
    int mxPart = int(lx/cutoff);
    if(mxPart < 3) mxPart = 3;
    int myPart = int(ly/cutoff);
    if(myPart < 3) myPart = 3;
    int mzPart = int(lz/cutoff);
    if(mzPart < 3) mzPart = 3;
    numNeighbors = mxPart * myPart * mzPart;

    cutilSafeCall(cudaMalloc((void**)&countPartInCellNonBonded,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&partInCellNonBonded,
			     maxNumberPartInCellNonBonded*numNeighbors*sizeof(int)));

    cutilSafeCall(cudaMemcpyToSymbol(mxNeighborsGPU,&mxPart,sizeof(int)));
    cutilSafeCall(cudaMemcpyToSymbol(myNeighborsGPU,&myPart,sizeof(int)));
    cutilSafeCall(cudaMemcpyToSymbol(mzNeighborsGPU,&mzPart,sizeof(int)));
    cutilSafeCall(cudaMemcpyToSymbol(mNeighborsGPU,&numNeighbors,sizeof(int)));
    
    cutilSafeCall(cudaMalloc((void**)&neighbor0GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor1GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor2GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor3GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor4GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbor5GPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxpyGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxmyGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxpzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxmzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxpyGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxmyGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxpzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxmzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxpypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxpymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxmypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighborpxmymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxpypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxpymzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxmypzGPU,numNeighbors*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&neighbormxmymzGPU,numNeighbors*sizeof(int)));
    
    cutilSafeCall(cudaMemcpyToSymbol(cutoffGPU,&cutoff,sizeof(double)));
    double invcutoff = 1./cutoff;
    cutilSafeCall(cudaMemcpyToSymbol(invcutoffGPU,&invcutoff,sizeof(double)));
    invcutoff = 1./(cutoff * cutoff);
    cutilSafeCall(cudaMemcpyToSymbol(invcutoff2GPU,&invcutoff,sizeof(double)));
       
    

    int block = (numNeighbors-1)/128 + 1;
    cutilSafeCall(cudaMalloc((void**)&pNeighbors,sizeof(vecinos)));
    initializeNeighborsFull<<<block,128>>>(neighbor0GPU,neighbor1GPU,neighbor2GPU,neighbor3GPU,neighbor4GPU,neighbor5GPU,
					   neighborpxpyGPU,neighborpxmyGPU,neighborpxpzGPU,neighborpxmzGPU,
					   neighbormxpyGPU,neighbormxmyGPU,neighbormxpzGPU,neighbormxmzGPU,
					   neighborpypzGPU,neighborpymzGPU,neighbormypzGPU,neighbormymzGPU,
					   neighborpxpypzGPU,neighborpxpymzGPU,neighborpxmypzGPU,neighborpxmymzGPU,
					   neighbormxpypzGPU,neighbormxpymzGPU,neighbormxmypzGPU,neighbormxmymzGPU, 
					   pNeighbors);

    initForcesNonBonded();

  }


  initDelta();
  allocateErrorArray();

  cutilSafeCall(cudaMalloc((void**)&pc,sizeof(particlesincell)));
  
  initParticlesInCell<<<1,1>>>(partincellX,partincellY,partincellZ,
			       countparticlesincellX,countparticlesincellY,countparticlesincellZ,
			       countPartInCellNonBonded,partInCellNonBonded,pc);

  double *auxDouble;
  auxDouble = new double [27*(nboundary+np)];
  for(int i=0;i<27*(nboundary+np);i++) auxDouble[i] = 0;
  cutilSafeCall(cudaMemcpy(fxboundaryGPU,auxDouble,27*(nboundary+np)*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(fyboundaryGPU,auxDouble,27*(nboundary+np)*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(fzboundaryGPU,auxDouble,27*(nboundary+np)*sizeof(double),cudaMemcpyHostToDevice));
  delete[] auxDouble;

  //Copy constant memory
  cudaMemcpyToSymbol(bondedForcesGPU,&bondedForces,sizeof(bool));

  
  cout << "CREATE BOUNDARIES GPU :         DONE" << endl; 

  return 1;
}
