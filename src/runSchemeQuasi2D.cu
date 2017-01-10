// Filename: runSchemeQuasiNeutrallyBuoyant.cu
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


bool runSchemeQuasi2D(){
  int threadsPerBlock = 512;
  if((ncells/threadsPerBlock) < 512) threadsPerBlock = 256;
  if((ncells/threadsPerBlock) < 256) threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 64) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 64) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

  int threadsPerBlockParticles = 512;
  if((np/threadsPerBlockParticles) < 512) threadsPerBlockParticles = 256;
  if((np/threadsPerBlockParticles) < 256) threadsPerBlockParticles = 128;
  if((np/threadsPerBlockParticles) < 64) threadsPerBlockParticles = 64;
  if((np/threadsPerBlockParticles) < 64) threadsPerBlockParticles = 32;
  int numBlocksParticles = (np-1)/threadsPerBlockParticles + 1;

  int threadsPerBlockNeighbors, numBlocksNeighbors;
  if(ncells>numNeighbors){
    threadsPerBlockNeighbors = 512;
    if((ncells/threadsPerBlockNeighbors) < 512) threadsPerBlockNeighbors = 256;
    if((ncells/threadsPerBlockNeighbors) < 256) threadsPerBlockNeighbors = 128;
    if((ncells/threadsPerBlockNeighbors) < 64) threadsPerBlockNeighbors = 64;
    if((ncells/threadsPerBlockNeighbors) < 64) threadsPerBlockNeighbors = 32;
    numBlocksNeighbors = (ncells-1)/threadsPerBlockNeighbors + 1;
  }
  else{
    threadsPerBlockNeighbors = 512;
    if((numNeighbors/threadsPerBlockNeighbors) < 512) threadsPerBlockNeighbors = 256;
    if((numNeighbors/threadsPerBlockNeighbors) < 256) threadsPerBlockNeighbors = 128;
    if((numNeighbors/threadsPerBlockNeighbors) < 64) threadsPerBlockNeighbors = 64;
    if((numNeighbors/threadsPerBlockNeighbors) < 64) threadsPerBlockNeighbors = 32;
    numBlocksNeighbors = (numNeighbors-1)/threadsPerBlockNeighbors + 1;
  }

  step = -numstepsRelaxation;

  //initialize random numbers
  size_t numberRandom = 6 * ncells;
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;

  //Initialize textures cells
  if(!texturesCells()) return 0;  

  initializeVecinos<<<numBlocks,threadsPerBlock>>>(vecino1GPU,
						   vecino2GPU,
						   vecino3GPU,
						   vecino4GPU,
						   vecinopxpyGPU,
						   vecinopxmyGPU,
						   vecinopxpzGPU,
						   vecinopxmzGPU,
						   vecinomxpyGPU,
						   vecinomxmyGPU,
						   vecinomxpzGPU,
						   vecinomxmzGPU,
						   vecinopypzGPU,
						   vecinopymzGPU,
						   vecinomypzGPU,
						   vecinomymzGPU,
						   vecinopxpypzGPU,
						   vecinopxpymzGPU,
						   vecinopxmypzGPU,
						   vecinopxmymzGPU,
						   vecinomxpypzGPU,
						   vecinomxpymzGPU,
						   vecinomxmypzGPU,
						   vecinomxmymzGPU);
  
  initializeVecinos2<<<numBlocks,threadsPerBlock>>>(vecino0GPU,
						    vecino1GPU,
						    vecino2GPU,
						    vecino3GPU,
						    vecino4GPU,
						    vecino5GPU);


  //Initialize plan
  cufftHandle FFT;
  cufftPlan3d(&FFT,mz,my,mx,CUFFT_Z2Z);

  //Initialize factors for fourier space update
  int threadsPerBlockdim, numBlocksdim;
  if((mx>=my)&&(mx>=mz)){
    threadsPerBlockdim = 128;
    numBlocksdim = (mx-1)/threadsPerBlockdim + 1;
  }
  else if((my>=mz)){
    threadsPerBlockdim = 128;
    numBlocksdim = (my-1)/threadsPerBlockdim + 1;
  }
  else{
    threadsPerBlockdim = 128;
    numBlocksdim = (mz-1)/threadsPerBlockdim + 1;
  }
  initializePrefactorFourierSpace_1<<<1,1>>>(gradKx,
					     gradKy,
					     gradKz,
					     expKx,
					     expKy,
					     expKz,pF);
  
  initializePrefactorFourierSpace_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);







  while(step<numsteps){
    
    //
    // Generate random numbers
    // generateRandomNumbers(numberRandom);
    

    //
    // STEP 1: UPDATE PARTICLE POSITIONS TO  q^{n+1/2}
    // Clear neighbor lists
    // countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);





								    
    step++;
    if(!(step%samplefreq)&&(step>0)){
      cout << "QuasiNeutrallyBuoyant 2D " << step << endl;
      if(!gpuToHostIncompressibleBoundaryRK2(numBlocksParticles,threadsPerBlockParticles)) return 0;
      if(!saveFunctionsSchemeIncompressibleBoundary2D(1,step)) return 0;
    }
  }

  //Free FFT
  cufftDestroy(FFT);
  freeRandomNumbersGPU();

  return 1;
}
