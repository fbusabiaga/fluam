// Filename: runSchemeIncompressibleBinaryMixtureMidPoint.cu
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


// A. Donev: This actually implements a backward Euler rather than a midpoint method
// For code to do midpoint but with a single random number per step see code below

bool runSchemeIncompressibleBinaryMixtureMidPoint(){
  int threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

  step = -numstepsRelaxation;

  //initialize random numbers
  size_t numberRandom = 9 * ncells;
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;

  //Initialize textures cells
  if(!texturesCells()) return 0;

  //Initialize neighbors list
  initializeVecinos<<<numBlocks,threadsPerBlock>>>(vecino1GPU,vecino2GPU,vecino3GPU,vecino4GPU,
						   vecinopxpyGPU,vecinopxmyGPU,vecinopxpzGPU,vecinopxmzGPU,
						   vecinomxpyGPU,vecinomxmyGPU,vecinomxpzGPU,vecinomxmzGPU,
						   vecinopypzGPU,vecinopymzGPU,vecinomypzGPU,vecinomymzGPU,
						   vecinopxpypzGPU,vecinopxpymzGPU,vecinopxmypzGPU,
						   vecinopxmymzGPU,
						   vecinomxpypzGPU,vecinomxpymzGPU,vecinomxmypzGPU,
						   vecinomxmymzGPU);
  initializeVecinos2<<<numBlocks,threadsPerBlock>>>(vecino0GPU,vecino1GPU,vecino2GPU,
						    vecino3GPU,vecino4GPU,vecino5GPU);


  //Initialize plan
  cufftHandle FFT;
  cufftPlan3d(&FFT,mz,my,mx,CUFFT_Z2Z);
  
  //Initialize factors for update in fourier space
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
  initializePrefactorFourierSpace_1<<<1,1>>>(gradKx,gradKy,gradKz,expKx,expKy,expKz,pF);
  initializePrefactorFourierSpace_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);
  

  while(step<numsteps){

    // Donev: Moved this here so the initial configuration is saved as well
    //if(!(step%samplefreq)&&(step>0)){
    if((step>=0)&&(!(step%abs(samplefreq)))&&(abs(samplefreq)>0)){
      cout << "INCOMPRESSIBLE UpdateHydroGrid" << step << endl;
      if(!gpuToHostIncompressible()) return 0;
      if(!saveFunctionsSchemeIncompressibleBinaryMixture(1)) return 0;
    }

    //Generate random numbers
    generateRandomNumbers(numberRandom);
    
    // Solving: (prefactor-dt*nu*L) v^{n+1} + G*pi = prefactor*v^{n} + W_v
    // and (1-0.5*dt*chi*L) c^{n+1} = (1+0.5*dt*chi*L) c^{n} + W_c
    // Here prefactor=1 is Backward Euler and prefactor=0 is overdamped (limiting) equation

    // ----------------------------------------
    // Step 1: Solve Backward Euler for velocity only as in single-component fluid case
    // ----------------------------------------
      
    // Calculate the right-hand side term W={W_v,W_c}, with no Laplacian terms
    kernelConstructW<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU,vxZ,vyZ,vzZ,dRand,0.0,identity_prefactor);//W

    // Forward Fourier transform:
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
    
    // Solve Backward-Euler system in Fourier space
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
    // A. Donev: Solve overdamped equation with coefficient (prefactor*I-L*dt)
    kernelUpdateVIncompressibleBE<<<numBlocks,threadsPerBlock>>>(vxZ,
								 vyZ,
								 vzZ,
								 vxZ,
								 vyZ,
								 vzZ,
								 pF,
								 identity_prefactor);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
    
    // Inverse Fourier transform:
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    
    // Update velocity:
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,
								   vxGPU,vyGPU,vzGPU);

    // ----------------------------------------
    // Step 2: Solve Crank-Nicolson for concentration
    // Here I reuse Floren's code and so it computes velocity and concentration
    // But we simply throw away the velocity in the end
    // ----------------------------------------
    kernelConstructWBinaryMixture_1<<<numBlocks,threadsPerBlock>>>(vxGPU,
								   vyGPU,
								   vzGPU,
								   cGPU,
                                                                   cGPU,
								   vxZ,
								   vyZ,
								   vzZ,
								   cZ,
								   dRand);
   
    // Note that here vx and vy are packed into vxZ
    // and vz and c are packed into vzZ so that only 2 instead of 4 FFTs are needed
    // Forward Fourier transform
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);

    // Now solve the actual linear system:
    kernelShiftBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,-1);
    kernelUpdateIncompressibleBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF);
    kernelShiftBinaryMixture_2<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,1);

    // Inverse Fourier transform:
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

   if(0) // Simple forward Euler (does not get the diffusion enhancement correctly)
   {
     // Nothing to be done
   }
   else // A predictor-corrector scheme that gets the enhanced diffusion correctly
   {

    if(0) // Calculate c<-c^{n+1}
    calculateVelocityPredictionBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,
									    vyZ,
									    vzZ,
									    cZ,
									    vxPredictionGPU,
									    vyPredictionGPU,
									    vzPredictionGPU,
									    cGPU,
									    cPredictionGPU);
    else // Calculate c<-c^{n+1/2}
    calculateVelocityPredictionBinaryMixtureMidPoint<<<numBlocks,threadsPerBlock>>>(vxZ,
										    vyZ,
										    vzZ,
										    cZ,
										    vxPredictionGPU,
										    vyPredictionGPU,
										    vzPredictionGPU,
										    cGPU,
										    cPredictionGPU);


    // Ignore vPredictionGPU and advect with the same velocity as before
    // but use cPredictionGPU for the advective fluxes of concentration
    kernelConstructWBinaryMixture_1<<<numBlocks,threadsPerBlock>>>(vxGPU,
								   vyGPU,
								   vzGPU,
                                                                   cGPU,
								   cPredictionGPU,
								   vxZ,
								   vyZ,
								   vzZ,
								   cZ,
								   dRand);


    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);

    kernelShiftBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,-1);
    kernelUpdateIncompressibleBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF);
    kernelShiftBinaryMixture_2<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,1);

    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

   }
   
    // We throw away the velocity into vPredictionGPU here but update the concentration:   
    calculateVelocityPredictionBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,
									    vyZ,
									    vzZ,
									    cZ,
									    vxPredictionGPU,
									    vyPredictionGPU,
									    vzPredictionGPU,
									    cGPU,
									    cGPU);
   

    step++;
  }

  freeRandomNumbersGPU();
  //Free FFT
  cufftDestroy(FFT);

  return 1;
}

// ==========================================================
// This is code by Floren to implement midpoint for the advection with Crank-Nicolson for diffusion
// This uses a single random forcing per step, which is not second-order nonlinearly
// ==========================================================

bool runSchemeIncompressibleBinaryMixtureMidPoint_Floren(){
  int threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

  step = -numstepsRelaxation;

  //initialize random numbers
  size_t numberRandom = 9 * ncells;
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;

  //Initialize textures cells
  if(!texturesCells()) return 0;

  //Initialize neighbors list
  initializeVecinos<<<numBlocks,threadsPerBlock>>>(vecino1GPU,vecino2GPU,vecino3GPU,vecino4GPU,
						   vecinopxpyGPU,vecinopxmyGPU,vecinopxpzGPU,vecinopxmzGPU,
						   vecinomxpyGPU,vecinomxmyGPU,vecinomxpzGPU,vecinomxmzGPU,
						   vecinopypzGPU,vecinopymzGPU,vecinomypzGPU,vecinomymzGPU,
						   vecinopxpypzGPU,vecinopxpymzGPU,vecinopxmypzGPU,
						   vecinopxmymzGPU,
						   vecinomxpypzGPU,vecinomxpymzGPU,vecinomxmypzGPU,
						   vecinomxmymzGPU);
  initializeVecinos2<<<numBlocks,threadsPerBlock>>>(vecino0GPU,vecino1GPU,vecino2GPU,
						    vecino3GPU,vecino4GPU,vecino5GPU);


  //Initialize plan
  cufftHandle FFT, FFT_D2Z, FFT_Z2D;
  cufftPlan3d(&FFT,mz,my,mx,CUFFT_Z2Z);
  cufftPlan3d(&FFT_D2Z,mz,my,mx,CUFFT_D2Z);
  cufftPlan3d(&FFT_Z2D,mz,my,mx,CUFFT_Z2D);
  
  //Initialize factors for update in fourier space
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
  initializePrefactorFourierSpace_1<<<1,1>>>(gradKx,gradKy,gradKz,expKx,expKy,expKz,pF);
  initializePrefactorFourierSpace_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);
  


  while(step<numsteps){
    //Generate random numbers
    generateRandomNumbers(numberRandom);

    //First substep
    kernelConstructWBinaryMixtureMidPoint_1<<<numBlocks,threadsPerBlock>>>(vxGPU,
									   vyGPU,
									   vzGPU,
									   cGPU,
									   vxZ,
									   vyZ,
									   vzZ,
									   cZ,
									   dRand);

    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
    //cufftExecD2Z(FFT_D2Z,vxPredictionGPU,vxZ);
    //cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);
    //cufftExecZ2Z(FFT,cZ,cZ,CUFFT_FORWARD);
    kernelShiftBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,-1);
    kernelUpdateIncompressibleBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF);
    kernelShiftBinaryMixture_2<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    //cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    //cufftExecZ2Z(FFT,cZ,cZ,CUFFT_INVERSE);
    calculateVelocityPredictionBinaryMixtureMidPoint<<<numBlocks,threadsPerBlock>>>(vxZ,
										    vyZ,
										    vzZ,
										    cZ,
										    vxPredictionGPU,
										    vyPredictionGPU,
										    vzPredictionGPU,
										    cGPU,
										    cPredictionGPU);


    //Second substep
    kernelConstructWBinaryMixtureMidPoint_2<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
									   vyPredictionGPU,
									   vzPredictionGPU,
									   cGPU,
									   cPredictionGPU,
									   vxZ,
									   vyZ,
									   vzZ,
									   cZ,
									   dRand);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
    //cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);
    //cufftExecZ2Z(FFT,cZ,cZ,CUFFT_FORWARD);
    kernelShiftBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,-1);
    kernelUpdateIncompressibleBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF);
    kernelShiftBinaryMixture_2<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,cZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    //cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    //cufftExecZ2Z(FFT,cZ,cZ,CUFFT_INVERSE);
    doubleComplexToDoubleNormalizedBinaryMixture<<<numBlocks,threadsPerBlock>>>(vxZ,
										vyZ,
										vzZ,
										cZ,
										vxGPU,
										vyGPU,
										vzGPU,
										cGPU);
    
    step++;
    if(!(step%samplefreq)&&(step>0)){
      cout << "INCOMPRESSIBLE " << step << endl;
      if(!gpuToHostIncompressible()) return 0;
      if(!saveFunctionsSchemeIncompressibleBinaryMixture(1)) return 0;
    }
  }

  freeRandomNumbersGPU();
  //Free FFT
  cufftDestroy(FFT);


  return 1;
}


