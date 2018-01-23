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

  // Initialize random numbers
  size_t numberRandom = 4 * (ncells + mx + my) + 2 * np * nDrift;
  if (numberRandom % 2){
    numberRandom += 1;
  }
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;

  // Initialize textures cells
  if(!texturesCellsQuasi2D()) return 0;  

  // Initialize plan
  cufftHandle FFT;
  cufftPlan2d(&FFT,my,mx,CUFFT_Z2Z);

  // Initialize factors for fourier space update
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
    if(((step % samplefreq == 0) or (step % sampleHydroGrid == 0)) and (step>=0)){
      if(!gpuToHostStokesLimit()) return 0;
      if(step % samplefreq == 0){
        // Save particle positions
        if(quasi2D){
          cout << "Quasi 2D " << step << endl;
        }
        else if(stokesLimit2D){
          cout << "stokesLimit 2D " << step << endl;
        }
        if(!saveFunctionsSchemeStokesLimit(1,step, samplefreq)) return 0;
      }
      if(step % sampleHydroGrid == 0){
        // Update Hydrogrid
        if(!saveFunctionsSchemeStokesLimit(3,step, samplefreq)) return 0;
      }
    }
    if((savefreq > 0) and (step>=0)){
      if(step % savefreq == 0){
        if(!gpuToHostStokesLimit()) return 0;
        // Save Hydrogrid
        if(!saveFunctionsSchemeStokesLimit(4,step, samplefreq)) return 0;
      }
    }

    // Generate random numbers
    generateRandomNumbers(numberRandom);
    
    // Clear neighbor lists
    countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
    
    // Set field to zero
    setFieldToZeroInput<<<numBlocks, threadsPerBlock>>>(vxZ, vyZ, vzZ);

    if(computeNonBondedForces){
      // Fill neighbor lists
      findNeighborListsQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>
        (pc, 
         errorKernel,
         rxcellGPU,
         rycellGPU,
         rzcellGPU,
         rxboundaryGPU,  // q^{n}
         ryboundaryGPU, 
         rzboundaryGPU);
    
      // Compute and Spread forces 
      // f = S*F
      kernelSpreadParticlesForceQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
                                                                                         rycellGPU,
                                                                                         vxZ,
                                                                                         vyZ,
                                                                                         bFV);    
    }

    // Spread thermal drift
    if(use_RFD){
      kernelSpreadThermalDriftQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
                                                                                       rycellGPU,
                                                                                       vxZ,
                                                                                       vyZ,
                                                                                       dRand,
                                                                                       rxboundaryGPU,
                                                                                       ryboundaryGPU,
                                                                                       4 * (ncells + mx + my));
    }


    //Add external forces
    //Raul added, call to shear flow kernel, this sums a sinusoidal force to each fluid cell.
    if(viscosityMeasureAmplitude != 0.0){

      addShearFlowQuasi2D<<<numBlocks, threadsPerBlock>>>(vxZ, vyZ, viscosityMeasureAmplitude);
      /* Only with CUDA 8.0+
      using vZType = typename remove_pointer<decltype(vyZ)>::type;
      using vTuple = thrust::tuple<vZType, int>;
      
      thrust::device_ptr<vZType> d_vy(vyZ);
      
      auto first = thrust::make_zip_iterator(thrust::make_tuple(d_vy, thrust::counting_iterator<int>(0)));

      double viscosityMeasureAmplitudeGPU = viscosityMeasureAmplitude;
      auto velocityModifier = [viscosityMeasureAmplitudeGPU] __device__  (vTuple v)->vTuple {		
	double& vy = thrust::get<1>(v).x;
	const int i = thrust::get<2>(v);
	const int ix = i%mxGPU;
	
	constexpr double pi2 = 2.0*3.1415926535897932385;
	vy += viscosityMeasureAmplitudeGPU*sin(pi2*(ix/(double)mxGPU));
	
	return v;
      };

      thrust::transform(first,
		        first+ncells,
			first,
			velocityModifier
			);
      */
    }
    
    // Transform force density field to Fourier space
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);

    if(quasi2D){
      // Compute deterministic fluid velocity
      kernelUpdateVRPYQuasi2D<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ);
      // Add stochastic velocity
      addStochasticVelocityRPYQuasi2D<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ,dRand,1.0,0);
    }
    else if(stokesLimit2D){
      //Raul added, call Saffman kernels if Saffman mode is a Saffman cut off is provided.
      if(saffmanCutOffWaveNumber != 0.0){
	// Compute deterministic fluid velocity
	kernelUpdateVIncompressibleSaffman2D<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxZ,vyZ,vzZ,pF); 
	// Add stochastic velocity
	addStochasticVelocitySaffman2D<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ,dRand);

      }
      else{
	// Compute deterministic fluid velocity
	kernelUpdateVIncompressibleSpectral2D<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxZ,vyZ,vzZ,pF); 
	// Add stochastic velocity
	addStochasticVelocitySpectral2D<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ,dRand);
      }
    }

    // Transform velocity field to real space
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);  
    doubleComplexToDoubleNormalized<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ,vzZ,vxGPU,vyGPU,vzGPU);

    // Update particles half-time step
    if(predictorCorrector){
      updateParticlesQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>
        (pc, 
         errorKernel,
         rxcellGPU,
         rycellGPU,
         rxboundaryGPU,  // q^{} to interpolate
         ryboundaryGPU, 
         rxboundaryPredictionGPU,  // q^{updated}
         ryboundaryPredictionGPU, 
         vxGPU,
         vyGPU,
         0.5 * dt);    
      
      // Update particles one time step
      updateParticlesQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>
        (pc, 
         errorKernel,
         rxcellGPU,
         rycellGPU,
         rxboundaryPredictionGPU,  // q^{} to interpolate
         ryboundaryPredictionGPU, 
         rxboundaryGPU,  // q^{updated}
         ryboundaryGPU, 
         vxGPU,
         vyGPU,
         dt);
    }
    else{// Forward Euler
      updateParticlesQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>
        (pc, 
         errorKernel,
         rxcellGPU,
         rycellGPU,
         rxboundaryGPU,  // q^{} to interpolate
         ryboundaryGPU, 
         rxboundaryGPU,  // q^{updated}
         ryboundaryGPU, 
         vxGPU,
         vyGPU,
         dt);    
    }
								    
    step++;
  }


  if(((step % samplefreq == 0) or (step % sampleHydroGrid == 0)) and (step>=0)){
    if(!gpuToHostStokesLimit()) return 0;
    if(step % samplefreq == 0){
      if(quasi2D){
        cout << "Quasi 2D " << step << endl;
      }
      else if(stokesLimit2D){
        cout << "stokesLimit 2D " << step << endl;
      }
      if(!saveFunctionsSchemeStokesLimit(1,step, samplefreq)) return 0;
    }
    if(step % sampleHydroGrid == 0){
      if(!saveFunctionsSchemeStokesLimit(3,step, samplefreq)) return 0;
    }
  }
  if(savefreq > 0){
    if(step % savefreq == 0){
      if(!gpuToHostStokesLimit()) return 0;
      // Save Hydrogrid
      if(!saveFunctionsSchemeStokesLimit(4,step, samplefreq)) return 0;
    }
  }
  // Free FFT
  cufftDestroy(FFT);
  freeRandomNumbersGPU();

  return 1;
}





















bool runSchemeQuasi2D_twoStokesSolve(){
// bool runSchemeQuasi2D(){
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

  // Initialize random numbers
  size_t numberRandom = 8 * (ncells + mx + my) + 2 * np;
  if (numberRandom % 2){
    numberRandom += 1;
  }
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;

  // Initialize textures cells
  if(!texturesCellsQuasi2D()) return 0;  

  // Initialize plan
  cufftHandle FFT;
  cufftPlan2d(&FFT,my,mx,CUFFT_Z2Z);

  // Initialize factors for fourier space update
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
    if(!(step%samplefreq)&&(step>=0)){
      cout << "Quasi 2D " << step << endl;
      if(!gpuToHostStokesLimit()) return 0;
      if(!saveFunctionsSchemeStokesLimit(1,step, samplefreq)) return 0;
    }

    // Generate random numbers
    generateRandomNumbers(numberRandom);
    
    // Clear neighbor lists
    countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
    
    // Set field to zero
    setFieldToZeroInput<<<numBlocks, threadsPerBlock>>>(vxZ, vyZ, vzZ);

    // Fill neighbor lists
    findNeighborListsQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>
      (pc, 
       errorKernel,
       rxcellGPU,
       rycellGPU,
       rzcellGPU,
       rxboundaryGPU,  // q^{n}
       ryboundaryGPU, 
       rzboundaryGPU);

    // Compute and Spread forces 
    // f = S*F
    kernelSpreadParticlesForceQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
                                                                                       rycellGPU,
                                                                                       vxZ,
                                                                                       vyZ,
                                                                                       bFV);

    // Spread thermal drift
    kernelSpreadThermalDriftQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
                                                                                     rycellGPU,
                                                                                     vxZ,
                                                                                     vyZ,
                                                                                     dRand,
                                                                                     rxboundaryGPU,
                                                                                     ryboundaryGPU,
                                                                                     8 * (ncells + mx + my));

    // Transform force density field to Fourier space
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);

    // Compute deterministic fluid velocity
    kernelUpdateVRPYQuasi2D<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ);

    // Add stochastic velocity
    addStochasticVelocityRPYQuasi2D<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ,dRand, sqrt(2.0), 0);

    // Transform velocity field to real space
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);  
    doubleComplexToDoubleNormalized<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ,vzZ,vxGPU,vyGPU,vzGPU);

    // Update particles half-time step
    updateParticlesQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>
      (pc, 
       errorKernel,
       rxcellGPU,
       rycellGPU,
       rxboundaryGPU,  // q^{} to interpolate
       ryboundaryGPU, 
       rxboundaryPredictionGPU,  // q^{updated}
       ryboundaryPredictionGPU, 
       vxGPU,
       vyGPU,
       0.5 * dt);    
    
    //Load textures with particles position q^{n+1/2}
    cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryPredictionGPU,(nboundary+np)*sizeof(double)));

    // Clear neighbor lists
    countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
    
    // Set field to zero
    setFieldToZeroInput<<<numBlocks, threadsPerBlock>>>(vxZ, vyZ, vzZ);

    // Fill neighbor lists
    findNeighborListsQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>
      (pc, 
       errorKernel,
       rxcellGPU,
       rycellGPU,
       rzcellGPU,
       rxboundaryPredictionGPU,  // q^{n}
       ryboundaryPredictionGPU, 
       rzboundaryGPU);

    // Compute and Spread forces 
    // f = S*F
    kernelSpreadParticlesForceQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
                                                                                       rycellGPU,
                                                                                       vxZ,
                                                                                       vyZ,
                                                                                       bFV);

    // Spread thermal drift
    kernelSpreadThermalDriftQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
                                                                                     rycellGPU,
                                                                                     vxZ,
                                                                                     vyZ,
                                                                                     dRand,
                                                                                     rxboundaryPredictionGPU,
                                                                                     ryboundaryPredictionGPU,
                                                                                     8 * (ncells + mx + my));

    // Transform force density field to Fourier space
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);

    // Compute deterministic fluid velocity
    kernelUpdateVRPYQuasi2D<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ);

    // Add stochastic velocity
    addStochasticVelocityRPYQuasi2D<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ,dRand, sqrt(0.5), 1);

    // Transform velocity field to real space
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);  
    doubleComplexToDoubleNormalized<<<numBlocks, threadsPerBlock>>>(vxZ,vyZ,vzZ,vxGPU,vyGPU,vzGPU);

    //Load textures with particles position q^{n+1/2}
    cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryGPU,(nboundary+np)*sizeof(double)));
    cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryGPU,(nboundary+np)*sizeof(double)));

    // Update particles one time step
    updateParticlesQuasi2D<<<numBlocksParticles,threadsPerBlockParticles>>>
      (pc, 
       errorKernel,
       rxcellGPU,
       rycellGPU,
       rxboundaryPredictionGPU,  // q^{} to interpolate
       ryboundaryPredictionGPU, 
       rxboundaryGPU,  // q^{updated}
       ryboundaryGPU, 
       vxGPU,
       vyGPU,
       dt);
								    
    step++;
  }

  if(!(step%samplefreq)&&(step>0)){
    if(quasi2D){
      cout << "Quasi 2D " << step << endl;
    }
    else if(stokesLimit2D){
      cout << "stokesLimit 2D " << step << endl;
    }
    if(!gpuToHostStokesLimit()) return 0;
    if(!saveFunctionsSchemeStokesLimit(1,step, samplefreq)) return 0;
  }

  // Free FFT
  cufftDestroy(FFT);
  freeRandomNumbersGPU();

  return 1;
}
