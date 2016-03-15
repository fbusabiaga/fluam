// Filename: runSchemeIncompressible.cu
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

bool runSchemeMHD(){
  int threadsPerBlock = 512;
  if((ncells/threadsPerBlock) < 512) threadsPerBlock = 256;
  if((ncells/threadsPerBlock) < 256) threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 64) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 64) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

  step = -numstepsRelaxation;

  //initialize random numbers
  size_t numberRandom = 6 * ncells;
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;

  //Initialize textures cells
  if(!texturesCells()) return 0;

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
  initializePrefactorFourierSpaceSpectral_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);
  // initializePrefactorFourierSpace_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);






  // A. Donev: Project the initial velocity to make sure it is div-free
  //---------------------------------------------------------
  //Copy velocities to complex variable
  doubleToDoubleComplex<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU,vxZ,vyZ,vzZ);
  doubleToDoubleComplex<<<numBlocks,threadsPerBlock>>>(bxGPU,byGPU,bzGPU,WxZ,WyZ,WzZ);

  //Take velocities to fourier space
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);
  cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_FORWARD);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,-1);
  //Project into divergence free space
  // initializeMHD<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,WxZ,WyZ,WzZ,pF);
  projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
  projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF);


  //Take velocities to real space
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
  kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,1);
  cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_INVERSE);

  //Copy velocities to real variables
  // doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxGPU,vyGPU,vzGPU);
  // doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,bxGPU,byGPU,bzGPU);
  //---------------------------------------------------------




  

  while(step<numsteps){
    //Generate random numbers
    generateRandomNumbers(numberRandom);

    // First substep
    kernelConstructWMHD_1<<<numBlocks,threadsPerBlock>>>(bxGPU,byGPU,bzGPU,
     							 vxZ,vyZ,vzZ,
							 WxZ,WyZ,WzZ,
     							 dRand);//W

    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,-1);//W
    kernelUpdateMHD<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,WxZ,WyZ,WzZ,pF);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,1);
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_INVERSE);
    calculateVelocityPrediction<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,
							       vxPredictionGPU,vyPredictionGPU,vzPredictionGPU);
    calculateVectorFieldMidPoint<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,
								bxGPU,byGPU,bzGPU,
								bxPredictionGPU,byPredictionGPU,bzPredictionGPU);

     
    // Second substep
    kernelConstructWMHD_2<<<numBlocks,threadsPerBlock>>>(bxGPU, byGPU, bzGPU,
							 vxPredictionGPU,vyPredictionGPU,vzPredictionGPU,
							 bxPredictionGPU,byPredictionGPU,bzPredictionGPU,
							 vxZ,vyZ,vzZ,
							 WxZ,WyZ,WzZ,
							 dRand);//W
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,-1);//W
    kernelUpdateMHD<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,WxZ,WyZ,WzZ,pF);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,1);
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_INVERSE);
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxGPU,vyGPU,vzGPU);
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,bxGPU,byGPU,bzGPU);

    step++;
    if(!(step%samplefreq)&&(step>0)){
      cout << "MHD " << step << endl;
      if(!gpuToHostMHD()) return 0;
      if(!saveFunctionsSchemeMHD(1)) return 0;
    }
  }

  freeRandomNumbersGPU();
  //Free FFT
  cufftDestroy(FFT);

  return 1;
}



















bool runSchemeMHDRK3(){
  int threadsPerBlock = 512;
  if((ncells/threadsPerBlock) < 512) threadsPerBlock = 256;
  if((ncells/threadsPerBlock) < 256) threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 64) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 64) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

  step = -numstepsRelaxation;

  //initialize random numbers
  size_t numberRandom = 6 * ncells;
  if(!initializeRandomNumbersGPU(numberRandom,seed)) return 0;

  //Initialize textures cells
  if(!texturesCells()) return 0;

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
  initializePrefactorFourierSpaceSpectral_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);
  // initializePrefactorFourierSpace_2<<<numBlocksdim,threadsPerBlockdim>>>(pF);


  // A. Donev: Project the initial velocity to make sure it is div-free
  //---------------------------------------------------------
  //Copy velocities to complex variable
  doubleToDoubleComplex<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU,vxZ,vyZ,vzZ);
  doubleToDoubleComplex<<<numBlocks,threadsPerBlock>>>(bxGPU,byGPU,bzGPU,WxZ,WyZ,WzZ);

  //Take velocities to fourier space
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);
  cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_FORWARD);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,-1);
  //Project into divergence free space
  // initializeMHD<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,WxZ,WyZ,WzZ,pF);
  projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
  projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF);


  //Take velocities to real space
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
  kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,1);
  cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_INVERSE);

  //Copy velocities to real variables
  doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxGPU,vyGPU,vzGPU);
  doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,bxGPU,byGPU,bzGPU);
  //---------------------------------------------------------
  

  currentTime = 0;
  while(step<numsteps){

    // Set time step
    if(1){
      double max_v_old = 0;
      double max_v = 0;
      { // vx
	thrust::device_vector<double> vec(vxGPU, vxGPU + ncells);
	max_v = thrust::transform_reduce(vec.begin(), vec.end(), absolute_value<double>(), double(0), thrust::maximum<double>());
	if(max_v > max_v_old){
	  max_v_old = max_v;
	}
      }
      { // vy
	thrust::device_vector<double> vec(vyGPU, vyGPU + ncells);
	max_v = thrust::transform_reduce(vec.begin(), vec.end(), absolute_value<double>(), double(0), thrust::maximum<double>());
	if(max_v > max_v_old){
	  max_v_old = max_v;
	}
      }
      { // vz
	thrust::device_vector<double> vec(vzGPU, vzGPU + ncells);
	max_v = thrust::transform_reduce(vec.begin(), vec.end(), absolute_value<double>(), double(0), thrust::maximum<double>());
	if(max_v > max_v_old){
	  max_v_old = max_v;
	}
      }
      { // bx
	thrust::device_vector<double> vec(bxGPU, bxGPU + ncells);
	max_v = thrust::transform_reduce(vec.begin(), vec.end(), absolute_value<double>(), double(0), thrust::maximum<double>());
	if(max_v > max_v_old){
	  max_v_old = max_v;
	}
      }
      { // by
	thrust::device_vector<double> vec(byGPU, byGPU + ncells);
	max_v = thrust::transform_reduce(vec.begin(), vec.end(), absolute_value<double>(), double(0), thrust::maximum<double>());
	if(max_v > max_v_old){
	  max_v_old = max_v;
	}
      }
      { // bz
	thrust::device_vector<double> vec(bzGPU, bzGPU + ncells);
	max_v = thrust::transform_reduce(vec.begin(), vec.end(), absolute_value<double>(), double(0), thrust::maximum<double>());
	if(max_v > max_v_old){
	  max_v_old = max_v;
	}
      }

      // Set time step
      if(max_v_old > 0){
	dt = CFLadvective * lx / (max_v_old * mx);
      }
      else{
	dt = CFLadvective * lx / (1.0 * mx);
      }
      if( dt > (0.1 * densfluid * lx * lx / (shearviscosity*mx*mx)) ){
	dt = 0.1 * densfluid * lx * lx / (shearviscosity*mx*mx);
      }
      if( dt > (0.1 * lx * lx / (diffusion*mx*mx)) ){
	dt = 0.1 * lx * lx / (diffusion*mx*mx);
      }
      
      // double max_host;
      // kernelMaxElement<<<1,1>>>(vxGPU,vyGPU,vzGPU,bxGPU,byGPU,bzGPU);
      // cudaMemcpyFromSymbol(&max_host, max_dev, sizeof(double));
      // if(max_host > 0){
      // 	dt = CFLadvective * lx / (max_host * mx);
      // }
      // else{
      // 	dt = CFLadvective * lx / (1.0 * mx);
      // }
      
      cutilSafeCall(cudaMemcpyToSymbol(dtGPU,&dt,sizeof(double)));
      currentTime += (long double)dt;
    }

    // FIRST SUB-STEP
    kernelConstructWMHD_explicit<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU,
								bxGPU,byGPU,bzGPU,
								vxZ,vyZ,vzZ,
								WxZ,WyZ,WzZ);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,-1);//W
    filterExponential<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
    filterExponential<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF);
    projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
    projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF);
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,1);
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_INVERSE);
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxPredictionGPU,vyPredictionGPU,vzPredictionGPU);
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,bxPredictionGPU,byPredictionGPU,bzPredictionGPU);


    // SECOND SUB-STEP
    kernelConstructWMHD_explicit<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,vyPredictionGPU,vzPredictionGPU,
								bxPredictionGPU,byPredictionGPU,bzPredictionGPU,
								vxZ,vyZ,vzZ,
								WxZ,WyZ,WzZ);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,-1);//W
    filterExponential<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
    filterExponential<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF);
    projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
    projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF);
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,1);
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_INVERSE);
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxPredictionGPU,vyPredictionGPU,vzPredictionGPU);
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,bxPredictionGPU,byPredictionGPU,bzPredictionGPU);
    addVectorFields<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU,
     						   vxPredictionGPU,vyPredictionGPU,vzPredictionGPU, 
						   vxPredictionGPU,vyPredictionGPU,vzPredictionGPU, 
						   0.75, 0.25);
    addVectorFields<<<numBlocks,threadsPerBlock>>>(bxGPU,byGPU,bzGPU,
     						   bxPredictionGPU,byPredictionGPU,bzPredictionGPU, 
     						   bxPredictionGPU,byPredictionGPU,bzPredictionGPU, 
     						   0.75, 0.25);
    
    // THIRD SUB-STEP
    kernelConstructWMHD_explicit<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,vyPredictionGPU,vzPredictionGPU,
								bxPredictionGPU,byPredictionGPU,bzPredictionGPU,
								vxZ,vyZ,vzZ,
								WxZ,WyZ,WzZ);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_FORWARD);//W
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_FORWARD);//W
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,-1);//W
    filterExponential<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
    filterExponential<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF);
    projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
    projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF);
    kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
    cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
    kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,1);
    cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_INVERSE);
    cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_INVERSE);
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxPredictionGPU,vyPredictionGPU,vzPredictionGPU);
    doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,bxPredictionGPU,byPredictionGPU,bzPredictionGPU);
    addVectorFields<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU,
     						   vxPredictionGPU,vyPredictionGPU,vzPredictionGPU, 
						   vxGPU,vyGPU,vzGPU, 
						   1.0/3.0, 2.0/3.0);
    addVectorFields<<<numBlocks,threadsPerBlock>>>(bxGPU,byGPU,bzGPU,
     						   bxPredictionGPU,byPredictionGPU,bzPredictionGPU, 
     						   bxGPU,byGPU,bzGPU, 
						   1.0/3.0, 2.0/3.0);

    if(0){
      //Copy velocities to complex variable
      doubleToDoubleComplex<<<numBlocks,threadsPerBlock>>>(vxGPU,vyGPU,vzGPU,vxZ,vyZ,vzZ);
      doubleToDoubleComplex<<<numBlocks,threadsPerBlock>>>(bxGPU,byGPU,bzGPU,WxZ,WyZ,WzZ);

      //Take velocities to fourier space
      cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
      cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
      cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
      kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);
      cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_FORWARD);//W
      cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_FORWARD);//W
      cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_FORWARD);//W
      kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,-1);
      //Project into divergence free space
      // initializeMHD<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,WxZ,WyZ,WzZ,pF);
      projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
      projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF);


      //Take velocities to real space
      kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
      cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
      cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
      cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
      kernelShift<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,pF,1);
      cufftExecZ2Z(FFT,WxZ,WxZ,CUFFT_INVERSE);
      cufftExecZ2Z(FFT,WyZ,WyZ,CUFFT_INVERSE);
      cufftExecZ2Z(FFT,WzZ,WzZ,CUFFT_INVERSE);

      //Copy velocities to real variables
      doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxGPU,vyGPU,vzGPU);
      doubleComplexToDoubleNormalized<<<numBlocks,threadsPerBlock>>>(WxZ,WyZ,WzZ,bxGPU,byGPU,bzGPU);
      //---------------------------------------------------------
    }

    
    step++;
    if(!(step%samplefreq)&&(step>0)){
      cout << "MHD " << step << " dt = " << dt << " current time = " << currentTime << endl;
      if(!gpuToHostMHD()) return 0;
      if(!saveFunctionsSchemeMHD(1)) return 0;
    }
  }

  freeRandomNumbersGPU();
  //Free FFT
  cufftDestroy(FFT);

  return 1;
}


