// Filename: firstStepQuasiNeutrallyBuoyant.cu
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



//
//
// For the first time step we use 
// the mid-point rule for the advection
//
// Predictor step, n to n+1
// Advection at n+1/2
// Corrector step from n to n+1
//
//



bool firstStepQuasiNeutrallyBuoyant(int numBlocksNeighbors,
				    int threadsPerBlockNeighbors,
				    int numBlocksParticles,
				    int threadsPerBlockParticles,
				    int numBlocks,
				    int threadsPerBlock,
				    size_t numberRandom,
				    cufftHandle FFT,
				    long long &step){
  

  //
  //
  //
  // Predictor step, from n to n+1
  //
  //
  //

  //initialize vxboundaryPredictionGPU to zero
  setArrayToZeroInput<<<numBlocksParticles,threadsPerBlockParticles>>>(vxboundaryPredictionGPU,
								       vyboundaryPredictionGPU,
								       vzboundaryPredictionGPU,
								       0);

  
  //Generate random numbers
  generateRandomNumbers(numberRandom);
    
  //Clear neighbor lists
  countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
  
  //Spread particle advection "m_e*nabla*S^n*(uu)^n"
  kernelSpreadParticlesAdvection<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										  rycellGPU,
										  rzcellGPU,
										  vxboundaryGPU,
										  vyboundaryGPU,
										  vzboundaryGPU,
										  fxboundaryGPU,
										  fyboundaryGPU,
										  fzboundaryGPU,
										  pc,errorKernel);
  
  //Add the particle advection and save in vxPredictionGPU
  kernelAddParticleAdvection<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
							    vyPredictionGPU,
							    vzPredictionGPU,
							    fxboundaryGPU,
							    fyboundaryGPU,
							    fzboundaryGPU);

  //Clear neighbor lists
  countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
  
  //Update particle positions q^{n+1/2}
  findNeighborParticlesQuasiNeutrallyBuoyant_1<<<numBlocksParticles,threadsPerBlockParticles>>>
    (pc,
     errorKernel,
     rxcellGPU,
     rycellGPU,
     rzcellGPU,
     rxboundaryGPU,  //q^{n}
     ryboundaryGPU,
     rzboundaryGPU,
     rxboundaryPredictionGPU, //q^{n+1/2}
     ryboundaryPredictionGPU,
     rzboundaryPredictionGPU,
     vxGPU, //v^n
     vyGPU,
     vzGPU);
       
  //Load textures with particles position q^{n+1/2}
  cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
    
    
  //Fill "countparticlesincellX" lists
  //and spread particle force F 
  kernelSpreadParticlesForce<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
									      rycellGPU,
									      rzcellGPU,
									      fxboundaryGPU,
									      fyboundaryGPU,
									      fzboundaryGPU,
									      pc,errorKernel,
									      bFV);




  //Construct vector W
  // W = v^n + 0.5*dt*nu*L*v^n + Advection(v^n) + (dt/rho)*f^n_{noise} + dt*SF/rho + dt*(m_e/rho)*div*Suu
  //and save advection
  kernelConstructWQuasiNeutrallyBuoyant<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
								       vyPredictionGPU,
								       vzPredictionGPU,
								       vxZ,//W
								       vyZ,//W
								       vzZ,//W
								       dRand,
								       fxboundaryGPU,
								       fyboundaryGPU,
								       fzboundaryGPU,
								       advXGPU,
								       advYGPU,
								       advZGPU);
  
  //Calculate velocity prediction with incompressibility "\tilde{v}^{n+1}"
  //Go to fourier space
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
  //Apply shift for the staggered grid
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
  //Update fluid velocity
  kernelUpdateVIncompressible<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxZ,vyZ,vzZ,pF);//W
  //Apply shift for the staggered grid
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  //Come back to real space
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

  //Store velocity prediction "\tilde{v}^{n+1}" 
  predictionVQuasiNeutrallyBuoyant<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,
								  vxGPU,
								  vyGPU,
								  vzGPU,
								  vxPredictionGPU,
								  vyPredictionGPU,
								  vzPredictionGPU);

  //Load textures with velocity prediction "\tilde{v}^{n+1}"
  cutilSafeCall( cudaBindTexture(0,texVxGPU,vxPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVyGPU,vyPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVzGPU,vzPredictionGPU,ncells*sizeof(double)));


  //Calculate velocity correction "\Delta v^{k=1}"
  //First calculate term "(u^n - J\tilde{v}^{n+1})" with prefactors
  kernelCorrectionVQuasiNeutrallyBuoyantSemiImplicitMix_1<<<numBlocksParticles,threadsPerBlockParticles>>>
    (rxcellGPU,
     rycellGPU,
     rzcellGPU,
     vxGPU,
     vyGPU,
     vzGPU,
     fxboundaryGPU,
     fyboundaryGPU,
     fzboundaryGPU,
     vxboundaryGPU,
     vyboundaryGPU,
     vzboundaryGPU,
     vxboundaryPredictionGPU,
     vyboundaryPredictionGPU,
     vzboundaryPredictionGPU);


  //Second spread perturbation with S
  kernelCorrectionVQuasiNeutrallyBuoyant_2<<<numBlocks,threadsPerBlock>>>(vxZ,
									  vyZ,
									  vzZ,
									  fxboundaryGPU,
									  fyboundaryGPU,
									  fzboundaryGPU);

  //Third apply incompressibility on "\Delta v^{k=1}", store it in vxZ.x
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);
  projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
  //Store "\Delta v^{k=1}" in vxZ
  normalizeFieldComplexRealPart<<<numBlocks,threadsPerBlock>>>(vxZ,
							       vyZ,
							       vzZ);

      
  //Calculate velocity correction "\Delta v^{k=2}"
  //First interpolate and spread m_e * S(u^n - J\tilde{v}^{n+1}) / rho
  kernelCorrectionVQuasiNeutrallyBuoyantSemiImplicit_1<<<numBlocksParticles,threadsPerBlockParticles>>>
    (rxcellGPU,
     rycellGPU,
     rzcellGPU,
     vxGPU,
     vyGPU,
     vzGPU,
     fxboundaryGPU,
     fyboundaryGPU,
     fzboundaryGPU,
     vxboundaryGPU,
     vyboundaryGPU,
     vzboundaryGPU,
     vxboundaryPredictionGPU,
     vyboundaryPredictionGPU,
     vzboundaryPredictionGPU);

  //Interpolate and spread velocity correction \Delta v^{k=1}
  //and store together with m_e * SJ(u^n - \tilde{v}^{n+1}) / rho
  kernelCorrectionVQuasiNeutrallyBuoyantSemiImplicitTEST4_2<<<numBlocksParticles,threadsPerBlockParticles>>>
    (rxcellGPU,
     rycellGPU,
     rzcellGPU,
     vxZ,
     vyZ,
     vzZ,
     fxboundaryGPU,
     fyboundaryGPU,
     fzboundaryGPU,
     vxboundaryGPU,
     vyboundaryGPU,
     vzboundaryGPU);


    
  //Update particle velocity
  //"u^{n+1/2} = 0.5 * (u^n + u^{n+1})
  updateParticlesQuasiNeutrallyBuoyantSemiImplicitTEST4_2<<<numBlocksParticles,threadsPerBlockParticles>>>
    (pc,
     errorKernel,
     rxboundaryGPU,
     ryboundaryGPU,
     rzboundaryGPU,
     vxboundaryGPU,
     vyboundaryGPU,
     vzboundaryGPU,
     vxboundaryPredictionGPU,
     vyboundaryPredictionGPU,
     vzboundaryPredictionGPU,
     rxcellGPU,
     rycellGPU,
     rzcellGPU,
     vxPredictionGPU,
     vyPredictionGPU,
     vzPredictionGPU,
     vxZ,
     vyZ,
     vzZ);
    
  //Load textures with velocity  "vx^n"
  cutilSafeCall( cudaBindTexture(0,texVxGPU,vxGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVyGPU,vyGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVzGPU,vzGPU,ncells*sizeof(double)));


  //construct vector W
  //W=m_e * SJ(u^n - \tilde{v}^{n+1})
  kernelConstructWQuasiNeutrallyBuoyantSemiImplicit<<<numBlocks,threadsPerBlock>>>(vxZ,//W
										   vyZ,//W
										   vzZ,//W
										   fxboundaryGPU,
										   fyboundaryGPU,
										   fzboundaryGPU);
     
  //Calculate velocity correction with incompressibility "\tilde{v}^{n+1}"
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
  kernelUpdateVIncompressible<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxZ,vyZ,vzZ,pF);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);


  //Calculate v^{n+0.5}
  //Store it in vxPredictionGPU
  calculateVelocityAtHalfTimeStep<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
								 vyPredictionGPU,
								 vzPredictionGPU,
								 vxGPU,
								 vyGPU,
								 vzGPU,
								 vxZ,
								 vyZ,
								 vzZ);

  //Calculate 0.5*dt*nu*L*\Delta v^{k=2} and store it
  //in vxZ.y
  laplacianDeltaV_2<<<numBlocks,threadsPerBlock>>>(vxZ,
  						   vyZ,
  						   vzZ);

  //Save in rxboundaryPredictionGPU
  //Update particle position q^{n+1/2} = q^n + 0.5 * dt * J^{n+1/2} v^{n+1/2}
  //q^{n+1/2} = 0.5 * (q^n + q^{n+1}) = q^n + 0.5 * dt * J^{n+1/2} v^{n+1/2}
  findNeighborParticlesQuasiNeutrallyBuoyantTEST4_3<<<numBlocksParticles,threadsPerBlockParticles>>>
    (pc, 
     errorKernel,
     rxcellGPU,
     rycellGPU,
     rzcellGPU,
     rxboundaryGPU, //q^{n} and q^{n+1}
     ryboundaryGPU,
     rzboundaryGPU,
     rxboundaryPredictionGPU, //q^{n+1/2}
     ryboundaryPredictionGPU, 
     rzboundaryPredictionGPU,
     vxPredictionGPU, // v^{n+1/2}
     vyPredictionGPU, 
     vzPredictionGPU);
  
  //
  //
  //
  // Calculate advection at n+1/2
  //
  //
  //

  
  //Calculate fluid advection
  //and save it in vxZ.x
  cutilSafeCall( cudaBindTexture(0,texVxGPU,vxPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVyGPU,vyPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVzGPU,vzPredictionGPU,ncells*sizeof(double)));

  calculateAdvectionFluid<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
  							 vyPredictionGPU,
  							 vzPredictionGPU,
							 vxZ,
							 vyZ,
							 vzZ);
  //Load textures with velocity  "vx^n"
  cutilSafeCall( cudaBindTexture(0,texVxGPU,vxGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVyGPU,vyGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVzGPU,vzGPU,ncells*sizeof(double)));
  
  //Clear neighbor lists
  countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);

  //Spread particle advection "m_e*nabla*S^n*(uu)^n"
  kernelSpreadParticlesAdvection<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										  rycellGPU,
										  rzcellGPU,
										  vxboundaryPredictionGPU,
										  vyboundaryPredictionGPU,
										  vzboundaryPredictionGPU,
										  fxboundaryGPU,
										  fyboundaryGPU,
										  fzboundaryGPU,
										  pc,errorKernel);

  //Save the particle advection in vxPredictionGPU
  kernelAddParticleAdvection_2<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
  							      vyPredictionGPU,
  							      vzPredictionGPU,
							      vxZ,
							      vyZ,
							      vzZ,
  							      fxboundaryGPU,
  							      fyboundaryGPU,
  							      fzboundaryGPU);
  
  
  //Calculate 0.5*dt*nu*J*L*\Delta v^{k=2} and store it
  //in vxboundaryPredictionGPU
  interpolateLaplacianDeltaV_2<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
										rycellGPU,
										rzcellGPU,
										vxZ,
										vyZ,
										vzZ,
										rxboundaryPredictionGPU,
										ryboundaryPredictionGPU,
										rzboundaryPredictionGPU,
										vxboundaryPredictionGPU,
										vyboundaryPredictionGPU,
										vzboundaryPredictionGPU);
      
  
  //
  //
  //
  //Corrector step
  //
  //
  //

  //Load textures with particles position q^{n}
  cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryGPU,(nboundary+np)*sizeof(double)));

  //Clear neighbor lists
  countToZero<<<numBlocksNeighbors,threadsPerBlockNeighbors>>>(pc);
  
  
  //Load textures with particles position q^{n}
  cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryGPU,(nboundary+np)*sizeof(double)));

  //Update particle positions q^{n+1/2}
  findNeighborParticlesQuasiNeutrallyBuoyant_1<<<numBlocksParticles,threadsPerBlockParticles>>>
    (pc, 
     errorKernel,
     rxcellGPU,
     rycellGPU,
     rzcellGPU,
     rxboundaryGPU,  //q^{n}
     ryboundaryGPU, 
     rzboundaryGPU,
     rxboundaryPredictionGPU, //q^{n+1/2}
     ryboundaryPredictionGPU, 
     rzboundaryPredictionGPU,
     vxGPU, //v^n
     vyGPU, 
     vzGPU);
       
  //Load textures with particles position q^{n+1/2}
  cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryPredictionGPU,(nboundary+np)*sizeof(double)));
    
    
  //Fill "countparticlesincellX" lists
  //and spread particle force F 
  kernelSpreadParticlesForce<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
									      rycellGPU,
									      rzcellGPU,
									      fxboundaryGPU,
									      fyboundaryGPU,
									      fzboundaryGPU,
									      pc,errorKernel,
									      bFV);

  //Construct vector W
  // W = v^n + 0.5*dt*nu*L*v^n + Advection(v^n) + (dt/rho)*f^n_{noise} + dt*SF/rho + dt*(m_e/rho)*div*Suu
  //and save advection
  kernelConstructWQuasiNeutrallyBuoyantTEST5_2<<<numBlocks,threadsPerBlock>>>(vxPredictionGPU,
									      vyPredictionGPU,
									      vzPredictionGPU,
									      vxZ,//W
									      vyZ,//W
									      vzZ,//W
									      dRand,
									      fxboundaryGPU,
									      fyboundaryGPU,
									      fzboundaryGPU,
									      advXGPU,
									      advYGPU,
									      advZGPU);
									      
    
  //Calculate velocity prediction with incompressibility "\tilde{v}^{n+1}"
  //Go to fourier space
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
  //Apply shift for the staggered grid
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
  //Update fluid velocity
  kernelUpdateVIncompressible<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxZ,vyZ,vzZ,pF);//W
  //Apply shift for the staggered grid
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  //Come back to real space
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);

  
  //Store velocity prediction "\tilde{v}^{n+1}" 
  predictionVQuasiNeutrallyBuoyant<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,
								  vxGPU,
								  vyGPU,
								  vzGPU,
								  vxPredictionGPU,
								  vyPredictionGPU,
								  vzPredictionGPU);

  //Load textures with velocity prediction "\tilde{v}^{n+1}"
  cutilSafeCall( cudaBindTexture(0,texVxGPU,vxPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVyGPU,vyPredictionGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVzGPU,vzPredictionGPU,ncells*sizeof(double)));


  //Calculate velocity correction "\Delta v^{k=1}"
  //First calculate term "(u^n - J\tilde{v}^{n+1})" with prefactors
  kernelCorrectionVQuasiNeutrallyBuoyantSemiImplicitMix_1<<<numBlocksParticles,threadsPerBlockParticles>>>
    (rxcellGPU,
     rycellGPU,
     rzcellGPU,
     vxGPU,
     vyGPU,
     vzGPU,
     fxboundaryGPU,
     fyboundaryGPU,
     fzboundaryGPU,
     vxboundaryGPU,
     vyboundaryGPU,
     vzboundaryGPU,
     vxboundaryPredictionGPU,
     vyboundaryPredictionGPU,
     vzboundaryPredictionGPU);

  //Second spread perturbation with S
  kernelCorrectionVQuasiNeutrallyBuoyant_2<<<numBlocks,threadsPerBlock>>>(vxZ,
									  vyZ,
									  vzZ,
									  fxboundaryGPU,
									  fyboundaryGPU,
									  fzboundaryGPU);

  //Third apply incompressibility on "\Delta v^{k=1}", store it in vxZ.x
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);
  projectionDivergenceFree<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF);
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);
  //Store "\Delta v^{k=1}" in vxZ
  normalizeFieldComplexRealPart<<<numBlocks,threadsPerBlock>>>(vxZ,
							       vyZ,
							       vzZ);
    
  //Calculate velocity correction "\Delta v^{k=2}"
  //First interpolate and spread m_e * S(u^n - J\tilde{v}^{n+1}) / rho
  kernelCorrectionVQuasiNeutrallyBuoyantSemiImplicit_1<<<numBlocksParticles,threadsPerBlockParticles>>>
    (rxcellGPU,
     rycellGPU,
     rzcellGPU,
     vxGPU,
     vyGPU,
     vzGPU,
     fxboundaryGPU,
     fyboundaryGPU,
     fzboundaryGPU,
     vxboundaryGPU,
     vyboundaryGPU,
     vzboundaryGPU,
     vxboundaryPredictionGPU,
     vyboundaryPredictionGPU,
     vzboundaryPredictionGPU);

  //Interpolate and spread velocity correction \Delta v^{k=1}
  //and store together with m_e * SJ(u^n - \tilde{v}^{n+1}) / rho
  kernelCorrectionVQuasiNeutrallyBuoyantSemiImplicitTEST4_2<<<numBlocksParticles,threadsPerBlockParticles>>>
    (rxcellGPU,
     rycellGPU,
     rzcellGPU,
     vxZ,
     vyZ,
     vzZ,
     fxboundaryGPU,
     fyboundaryGPU,
     fzboundaryGPU,
     vxboundaryGPU,
     vyboundaryGPU,
     vzboundaryGPU);

  //Update particle velocity
  //"u^{n+1}
  updateParticlesQuasiNeutrallyBuoyantSemiImplicitTEST4<<<numBlocksParticles,threadsPerBlockParticles>>>
    (pc,
     errorKernel,
     rxboundaryGPU,
     ryboundaryGPU,
     rzboundaryGPU,
     vxboundaryGPU,
     vyboundaryGPU,
     vzboundaryGPU,
     rxcellGPU,
     rycellGPU,
     rzcellGPU,
     vxPredictionGPU,
     vyPredictionGPU,
     vzPredictionGPU,
     vxZ,
     vyZ,
     vzZ,
     vxboundaryPredictionGPU,
     vyboundaryPredictionGPU,
     vzboundaryPredictionGPU);
    
  //Load textures with velocity  "vx^n"
  cutilSafeCall( cudaBindTexture(0,texVxGPU,vxGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVyGPU,vyGPU,ncells*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texVzGPU,vzGPU,ncells*sizeof(double)));

  //construct vector W
  //W=m_e * SJ(u^n - \tilde{v}^{n+1})
  kernelConstructWQuasiNeutrallyBuoyantSemiImplicit<<<numBlocks,threadsPerBlock>>>(vxZ,//W
										   vyZ,//W
										   vzZ,//W
										   fxboundaryGPU,
										   fyboundaryGPU,
										   fzboundaryGPU);
     
  //Calculate velocity correction with incompressibility "\tilde{v}^{n+1}"
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_FORWARD);//W
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_FORWARD);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,-1);//W
  kernelUpdateVIncompressible<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,vxZ,vyZ,vzZ,pF);//W
  kernelShift<<<numBlocks,threadsPerBlock>>>(vxZ,vyZ,vzZ,pF,1);
  cufftExecZ2Z(FFT,vxZ,vxZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vyZ,vyZ,CUFFT_INVERSE);
  cufftExecZ2Z(FFT,vzZ,vzZ,CUFFT_INVERSE);


  //Calculate v^{n+0.5}
  //Store it in vxGPU
  calculateVelocityAtHalfTimeStep<<<numBlocks,threadsPerBlock>>>(vxGPU,
								 vyGPU,
								 vzGPU,
								 vxPredictionGPU,
								 vyPredictionGPU,
								 vzPredictionGPU,
								 vxZ,
								 vyZ,
								 vzZ);
  
  //Update particle position q^{n+1} = q^n + dt * J^{n+1/2} v^{n+1/2}
  findNeighborParticlesQuasiNeutrallyBuoyantTEST4_2<<<numBlocksParticles,threadsPerBlockParticles>>>
    (pc, 
     errorKernel,
     rxcellGPU,
     rycellGPU,
     rzcellGPU,
     rxboundaryGPU, //q^{n} and q^{n+1}
     ryboundaryGPU,
     rzboundaryGPU,
     rxboundaryPredictionGPU, //q^{n+1/2}
     ryboundaryPredictionGPU, 
     rzboundaryPredictionGPU,
     vxGPU, // v^{n+1/2}
     vyGPU, 
     vzGPU);




  //Calculate 0.5*dt*nu*L*\Delta v^{k=2} and store it
  //in vxGPU
  laplacianDeltaV<<<numBlocks,threadsPerBlock>>>(vxZ,
						 vyZ,
						 vzZ,
						 vxGPU,
						 vyGPU,
						 vzGPU);
  
  //Calculate 0.5*dt*nu*J*L*\Delta v^{k=2} and store it
  //in vxboundaryPredictionGPU
  interpolateLaplacianDeltaV<<<numBlocksParticles,threadsPerBlockParticles>>>(rxcellGPU,
									      rycellGPU,
									      rzcellGPU,
									      vxGPU,
									      vyGPU,
									      vzGPU,
									      rxboundaryPredictionGPU,
									      ryboundaryPredictionGPU,
									      rzboundaryPredictionGPU,
									      vxboundaryPredictionGPU,
									      vyboundaryPredictionGPU,
									      vzboundaryPredictionGPU);
  

  //Load textures with particles position q^{n}
  cutilSafeCall( cudaBindTexture(0,texrxboundaryGPU,rxboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texryboundaryGPU,ryboundaryGPU,(nboundary+np)*sizeof(double)));
  cutilSafeCall( cudaBindTexture(0,texrzboundaryGPU,rzboundaryGPU,(nboundary+np)*sizeof(double)));


  //Update fluid velocity
  updateFluidQuasiNeutrallyBuoyantSemiImplicit<<<numBlocks,threadsPerBlock>>>(vxGPU,
									      vyGPU,
									      vzGPU,
									      vxPredictionGPU,
									      vyPredictionGPU,
									      vzPredictionGPU,
									      vxZ,
									      vyZ,
									      vzZ);




  step++;
  if(!(step%samplefreq)&&(step>0)){
    cout << "INCOMPRESSIBLE BOUNDARY  ;) " << step << endl;
    if(!gpuToHostIncompressibleBoundaryRK2(numBlocksParticles,threadsPerBlockParticles)) return 0;
    if(!saveFunctionsSchemeIncompressibleBoundary(1,step)) return 0;
  }
  
  
  
  
  

  //cutilSafeCall(cudaMemcpyToSymbol(dtGPU,&dt,sizeof(double)));
  
  
  
  
  
  return 1;
}

