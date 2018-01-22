
//Raul added, a kernel to add a shear flow to the fluid. A sinusoidal force is added to each fluid cell. This shear flow is intended to be used for viscosity measurements.
__global__ void addShearFlowQuasi2D(cufftDoubleComplex *vxZ,
				    cufftDoubleComplex *vyZ,
				    double viscosityMeasureAmplitude){
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;
  const int ix = i%mxGPU;
	
  const double pi2 = 2.0*3.1415926535897932385;
  
  vyZ[i].x += viscosityMeasureAmplitude*sin(pi2*(ix/(double)mxGPU));
}
				   

__global__ void findNeighborListsQuasi2D(particlesincell* pc, 
					 int* errorKernel,
					 const double* rxcellGPU,
					 const double* rycellGPU,
					 const double* rzcellGPU,
					 const double* rxboundaryGPU,  //q^{n}
					 const double* ryboundaryGPU, 
					 const double* rzboundaryGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);

  int icel, np;
 
  // Particle location in cells for neighbor lists
  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    rx = rx - (int(rx*invlxGPU + 0.5*((rx>0)-(rx<0)))) * lxGPU;
    int jx   = int(rx * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    ry = ry - (int(ry*invlyGPU + 0.5*((ry>0)-(ry<0)))) * lyGPU;
    int jy   = int(ry * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    icel = jx + jy * mxNeighborsGPU;
  }
  np = atomicAdd(&pc->countPartInCellNonBonded[icel],1);
  if(np >= maxNumberPartInCellNonBondedGPU){
    errorKernel[0] = 1;
    errorKernel[4] = 1;
    return;
  }
  pc->partInCellNonBonded[mNeighborsGPU*np+icel] = i;
}


__global__ void kernelSpreadParticlesForceQuasi2D(const double* rxcellGPU, 
						  const double* rycellGPU, 
						  cufftDoubleComplex* vxZ,
						  cufftDoubleComplex* vyZ,
						  const bondedForcesVariables* bFV){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double fx = 0.0;
  double fy = 0.0;
  double f;

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
 
  // INCLUDE EXTERNAL FORCES HERE
  // Example: harmonic potential 
  //  V(r) = (1/2) * k * ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
  //
  // with spring constant k=0.01
  // and x0=y0=z0=0
  //
  // fx = -0.01*rx;
  // fy = -0.01*ry;
  // fz = -0.01*rz;


  // NEW bonded forces
  if(bondedForcesGPU){
    // call function for bonded forces particle-particle
    forceBondedParticleParticleGPU_2D(i,
				      fx,
				      fy,
				      rx,
				      ry,
				      bFV);
  }
    
  double rxij, ryij, r2;
  int vecino1, vecino2, vecino3, vecino4;
  int vecinopxpy, vecinopxmy;
  int vecinomxpy, vecinomxmy;  
  int icel;
  double r;

  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    icel = jx + jy * mxNeighborsGPU;
  }
  
  int np;
  if(computeNonBondedForcesGPU){
    //Particles in Cell i
    np = tex1Dfetch(texCountParticlesInCellNonBonded,icel);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+icel);
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }  
    //Particles in Cell vecino1
    vecino1 = tex1Dfetch(texneighbor1GPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino1);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino1);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecino2
    vecino2 = tex1Dfetch(texneighbor2GPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino2);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino2);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecino3
    vecino3 = tex1Dfetch(texneighbor3GPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino3);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino3);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecino4
    vecino4 = tex1Dfetch(texneighbor4GPU, icel);
    //printf("VECINO %i %i \n",icel,vecino4);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecino4);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecino4);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecinopxpy
    vecinopxpy = tex1Dfetch(texneighborpxpyGPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxpy);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxpy);    
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecinopxmy
    vecinopxmy = tex1Dfetch(texneighborpxmyGPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinopxmy);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinopxmy);
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecinomxpy
    vecinomxpy = tex1Dfetch(texneighbormxpyGPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxpy);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxpy);
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
    //Particles in Cell vecinomxmy
    vecinomxmy = tex1Dfetch(texneighbormxmyGPU, icel);
    np = tex1Dfetch(texCountParticlesInCellNonBonded,vecinomxmy);
    for(int j=0;j<np;j++){
      int particle = tex1Dfetch(texPartInCellNonBonded,mNeighborsGPU*j+vecinomxmy);
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      r2 = rxij*rxij + ryij*ryij ;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
    }
  }


  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    icel  = jx + jy * mxGPU;
  }

  // printf("rx = %f, ry = %f, icel = %i \n", rx, ry, icel);

  // Loop over neighbor cells
  {
    double rx_distance, ry_distance, norm;
    int kx, ky, kx_neigh, ky_neigh, icel_neigh;
    ky = icel / mxGPU;
    kx = icel % mxGPU;
    // printf("kx = %i,  ky = %i \n", kx, ky);
    for(int ix=-kernelWidthGPU; ix<=kernelWidthGPU; ix++){
      kx_neigh = (kx + ix + mxGPU) % mxGPU;
      rx_distance = rx - (kx_neigh * lxGPU / mxGPU) + lxGPU * 0.5;
      rx_distance = rx_distance - int(rx_distance*invlxGPU + 0.5*((rx_distance>0)-(rx_distance<0)))*lxGPU;

      for(int iy=-kernelWidthGPU; iy<=kernelWidthGPU; iy++){
	ky_neigh = (ky + iy + myGPU) % myGPU;
	icel_neigh = kx_neigh + ky_neigh * mxGPU;

	ry_distance = ry - (ky_neigh * lyGPU / myGPU) + lyGPU * 0.5;
	ry_distance = ry_distance - int(ry_distance*invlyGPU + 0.5*((ry_distance>0)-(ry_distance<0)))*lyGPU;

	r2 = rx_distance*rx_distance + ry_distance*ry_distance;
	norm = GaussianKernel2DGPU(r2, GaussianVarianceGPU);

	atomicAdd(&vxZ[icel_neigh].x, norm * fx);
	atomicAdd(&vyZ[icel_neigh].x, norm * fy);
      }
    }
  } 
}



__global__ void kernelSpreadThermalDriftQuasi2D(const double* rxcellGPU, 
						const double* rycellGPU, 
						cufftDoubleComplex* vxZ,
						cufftDoubleComplex* vyZ,
						const double *dRand,
						const double* rxboundaryGPU,  // q^{} to interpolate
						const double* ryboundaryGPU, 
						int offset){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  // double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  // double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rx = rxboundaryGPU[i];
  double ry = ryboundaryGPU[i];
    
  double r2;
  int icel;
  double r;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    icel   = jx  + jy  * mxGPU;
  }


  // Loop over neighbor cells
  // for(int iDrift=0; iDrift < nDriftGPU; iDrift++){
  //   int offsetDrift = offset + 2 * npGPU * iDrift;
  //   double rx_distance_p, ry_distance_p, rx_distance_m, ry_distance_m;
  //   double norm;
  //   int kx, ky, kx_neigh, ky_neigh, icel_neigh;
  //   ky = icel / mxGPU;
  //   kx = icel % mxGPU;
  //   for(int ix=-kernelWidthGPU; ix<=kernelWidthGPU; ix++){
  //     kx_neigh = (kx + ix + mxGPU) % mxGPU;

  //     rx_distance_p = (rx + 0.5*deltaRFDGPU*dRand[offsetDrift+i]) - (kx_neigh * lxGPU / mxGPU) + lxGPU * 0.5;
  //     rx_distance_p = rx_distance_p - int(rx_distance_p*invlxGPU + 0.5*((rx_distance_p>0)-(rx_distance_p<0)))*lxGPU;
  //     rx_distance_m = (rx - 0.5*deltaRFDGPU*dRand[offsetDrift+i]) - (kx_neigh * lxGPU / mxGPU) + lxGPU * 0.5;
  //     rx_distance_m = rx_distance_m - int(rx_distance_m*invlxGPU + 0.5*((rx_distance_m>0)-(rx_distance_m<0)))*lxGPU;

  //     for(int iy=-kernelWidthGPU; iy<=kernelWidthGPU; iy++){
  // 	ky_neigh = (ky + iy + myGPU) % myGPU;
  // 	icel_neigh = kx_neigh + ky_neigh * mxGPU;

  // 	ry_distance_p = (ry + 0.5*deltaRFDGPU*dRand[offsetDrift+npGPU+i]) - (ky_neigh * lyGPU / myGPU) + lyGPU * 0.5;
  // 	ry_distance_p = ry_distance_p - int(ry_distance_p*invlyGPU + 0.5*((ry_distance_p>0)-(ry_distance_p<0)))*lyGPU;
  // 	ry_distance_m = (ry - 0.5*deltaRFDGPU*dRand[offsetDrift+npGPU+i]) - (ky_neigh * lyGPU / myGPU) + lyGPU * 0.5;
  // 	ry_distance_m = ry_distance_m - int(ry_distance_m*invlyGPU + 0.5*((ry_distance_m>0)-(ry_distance_m<0)))*lyGPU;

  // 	// Spread drift kT*S(q+0.5*delta*W)*W
  // 	r2 = rx_distance_p*rx_distance_p + ry_distance_p*ry_distance_p;
  // 	norm = GaussianKernel2DGPU(r2, GaussianVarianceGPU) * temperatureGPU / (nDriftGPU * deltaRFDGPU);

  // 	atomicAdd(&vxZ[icel_neigh].x, norm * dRand[offsetDrift + i]);
  // 	atomicAdd(&vyZ[icel_neigh].x, norm * dRand[offsetDrift + npGPU + i]);
	
  // 	// Spread drift -kT*S(q-0.5*delta*W)*W
  // 	r2 = rx_distance_m*rx_distance_m + ry_distance_m*ry_distance_m;
  // 	norm = GaussianKernel2DGPU(r2, GaussianVarianceGPU) * temperatureGPU / (nDriftGPU * deltaRFDGPU);
	
  // 	atomicAdd(&vxZ[icel_neigh].x, -norm * dRand[offsetDrift + i]);
  // 	atomicAdd(&vyZ[icel_neigh].x, -norm * dRand[offsetDrift + npGPU + i]);       
  //     }
  //   }
  // } 


  { // Deterministic spreading
    double rx_distance_p, ry_distance_p;
    double norm;
    int kx, ky, kx_neigh, ky_neigh, icel_neigh;
    ky = icel / mxGPU;
    kx = icel % mxGPU;
    for(int ix=-kernelWidthGPU; ix<=kernelWidthGPU; ix++){
      kx_neigh = (kx + ix + mxGPU) % mxGPU;

      rx_distance_p = rx  - (kx_neigh * lxGPU / mxGPU) + lxGPU * 0.5;
      rx_distance_p = rx_distance_p - int(rx_distance_p*invlxGPU + 0.5*((rx_distance_p>0)-(rx_distance_p<0)))*lxGPU;

      for(int iy=-kernelWidthGPU; iy<=kernelWidthGPU; iy++){
  	ky_neigh = (ky + iy + myGPU) % myGPU;
  	icel_neigh = kx_neigh + ky_neigh * mxGPU;

  	ry_distance_p = ry - (ky_neigh * lyGPU / myGPU) + lyGPU * 0.5;
  	ry_distance_p = ry_distance_p - int(ry_distance_p*invlyGPU + 0.5*((ry_distance_p>0)-(ry_distance_p<0)))*lyGPU;

  	// Spread drift 
  	r2 = rx_distance_p*rx_distance_p + ry_distance_p*ry_distance_p;
  	norm = GaussianKernel2DGPU(r2, GaussianVarianceGPU) * temperatureGPU / GaussianVarianceGPU;
	
  	atomicAdd(&vxZ[icel_neigh].x, -norm * rx_distance_p);
  	atomicAdd(&vyZ[icel_neigh].x, -norm * ry_distance_p);
      }
    }
  }
}



__global__ void kernelUpdateVQuasi2D(cufftDoubleComplex *vxZ, 
				     cufftDoubleComplex *vyZ){
				     
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int wx, wy;
  wy = i / mxGPU;
  wx = i % mxGPU; 

  if(wx > mxGPU / 2){
    wx -= mxGPU;
  }
  if(wy > myGPU / 2){
    wy -= myGPU;
  }

  bool xHalf = (mxGPU % 2) == 0 && (wx == mxGPU / 2);
  bool yHalf = (myGPU % 2) == 0 && (wy == myGPU / 2);
  if(xHalf and wy < 0){
    wx *= -1;
  }
  else if(yHalf and wx < 0){
    wy *= -1;
  }

  double pi = 3.1415926535897932385;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  double k_inv = rsqrt(kx*kx + ky*ky);
  double k3_inv = k_inv * k_inv * k_inv;
  cufftDoubleComplex Wx, Wy;

  if(i == 0){
    vxZ[i].x = 0;
    vxZ[i].y = 0;
    vyZ[i].x = 0;
    vyZ[i].y = 0;
  }
  else{
    Wx.x = (k3_inv / shearviscosityGPU) * (0.5 * ky * (ky*vxZ[i].x - kx*vyZ[i].x) + 0.25 * kx * (kx*vxZ[i].x + ky*vyZ[i].x));
    Wx.y = (k3_inv / shearviscosityGPU) * (0.5 * ky * (ky*vxZ[i].y - kx*vyZ[i].y) + 0.25 * kx * (kx*vxZ[i].y + ky*vyZ[i].y));   
    Wy.x = (k3_inv / shearviscosityGPU) * (0.5 * (-kx) * (ky*vxZ[i].x - kx*vyZ[i].x) + 0.25 * ky * (kx*vxZ[i].x + ky*vyZ[i].x));
    Wy.y = (k3_inv / shearviscosityGPU) * (0.5 * (-kx) * (ky*vxZ[i].y - kx*vyZ[i].y) + 0.25 * ky * (kx*vxZ[i].y + ky*vyZ[i].y));

    vxZ[i].x = Wx.x;
    vxZ[i].y = Wx.y;
    vyZ[i].x = Wy.x;
    vyZ[i].y = Wy.y;
  }
}


__global__ void kernelUpdateVRPYQuasi2D(cufftDoubleComplex *vxZ, 
					cufftDoubleComplex *vyZ){
				     
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int wx, wy;
  wy = i / mxGPU;
  wx = i % mxGPU; 

  if(wx > mxGPU / 2){
    wx -= mxGPU;
  }
  if(wy > myGPU / 2){
    wy -= myGPU;
  }

  bool xHalf = (mxGPU % 2) == 0 && (wx == mxGPU / 2);
  bool yHalf = (myGPU % 2) == 0 && (wy == myGPU / 2);
  if(xHalf and wy < 0){
    wx *= -1;
  }
  else if(yHalf and wx < 0){
    wy *= -1;
  }

  double pi = 3.1415926535897932385;
  double pi_half_inv = 0.56418958354775628695;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  double k_inv = rsqrt(kx*kx + ky*ky);
  double k3_inv = k_inv * k_inv * k_inv;
  cufftDoubleComplex Wx, Wy;
  double f, g;
  double k = 1.0 / k_inv;
  double sigma = sqrt(GaussianVarianceGPU);
  double k_norm = sigma * k;

  if(i==0){
    vxZ[i].x = 0;
    vxZ[i].y = 0;
    vyZ[i].x = 0;
    vyZ[i].y = 0;
  }
  else{
    f = 0.5 * (erfc(k_norm) * (0.5 + k_norm*k_norm)*exp(k_norm*k_norm) - pi_half_inv * k_norm);
    g = 0.5 * erfc(k_norm) * exp(k_norm * k_norm) ;

    Wx.x = (k3_inv / shearviscosityGPU) * (g *   ky  * (ky*vxZ[i].x - kx*vyZ[i].x) + f * kx * (kx*vxZ[i].x + ky*vyZ[i].x));
    Wx.y = (k3_inv / shearviscosityGPU) * (g *   ky  * (ky*vxZ[i].y - kx*vyZ[i].y) + f * kx * (kx*vxZ[i].y + ky*vyZ[i].y));   
    Wy.x = (k3_inv / shearviscosityGPU) * (g * (-kx) * (ky*vxZ[i].x - kx*vyZ[i].x) + f * ky * (kx*vxZ[i].x + ky*vyZ[i].x));
    Wy.y = (k3_inv / shearviscosityGPU) * (g * (-kx) * (ky*vxZ[i].y - kx*vyZ[i].y) + f * ky * (kx*vxZ[i].y + ky*vyZ[i].y));

    vxZ[i].x = Wx.x;
    vxZ[i].y = Wx.y;
    vyZ[i].x = Wy.x;
    vyZ[i].y = Wy.y;
  }
}



__global__ void addStochasticVelocityQuasi2D(cufftDoubleComplex *vxZ, cufftDoubleComplex *vyZ, const double *dRand){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int wx, wy, fx, fy, shift;
  int nModes = (ncellsGPU + mxGPU + myGPU);
  wy = i / mxGPU;
  wx = i % mxGPU;
  fx = wx;
  fy = wy;
  shift = 0;

  if(wx > mxGPU / 2){
    fx = wx - mxGPU;
    wx = mxGPU - wx;
  }
  if(wy > myGPU / 2){
    fy = wy - myGPU;
    wy = myGPU - wy;
  }
  bool xHalf = (mxGPU % 2) == 0 && (wx == mxGPU / 2);
  bool yHalf = (myGPU % 2) == 0 && (wy == myGPU / 2);
  if(xHalf and fy < 0){
    fx *= -1;
  }
  else if(yHalf and fx < 0){
    fy *= -1;
  }

  double pi = 3.1415926535897932385;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  if(fx * fy < 0){
    shift = nModes / 2;
    kx *= -1.0;
  }

  int index = wy * mxGPU + wx + shift;
  double k = sqrt(kx*kx + ky*ky);
  double k3half_inv = rsqrt(k * k * k);
  double sqrtTwo_inv = rsqrt(2.0);
  cufftDoubleComplex Wx, Wy;
  double prefactor = fact1GPU;
  
  if(i==0){
    Wx.x = 0;
    Wx.y = 0;
    Wy.x = 0;
    Wy.y = 0;
  }
  else if((xHalf and wy==0) || (yHalf and wx==0) || (xHalf and yHalf)){
    Wx.x = sqrt(2.0) * prefactor * k3half_inv * (sqrtTwo_inv *   ky  * dRand[           index] + 0.5 * kx * dRand[nModes   + index]);
    Wy.x = sqrt(2.0) * prefactor * k3half_inv * (sqrtTwo_inv * (-kx) * dRand[           index] + 0.5 * ky * dRand[nModes   + index]);
    Wx.y = 0; 
    Wy.y = 0; 
  }
  else{
    Wx.x = prefactor * k3half_inv * (sqrtTwo_inv *   ky  * dRand[           index] + 0.5 * kx * dRand[nModes   + index]);
    Wy.x = prefactor * k3half_inv * (sqrtTwo_inv * (-kx) * dRand[           index] + 0.5 * ky * dRand[nModes   + index]);
    Wx.y = prefactor * k3half_inv * (sqrtTwo_inv *   ky  * dRand[nModes*2 + index] + 0.5 * kx * dRand[nModes*3 + index]);
    Wy.y = prefactor * k3half_inv * (sqrtTwo_inv * (-kx) * dRand[nModes*2 + index] + 0.5 * ky * dRand[nModes*3 + index]); 
  }

  if((fx < 0) or (fx == 0 and fy < 0)){
    Wx.y *= -1.0;
    Wy.y *= -1.0;
  }

  vxZ[i].x += Wx.x;
  vxZ[i].y += Wx.y;
  vyZ[i].x += Wy.x;
  vyZ[i].y += Wy.y;
}


__global__ void addStochasticVelocityRPYQuasi2D(cufftDoubleComplex *vxZ, 
						cufftDoubleComplex *vyZ, 
						const double *dRand, 
						const double factorScheme,
						const bool secondNoise){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int wx, wy, fx, fy, shift;
  int nModes = (ncellsGPU + mxGPU + myGPU);
  wy = i / mxGPU;
  wx = i % mxGPU;
  fx = wx;
  fy = wy;
  shift = 0;

  if(wx > mxGPU / 2){
    fx = wx - mxGPU;
    wx = mxGPU - wx;
  }
  if(wy > myGPU / 2){
    fy = wy - myGPU;
    wy = myGPU - wy;
  }
  bool xHalf = (mxGPU % 2) == 0 && (wx == mxGPU / 2);
  bool yHalf = (myGPU % 2) == 0 && (wy == myGPU / 2);
  if(xHalf and fy < 0){
    fx *= -1;
  }
  else if(yHalf and fx < 0){
    fy *= -1;
  }

  double pi = 3.1415926535897932385;
  double pi_half_inv = 0.56418958354775628695;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  if(fx * fy < 0){
    shift = nModes / 2;
    kx *= -1.0;
  }

  int index = wy * mxGPU + wx + shift;
  double k = sqrt(kx*kx + ky*ky);
  double k3half_inv = rsqrt(k * k * k);
  cufftDoubleComplex Wx, Wy;
  double prefactor = fact1GPU * factorScheme;

  double f_half, g_half;
  double sigma = sqrt(GaussianVarianceGPU);
  double k_norm = sigma * k;

  if(i==0){
    Wx.x = 0;
    Wx.y = 0;
    Wy.x = 0;
    Wy.y = 0;
  }
  else if((xHalf and wy==0) || (yHalf and wx==0) || (xHalf and yHalf)){
    f_half = sqrt( 0.5 * (erfc(k_norm) * (0.5 + k_norm*k_norm)*exp(k_norm*k_norm) - pi_half_inv * k_norm) );
    g_half = sqrt( 0.5 * erfc(k_norm) * exp(k_norm * k_norm) );

    Wx.x = sqrt(2.0) * prefactor * k3half_inv * (g_half *   ky  * dRand[           index] + f_half * kx * dRand[nModes   + index]);
    Wy.x = sqrt(2.0) * prefactor * k3half_inv * (g_half * (-kx) * dRand[           index] + f_half * ky * dRand[nModes   + index]);
    Wx.y = 0; // prefactor * k3half_inv * (g_half *   ky  * dRand[nModes*2 + index] + f_half * kx * dRand[nModes*3 + index]);
    Wy.y = 0; // prefactor * k3half_inv * (g_half * (-kx) * dRand[nModes*2 + index] + f_half * ky * dRand[nModes*3 + index]); 
  }
  else{
    f_half = sqrt( 0.5 * (erfc(k_norm) * (0.5 + k_norm*k_norm)*exp(k_norm*k_norm) - pi_half_inv * k_norm) );
    g_half = sqrt( 0.5 * erfc(k_norm) * exp(k_norm * k_norm) );

    Wx.x = prefactor * k3half_inv * (g_half *   ky  * dRand[           index] + f_half * kx * dRand[nModes   + index]);
    Wy.x = prefactor * k3half_inv * (g_half * (-kx) * dRand[           index] + f_half * ky * dRand[nModes   + index]);
    Wx.y = prefactor * k3half_inv * (g_half *   ky  * dRand[nModes*2 + index] + f_half * kx * dRand[nModes*3 + index]);
    Wy.y = prefactor * k3half_inv * (g_half * (-kx) * dRand[nModes*2 + index] + f_half * ky * dRand[nModes*3 + index]); 
  }

  if((fx < 0) or (fx == 0 and fy < 0)){
    Wx.y *= -1.0;
    Wy.y *= -1.0;
  }

  vxZ[i].x += Wx.x;
  vxZ[i].y += Wx.y;
  vyZ[i].x += Wy.x;
  vyZ[i].y += Wy.y;
}


__global__ void updateParticlesQuasi2D(particlesincell* pc, 
				       int* errorKernel,
				       const double* rxcellGPU,
				       const double* rycellGPU,
				       double* rxboundaryGPU,  // q^{} to interpolate
				       double* ryboundaryGPU, 
				       double* rxboundaryPredictionGPU,  // q^{udpdated}
				       double* ryboundaryPredictionGPU, 
				       const double* vxGPU,
				       const double* vyGPU,
				       double dt){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  // double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  // double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rx = rxboundaryGPU[i];
  double ry = ryboundaryGPU[i];

  // printf("rx = %f, ry = %f \n", rx, ry);
					 
  double r;
  int icel;

  // Particle location in grid cells
  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    icel  = jx + jy * mxGPU;
  }

  // Interpolate fluid velocity
  double ux = 0.0, uy = 0.0;

  // Loop over neighbor cells
  {
    double rx_distance, ry_distance, norm, r2;
    int kx, ky, kx_neigh, ky_neigh, icel_neigh;
    ky = icel / mxGPU;
    kx = icel % mxGPU;
    // double icel_double = double(icel);
    for(int ix=-kernelWidthGPU; ix<=kernelWidthGPU; ix++){
      kx_neigh = (kx + ix + mxGPU) % mxGPU;
      rx_distance = rx - (kx_neigh * lxGPU / mxGPU) + lxGPU * 0.5;
      // rx_distance = rx - (kx_neigh * lxGPU / mxGPU) + lxGPU * 0.5 + 0.5 * dxGPU;
      rx_distance = rx_distance - int(rx_distance*invlxGPU + 0.5*((rx_distance>0)-(rx_distance<0)))*lxGPU;

      for(int iy=-kernelWidthGPU; iy<=kernelWidthGPU; iy++){
	ky_neigh = (ky + iy + myGPU) % myGPU;
	icel_neigh = kx_neigh + ky_neigh * mxGPU;

	ry_distance = ry - (ky_neigh * lyGPU / myGPU) + lyGPU * 0.5;
	// ry_distance = ry - (ky_neigh * lyGPU / myGPU) + lyGPU * 0.5 + 0.5 * dyGPU;
	ry_distance = ry_distance - int(ry_distance*invlyGPU + 0.5*((ry_distance>0)-(ry_distance<0)))*lyGPU;
	r2 = rx_distance*rx_distance + ry_distance*ry_distance;
	norm = GaussianKernel2DGPU(r2, GaussianVarianceGPU);

	/*double dlx, dly;
	{ // For the 3-point kernel
	  dlx = delta(rx_distance);
	  dly = delta(ry_distance);
	  norm = dlx * dly;
	  }*/

	ux += norm * vxGPU[icel_neigh];
	uy += norm * vyGPU[icel_neigh];
      }
    }
  }

  double volumeCell = dxGPU * dyGPU;
  // printf("i = %i, ux = %15.12e, uy = %15.12e, dt = %e \n", i, volumeCell * ux, volumeCell * uy, dt);

  rxboundaryPredictionGPU[i] = fetch_double(texrxboundaryGPU,nboundaryGPU+i) + volumeCell * ux * dt;
  ryboundaryPredictionGPU[i] = fetch_double(texryboundaryGPU,nboundaryGPU+i) + volumeCell * uy * dt;
}



__global__ void kernelUpdateVIncompressibleSpectral2D(cufftDoubleComplex *vxZ, 
						      cufftDoubleComplex *vyZ,
						      cufftDoubleComplex *vzZ, 
						      cufftDoubleComplex *WxZ, 
						      cufftDoubleComplex *WyZ, 
						      cufftDoubleComplex *WzZ, 
						      prefactorsFourier *pF){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  // Find mode
  int wx, wy;
  wy = i / mxGPU;
  wx = i % mxGPU;

  if(wx > mxGPU / 2){
    wx -= mxGPU;
  }
  if(wy > myGPU / 2){
    wy -= myGPU;
  }

  bool xHalf = (mxGPU % 2) == 0 && (wx == mxGPU / 2);
  bool yHalf = (myGPU % 2) == 0 && (wy == myGPU / 2);
  if(xHalf and wy < 0){
    wx *= -1;
  }
  else if(yHalf and wx < 0){
    wy *= -1;
  }

  double pi = 3.1415926535897932385;
  double kx = wx * 2.0 * pi / lxGPU;
  double ky = wy * 2.0 * pi / lyGPU;
  
  // Construct L
  double L;
  L = -(kx * kx) - (ky * ky);
  
  // Construct denominator
  double denominator = -shearviscosityGPU * L;
  
  // Construct GW
  cufftDoubleComplex GW;
  GW.x = kx * WxZ[i].x + ky * WyZ[i].x ;
  GW.y = kx * WxZ[i].y + ky * WyZ[i].y ;
  
  if(i == 0){
    vxZ[i].x = 0; 
    vxZ[i].y = 0; 
    vyZ[i].x = 0; 
    vyZ[i].y = 0; 
  }
  else{
    vxZ[i].x = (WxZ[i].x + kx * GW.x / L) / denominator;
    vxZ[i].y = (WxZ[i].y + kx * GW.y / L) / denominator;
    vyZ[i].x = (WyZ[i].x + ky * GW.x / L) / denominator;
    vyZ[i].y = (WyZ[i].y + ky * GW.y / L) / denominator;
  }
}

__global__ void addStochasticVelocitySpectral2D(cufftDoubleComplex *vxZ, cufftDoubleComplex *vyZ, const double *dRand){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int wx, wy, fx, fy, shift;
  int nModes = (ncellsGPU + mxGPU + myGPU);
  wy = i / mxGPU;
  wx = i % mxGPU;
  fx = wx;
  fy = wy;
  shift = 0;

  if(wx > mxGPU / 2){
    fx = wx - mxGPU;
    wx = mxGPU - wx;
  }
  if(wy > myGPU / 2){
    fy = wy - myGPU;
    wy = myGPU - wy;
  }
  bool xHalf = (mxGPU % 2) == 0 && (wx == mxGPU / 2);
  bool yHalf = (myGPU % 2) == 0 && (wy == myGPU / 2);
  if(xHalf and fy < 0){
    fx *= -1;
  }
  else if(yHalf and fx < 0){
    fy *= -1;
  }

  double pi = 3.1415926535897932385;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  if(fx * fy < 0){
    shift = nModes / 2;
    kx *= -1.0;
  }

  int index = wy * mxGPU + wx + shift;
  double k2_inv = 1.0 / (kx*kx + ky*ky);
  cufftDoubleComplex Wx, Wy;
  double prefactor = fact1GPU;

  if(((mxGPU % 2) == 0 && (wx == mxGPU / 2)) || ((myGPU % 2) == 0 && (wy == myGPU / 2)) || (i == 0)){
    Wx.x = 0;
    Wx.y = 0;
    Wy.x = 0;
    Wy.y = 0;
  }
  else if((xHalf and wy==0) || (yHalf and wx==0) || (xHalf and yHalf)){
    Wx.x = sqrt(2.0) * prefactor * k2_inv * (ky    * dRand[           index]);
    Wy.x = sqrt(2.0) * prefactor * k2_inv * ((-kx) * dRand[           index]);   
    Wx.y = 0; 
    Wy.y = 0; 
  }
  else{
    Wx.x = prefactor * k2_inv * (ky    * dRand[           index]);
    Wy.x = prefactor * k2_inv * ((-kx) * dRand[           index]);   
    Wx.y = prefactor * k2_inv * (ky    * dRand[nModes*2 + index]);
    Wy.y = prefactor * k2_inv * ((-kx) * dRand[nModes*2 + index]);
  }

  if((fx < 0) or (fx == 0 and fy < 0)){
    Wx.y *= -1.0;
    Wy.y *= -1.0;
  }

  vxZ[i].x += Wx.x;
  vxZ[i].y += Wx.y;
  vyZ[i].x += Wy.x;
  vyZ[i].y += Wy.y;
}




__global__ void kernelUpdateVIncompressibleStokes2D(cufftDoubleComplex *vxZ, 
						    cufftDoubleComplex *vyZ,
						    cufftDoubleComplex *vzZ, 
						    cufftDoubleComplex *WxZ, 
						    cufftDoubleComplex *WyZ, 
						    cufftDoubleComplex *WzZ, 
						    prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int wx, wy;
  wy = i / mxGPU;
  wx = i % mxGPU;

  double kx, ky;
  kx = pF->gradKx[wx].y;
  ky = pF->gradKy[wy].y;

  /*if(wx > mxGPU / 2){
    kx = -1.0 * kx;
  }
  if(wy > myGPU / 2){
    ky = -1.0 * ky;
    }*/
  
  //Construct L
  double L;
  L = -kx*kx - ky*ky;

  //Construct denominator
  double denominator = -shearviscosityGPU * L;

  //Construct GW
  cufftDoubleComplex GW;
  GW.x = kx * WxZ[i].x + ky * WyZ[i].x ;
  GW.y = kx * WxZ[i].y + ky * WyZ[i].y ;
  
  if(i==0){
    vxZ[i].x = 0; // WxZ[i].x;
    vxZ[i].y = 0; // WxZ[i].y;
    vyZ[i].x = 0; // WyZ[i].x;
    vyZ[i].y = 0; // WyZ[i].y;
  }
  else{
    vxZ[i].x = (WxZ[i].x + kx * GW.x / L) / denominator;
    vxZ[i].y = (WxZ[i].y + kx * GW.y / L) / denominator;
    vyZ[i].x = (WyZ[i].x + ky * GW.x / L) / denominator;
    vyZ[i].y = (WyZ[i].y + ky * GW.y / L) / denominator;
  }
  
}





//Raul Added. Saffman kernel cuda kernels.

__global__ void kernelUpdateVIncompressibleSaffman2D(cufftDoubleComplex *vxZ, 
						      cufftDoubleComplex *vyZ,
						      cufftDoubleComplex *vzZ, 
						      cufftDoubleComplex *WxZ, 
						      cufftDoubleComplex *WyZ, 
						      cufftDoubleComplex *WzZ, 
						      prefactorsFourier *pF){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  // Find mode
  int wx, wy;
  wy = i / mxGPU;
  wx = i % mxGPU;

  if(wx > mxGPU / 2){
    wx -= mxGPU;
  }
  if(wy > myGPU / 2){
    wy -= myGPU;
  }

  bool xHalf = (mxGPU % 2) == 0 && (wx == mxGPU / 2);
  bool yHalf = (myGPU % 2) == 0 && (wy == myGPU / 2);
  if(xHalf and wy < 0){
    wx *= -1;
  }
  else if(yHalf and wx < 0){
    wy *= -1;
  }

  double pi = 3.1415926535897932385;
  double kx = wx * 2.0 * pi / lxGPU;
  double ky = wy * 2.0 * pi / lyGPU;
  
  // Construct L
  double L = -(kx * kx) - (ky * ky);

  

    // Raul added. The kernel is now modified to 1/(k*(k+kc)) if a saffman cut off wavenumber is provided. kc=0 falls back to the original code
    // Construct denominator
  double kmod = sqrt(-L);
  double PBCcorr=1.0;
  if(saffmanLayerWidthGPU != 0.0)
    PBCcorr = tanh(kmod*saffmanLayerWidthGPU);
  double denominator = shearviscosityGPU*(kmod*(kmod+saffmanCutOffWaveNumberGPU*PBCcorr));

  
  // Construct GW
  cufftDoubleComplex GW;
  GW.x = kx * WxZ[i].x + ky * WyZ[i].x ;
  GW.y = kx * WxZ[i].y + ky * WyZ[i].y ;
  
  if(i == 0){
    vxZ[i].x = 0; 
    vxZ[i].y = 0; 
    vyZ[i].x = 0; 
    vyZ[i].y = 0; 
  }
  else{
    vxZ[i].x = (WxZ[i].x + kx * GW.x / L) / denominator;
    vxZ[i].y = (WxZ[i].y + kx * GW.y / L) / denominator;
    vyZ[i].x = (WyZ[i].x + ky * GW.x / L) / denominator;
    vyZ[i].y = (WyZ[i].y + ky * GW.y / L) / denominator;
  }
}



__global__ void addStochasticVelocitySaffman2D(cufftDoubleComplex *vxZ, cufftDoubleComplex *vyZ, const double *dRand){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int wx, wy, fx, fy, shift;
  int nModes = (ncellsGPU + mxGPU + myGPU);
  wy = i / mxGPU;
  wx = i % mxGPU;
  fx = wx;
  fy = wy;
  shift = 0;

  if(wx > mxGPU / 2){
    fx = wx - mxGPU;
    wx = mxGPU - wx;
  }
  if(wy > myGPU / 2){
    fy = wy - myGPU;
    wy = myGPU - wy;
  }
  bool xHalf = (mxGPU % 2) == 0 && (wx == mxGPU / 2);
  bool yHalf = (myGPU % 2) == 0 && (wy == myGPU / 2);
  if(xHalf and fy < 0){
    fx *= -1;
  }
  else if(yHalf and fx < 0){
    fy *= -1;
  }

  double pi = 3.1415926535897932385;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  if(fx * fy < 0){
    shift = nModes / 2;
    kx *= -1.0;
  }

  int index = wy * mxGPU + wx + shift;


  //Raul Added, Saffman correction to stochastic velocity
  double kmod2 = (kx*kx+ky*ky);
  double kmod  = sqrt(kmod2);
  double PBCcorr=1.0;
  if(saffmanLayerWidthGPU != 0.0)
    PBCcorr = tanh(kmod*saffmanLayerWidthGPU);
  double sqrtgk = rsqrt(kmod*kmod2*(kmod+saffmanCutOffWaveNumberGPU*PBCcorr));
    
  cufftDoubleComplex Wx, Wy;
  double prefactor = fact1GPU;

  if(((mxGPU % 2) == 0 && (wx == mxGPU / 2)) || ((myGPU % 2) == 0 && (wy == myGPU / 2)) || (i == 0)){
    Wx.x = 0;
    Wx.y = 0;
    Wy.x = 0;
    Wy.y = 0;
  }
  else if((xHalf and wy==0) || (yHalf and wx==0) || (xHalf and yHalf)){
    Wx.x = sqrt(2.0) * prefactor * sqrtgk * (ky    * dRand[           index]);
    Wy.x = sqrt(2.0) * prefactor * sqrtgk * ((-kx) * dRand[           index]);   
    Wx.y = 0; 
    Wy.y = 0; 
  }
  else{
    Wx.x = prefactor * sqrtgk * (ky    * dRand[           index]);
    Wy.x = prefactor * sqrtgk * ((-kx) * dRand[           index]);   
    Wx.y = prefactor * sqrtgk * (ky    * dRand[nModes*2 + index]);
    Wy.y = prefactor * sqrtgk * ((-kx) * dRand[nModes*2 + index]);
  }

  if((fx < 0) or (fx == 0 and fy < 0)){
    Wx.y *= -1.0;
    Wy.y *= -1.0;
  }

  vxZ[i].x += Wx.x;
  vxZ[i].y += Wx.y;
  vyZ[i].x += Wy.x;
  vyZ[i].y += Wy.y;
}

