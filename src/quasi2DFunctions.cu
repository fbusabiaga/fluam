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
    icel  = jx;
    icel += jy * mxNeighborsGPU;
  }
  np = atomicAdd(&pc->countPartInCellNonBonded[icel],1);
  if(np >= maxNumberPartInCellNonBondedGPU){
    errorKernel[0] = 1;
    errorKernel[4] = 1;
    return;
  }
  pc->partInCellNonBonded[mNeighborsGPU*np+icel] = i;

  // printf("np = %i,  icel = %i \n", np, icel);
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
    icel  = jx;
    icel += jy * mxNeighborsGPU;
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
	  norm = dlx * dly / (volumeGPU);
	  }*/

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
						double *dRand){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
    
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
  {
    double rx_distance_p, ry_distance_p, rx_distance_m, ry_distance_m;
    double norm;
    int kx, ky, kx_neigh, ky_neigh, icel_neigh;
    ky = icel / mxGPU;
    kx = icel % mxGPU;
    for(int ix=-kernelWidthGPU; ix<=kernelWidthGPU; ix++){
      kx_neigh = (kx + ix + mxGPU) % mxGPU;

      rx_distance_p = (rx + 0.5*deltaRFDGPU*dRand[3*ncellsGPU+i]) - (kx_neigh * lxGPU / mxGPU) + lxGPU * 0.5;
      rx_distance_p = rx_distance_p - int(rx_distance_p*invlxGPU + 0.5*((rx_distance_p>0)-(rx_distance_p<0)))*lxGPU;
      rx_distance_m = (rx - 0.5*deltaRFDGPU*dRand[3*ncellsGPU+i]) - (kx_neigh * lxGPU / mxGPU) + lxGPU * 0.5;
      rx_distance_m = rx_distance_m - int(rx_distance_m*invlxGPU + 0.5*((rx_distance_m>0)-(rx_distance_m<0)))*lxGPU;

      for(int iy=-kernelWidthGPU; iy<=kernelWidthGPU; iy++){
	ky_neigh = (ky + iy + myGPU) % myGPU;
	icel_neigh = kx_neigh + ky_neigh * mxGPU;

	ry_distance_p = (ry + 0.5*deltaRFDGPU*dRand[3*ncellsGPU+npGPU+i]) - (ky_neigh * lyGPU / myGPU) + lyGPU * 0.5;
	ry_distance_p = ry_distance_p - int(ry_distance_p*invlyGPU + 0.5*((ry_distance_p>0)-(ry_distance_p<0)))*lyGPU;
	ry_distance_m = (ry - 0.5*deltaRFDGPU*dRand[3*ncellsGPU+npGPU+i]) - (ky_neigh * lyGPU / myGPU) + lyGPU * 0.5;
	ry_distance_m = ry_distance_m - int(ry_distance_m*invlyGPU + 0.5*((ry_distance_m>0)-(ry_distance_m<0)))*lyGPU;

	// Spread drift kT*S(q+0.5*delta*W)*W
	r2 = rx_distance_p*rx_distance_p + ry_distance_p*ry_distance_p;
	norm = GaussianKernel2DGPU(r2, GaussianVarianceGPU) * temperatureGPU / deltaRFDGPU;

	atomicAdd(&vxZ[icel_neigh].x, norm * dRand[3*ncellsGPU + i]);
	atomicAdd(&vyZ[icel_neigh].x, norm * dRand[3*ncellsGPU + npGPU + i]);

	// Spread drift -kT*S(q-0.5*delta*W)*W
	r2 = rx_distance_m*rx_distance_m + ry_distance_m*ry_distance_m;
	norm = GaussianKernel2DGPU(r2, GaussianVarianceGPU) * temperatureGPU / deltaRFDGPU;

	atomicAdd(&vxZ[icel_neigh].x, -norm * dRand[3*ncellsGPU + i]);
	atomicAdd(&vyZ[icel_neigh].x, -norm * dRand[3*ncellsGPU + npGPU + i]);
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

  double pi = 3.1415926535897932385;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  double k_inv = rsqrt(kx*kx + ky*ky);
  double k3_inv = k_inv * k_inv * k_inv;
  cufftDoubleComplex Wx, Wy;

  if(i == 0 || ((mxGPU % 2) == 0 && (wx == mxGPU / 2)) || ((myGPU % 2) == 0 && (wy == myGPU / 2))){
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


__global__ void addStochasticVelocityQuasi2D(cufftDoubleComplex *vxZ, cufftDoubleComplex *vyZ, const double *dRand){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //Find mode
  int wx, wy, fx, fy, shift;
  int nModes = (ncellsGPU + mxGPU) / 2;
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
  if(fx * fy < 0){
    shift += nModes / 2;
  }

  int index = wy * mxGPU + wx + shift;
  double pi = 3.1415926535897932385;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  double k = sqrt(kx*kx + ky*ky);
  double k3half_inv = rsqrt(k * k * k);
  double sqrtTwo_inv = rsqrt(2.0);
  cufftDoubleComplex Wx, Wy;
  // double prefactor = sqrt(2.0 * temperatureGPU  / (shearviscosityGPU * dtGPU * lxGPU * lyGPU)) * ncellsGPU;
  double prefactor = fact1GPU;

  // Wx.x = 0.0;
  // Wx.y = 0.0;
  // Wy.x = 0.0;
  // Wy.y = 0.0;

  // Wx.x = prefactor * k3half_inv * (sqrtTwo_inv *   ky  * dRand[index]            + 0.5 * kx * dRand[nModes   + index]);
  // Wy.x = prefactor * k3half_inv * (sqrtTwo_inv * (-kx) * dRand[nModes*2 + index] + 0.5 * ky * dRand[nModes*3 + index]);
  // Wx.y = prefactor * k3half_inv * (sqrtTwo_inv *   ky  * dRand[nModes*4 + index] + 0.5 * kx * dRand[nModes*5 + index]);
  // Wy.y = prefactor * k3half_inv * (sqrtTwo_inv * (-kx) * dRand[nModes*6 + index] + 0.5 * ky * dRand[nModes*7 + index]); 

  Wx.x = prefactor * k3half_inv * (1.0 * sqrtTwo_inv *   ky  * dRand[           index] + 0.5 * kx * dRand[nModes   + index]);
  Wy.x = prefactor * k3half_inv * (1.0 * sqrtTwo_inv * (-kx) * dRand[nModes*2 + index] + 0.5 * ky * dRand[nModes*3 + index]);
  Wx.y = prefactor * k3half_inv * (1.0 * sqrtTwo_inv *   ky  * dRand[nModes*4 + index] + 0.5 * kx * dRand[nModes*5 + index]);
  Wy.y = prefactor * k3half_inv * (1.0 * sqrtTwo_inv * (-kx) * dRand[nModes*6 + index] + 0.5 * ky * dRand[nModes*7 + index]); 

  if(fx < 0){
    Wx.y *= -1.0;
    Wy.y *= -1.0;
  }

  /*if(wx == 0 and wy == 2){
    printf("Initial values wx = %i, wy = %i, Used values wx = %i, wy = %i, shift = %i, index = %i, fx = %i, Wx.y = %f \n", i % mxGPU, i / mxGPU, wx, wy, shift, index, fx, Wx.y);
    }*/
 
  if(((mxGPU % 2) == 0 && (wx == mxGPU / 2)) || ((myGPU % 2) == 0 && (wy == myGPU / 2)) || (i == 0)){
    Wx.x = 0;
    Wx.y = 0;
    Wy.x = 0;
    Wy.y = 0;
  }
  else if(wx == 0 or wy == 0){
    Wx.y = 0;
    Wy.y = 0;
  }
  else{
    vxZ[i].x += Wx.x;
    vxZ[i].y += Wx.y;
    vyZ[i].x += Wy.x;
    vyZ[i].y += Wy.y;
  }
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
  // printf("i = %i, ux = %e, uy = %e, dt = %e \n", i, volumeCell * ux, volumeCell * uy, dt);

  // rxboundaryGPU[i] += volumeCell * ux * dt;
  // ryboundaryGPU[i] += volumeCell * uy * dt;
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
  
  if((i == 0) || ((mxGPU % 2) == 0 && (wx == mxGPU / 2)) || ((myGPU % 2) == 0 && (wy == myGPU / 2))){
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
