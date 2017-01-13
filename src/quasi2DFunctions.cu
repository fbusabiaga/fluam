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
  int np = atomicAdd(&pc->countparticlesincellX[icel],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[1]=maxNumberPartInCellGPU;
    return;
  }
  pc->partincellX[ncellstGPU*np+icel] = nboundaryGPU+i; 
 

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
  
}



__global__ void kernelSpreadParticlesForceQuasi2D(const double* rxcellGPU, 
						  const double* rycellGPU, 
						  const double* rzcellGPU,
						  double* fxboundaryGPU, 
						  double* fyboundaryGPU, 
						  double* fzboundaryGPU,
						  cufftDoubleComplex* vxZ,
						  cufftDoubleComplex* vyZ,
						  const particlesincell* pc, 
						  int* errorKernel,
						  const bondedForcesVariables* bFV){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double fx = 0.;
  double fy = 0.;
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
  double r, rp, rm;

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

  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;

  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    icel  = jx + jy * mxGPU;
  }

  // Loop over neighbor cells
  {
    double rx_distance, ry_distance, norm, old;
    int kx, ky, k, kx_neigh, ky_neigh;
    ky = i / mxGPU;
    kx = i % mxGPU;
    // double icel_double = double(icel);
    for(int ix=-kernelWidthGPU; ix<kernelWidthGPU; ix++){
      kx_neigh = (kx + ix + mxGPU) % mxGPU;
      rx_distance = rx - (kx_neigh * lxGPU / mxGPU);
      for(int iy=-kernelWidthGPU; iy<kernelWidthGPU; iy++){
	ky_neigh = (ky + iy + myGPU) % myGPU;
	ry_distance = ry - (ky_neigh * lyGPU / myGPU);
	r2 = rx_distance*rx_distance + ry_distance*ry_distance;
	norm = GaussianKernel2DGPU(r2, GaussianVarianceGPU);
	old = atomicAdd(&vxZ[i].x, norm * fx);
	old = atomicAdd(&vyZ[i].x, norm * fy);
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
    wx = wx - mxGPU;
  }
  if(wy > myGPU / 2){
    wy = wy - myGPU;
  }
  double pi = 3.1415926535897932385;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  double k_inv = rsqrt(kx*kx + ky*ky);
  double k3_inv = k_inv * k_inv * k_inv;

  vxZ[i].x = (k3_inv / shearviscosityGPU) * 0.5 * ky * (ky*vxZ[i].x - kx*vyZ[i].x) + 0.25 * kx * (kx*vxZ[i].x + ky*vyZ[i].x);
  vxZ[i].y = (k3_inv / shearviscosityGPU) * 0.5 * ky * (ky*vxZ[i].y - kx*vyZ[i].y) + 0.25 * kx * (kx*vxZ[i].y + ky*vyZ[i].y);
  
  vyZ[i].x = (k3_inv / shearviscosityGPU) * 0.5 * (-kx) * (ky*vxZ[i].x - kx*vyZ[i].x) + 0.25 * ky * (kx*vxZ[i].x + ky*vyZ[i].x);
  vyZ[i].x = (k3_inv / shearviscosityGPU) * 0.5 * (-kx) * (ky*vxZ[i].y - kx*vyZ[i].y) + 0.25 * ky * (kx*vxZ[i].y + ky*vyZ[i].y);
  
}
