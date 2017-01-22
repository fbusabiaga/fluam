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

  // Particle location in grid cells
  /*double r;
  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    icel  = jx + jy * mxGPU;
    // printf("rx = %f,  ry = %f \n", rx, ry);
    // printf("jx = %i,  jy = %i,  icel = %i \n", jx, jy, icel);
  }
  np = atomicAdd(&pc->countparticlesincellX[icel],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[1]=maxNumberPartInCellGPU;
    return;
  }
  pc->partincellX[ncellstGPU*np+icel] = nboundaryGPU+i; */
  
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
  
  double fx = 1.0;
  double fy = 1.0;
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

  // Loop over neighbor cells
  {
    double rx_distance, ry_distance, norm;
    int kx, ky, kx_neigh, ky_neigh, icel_neigh;
    ky = icel / mxGPU;
    kx = icel % mxGPU;
    // double icel_double = double(icel);
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
	// printf("kx = %i,  ky = %i \n", kx_neigh, ky_neigh);
	// printf("rx = %f,  ry = %f,  r2 = %f \n", rx_distance, ry_distance, r2);
	// printf("norm = %f, old = %f \n", norm, old);
      }
    }
  } 
  // printf("fx = %f,  fy = %f \n", fx, fy);
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
  cufftDoubleComplex Wx, Wy;

  if(i == 0){
    // vxZ[i].x = 0;
    // vxZ[i].y = 0;
    // vyZ[i].x = 0;
    // vyZ[i].y = 0;
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

  // printf("i = %i, vx.x = %f, vx.y = %f \n", i, vxZ[i].x, vxZ[i].y);
  // printf("i = %i, vy.x = %f, vy.y = %f \n", i, vyZ[i].x, vyZ[i].y);
}



__global__ void updateParticlesQuasi2D(particlesincell* pc, 
				       int* errorKernel,
				       const double* rxcellGPU,
				       const double* rycellGPU,
				       double* rxboundaryGPU,  // q^{n} -> q^{n+dt}
				       double* ryboundaryGPU, 
				       const double* vxGPU,
				       const double* vyGPU,
				       double dt){

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
      rx_distance = rx_distance - int(rx_distance*invlxGPU + 0.5*((rx_distance>0)-(rx_distance<0)))*lxGPU;

      for(int iy=-kernelWidthGPU; iy<=kernelWidthGPU; iy++){
	ky_neigh = (ky + iy + myGPU) % myGPU;
	icel_neigh = kx_neigh + ky_neigh * mxGPU;

	ry_distance = ry - (ky_neigh * lyGPU / myGPU) + lyGPU * 0.5;
	ry_distance = ry_distance - int(ry_distance*invlyGPU + 0.5*((ry_distance>0)-(ry_distance<0)))*lyGPU;
	r2 = rx_distance*rx_distance + ry_distance*ry_distance;
	norm = GaussianKernel2DGPU(r2, GaussianVarianceGPU);

	ux += norm * vxGPU[icel_neigh];
	uy += norm * vyGPU[icel_neigh];
      }
    }
  }

  double volumeCell = dxGPU * dyGPU;
  printf("i = %i, ux = %f, uy = %f, dt = %f \n", i, volumeCell * ux, volumeCell * uy, dt);
  rxboundaryGPU[i] += volumeCell * ux * dt;
  ryboundaryGPU[i] += volumeCell * uy * dt;

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

  //Find mode
  int wx, wy;
  wy = i / mxGPU;
  wx = i % mxGPU;

  if(wx > mxGPU / 2){
    wx = wx - mxGPU;
    // wx = mxGPU - wx;
    // wx = mxGPU / 2 - wx;
  }
  if(wy > myGPU / 2){
    wy = wy - myGPU;
    // wy = myGPU - wy;
    // wy = myGPU / 2 - wy;
  }
  double pi = 3.1415926535897932385;
  double kx = wx * 2 * pi / lxGPU;
  double ky = wy * 2 * pi / lyGPU;
  
  //Construct L
  double L;
  L = -(kx * kx) - (ky * ky)  ;
  
  //Construct GG
  double GG;
  GG = L;
  
  //Construct denominator
  double denominator = -shearviscosityGPU * L;
  
  //Construct GW
  cufftDoubleComplex GW;
  GW.x = kx * WxZ[i].x + ky * WyZ[i].x ;
  GW.y = kx * WxZ[i].y + ky * WyZ[i].y ;
  
  if(i==0){
    vxZ[i].x = WxZ[i].x;
    vxZ[i].y = WxZ[i].y;
    vyZ[i].x = WyZ[i].x;
    vyZ[i].y = WyZ[i].y;
  }
  else{
    vxZ[i].x = (WxZ[i].x + kx * GW.x / GG) / denominator;
    vxZ[i].y = (WxZ[i].y + kx * GW.y / GG) / denominator;
    vyZ[i].x = (WyZ[i].x + ky * GW.x / GG) / denominator;
    vyZ[i].y = (WyZ[i].y + ky * GW.y / GG) / denominator;
  }
}
