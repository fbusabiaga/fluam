
//Fill "countparticlesincellX" lists
//and spread particle force F 
__global__ void kernelSpreadParticlesForceStokesLimitBigSystem(const double* rxcellGPU, 
							       const double* rycellGPU, 
							       const double* rzcellGPU,
							       double* fxboundaryGPU,
							       double* fyboundaryGPU,
							       double* fzboundaryGPU,
							       double* vxboundaryGPU,//Fx, this is the force on the particle.
							       double* vyboundaryGPU,
							       double* vzboundaryGPU,
							       particlesincell* pc,
							       int* errorKernel,
							       const bondedForcesVariables* bFV,
							       const vecinos* pNeighbors){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double fx = 0.;
  double fy = 0.;
  double fz = 0.;
  double f;
 
  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  
  //INCLUDE EXTERNAL FORCES HERE

  //Example: harmonic potential 
  // V(r) = (1/2) * k * ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
  //
  //with spring constant k=0.01
  //and x0=y0=z0=0
  //
  //fx = -0.01*rx;
  //fy = -0.01*ry;
  //fz = -0.01*rz;


  
  if(particlesWallGPU){
    //INCLUDE WALL REPULSION HERE
    //We use a repulsive Lennard-Jones
    double sigmaWall = 2*dyGPU;
    double cutoffWall = 1.12246204830937302 * sigmaWall; // 2^{1/6} * sigmaWall

    //IMPORTANT, for particleWall lyGPU stores ly+2*dy
    if(ry<(-0.5*lyGPU+cutoffWall+dyGPU)){//Left wall
      
      double distance = (0.5*lyGPU-dyGPU) + ry; //distance >= 0
      fy += 48 * temperatureGPU * (pow((sigmaWall/distance),13) - 0.5*pow((sigmaWall/distance),7));
      
    }
    else if(ry>(0.5*lyGPU-cutoffWall-dyGPU)){//Right wall
      
      double distance = (0.5*lyGPU-dyGPU) - ry; //distance >= 0
      fy -= 48 * temperatureGPU * (pow((sigmaWall/distance),13) - 0.5*pow((sigmaWall/distance),7));
      
    }
  }





  //NEW bonded forces
  if(bondedForcesGPU){
    //call function for bonded forces particle-particle
    forceBondedParticleParticleGPU(i,
				   fx,
				   fy,
				   fz,
				   rx,
				   ry,
				   rz,
				   bFV);
  }
    
  double rxij, ryij, rzij, r2;

  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;
  
  int icel;
  double r, rp, rm;

  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    double invdz = double(mzNeighborsGPU)/lzGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdz + 0.5*mzNeighborsGPU) % mzNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
    icel += jz * mxNeighborsGPU * myNeighborsGPU;    
  }
  
  int np;
  if(computeNonBondedForcesGPU){
    //Particles in Cell i
    np = pc->countPartInCellNonBonded[icel];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+icel];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }  
    //Particles in Cell vecino0
    vecino0 = pNeighbors->vecino0GPU[icel];
    np = pc->countPartInCellNonBonded[vecino0];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecino0];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecino1
    vecino1 = pNeighbors->vecino1GPU[icel];
    np = pc->countPartInCellNonBonded[vecino1];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecino1]; 
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecino2
    vecino2 = pNeighbors->vecino2GPU[icel];
    np = pc->countPartInCellNonBonded[vecino2];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecino2]; 
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecino3
    vecino3 = pNeighbors->vecino3GPU[icel];
    np = pc->countPartInCellNonBonded[vecino3];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecino3];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecino4
    vecino4 = pNeighbors->vecino4GPU[icel];
    np = pc->countPartInCellNonBonded[vecino4];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecino4];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecino5
    vecino5 = pNeighbors->vecino5GPU[icel];
    np = pc->countPartInCellNonBonded[vecino5];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecino5];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopxpy
    vecinopxpy = pNeighbors->vecinopxpyGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopxpy];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopxpy];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopxmy
    vecinopxmy = pNeighbors->vecinopxmyGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopxmy];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopxmy];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopxpz
    vecinopxpz = pNeighbors->vecinopxpzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopxpz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopxpz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopxmz
    vecinopxmz = pNeighbors->vecinopxmzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopxmz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopxmz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinomxpy
    vecinomxpy = pNeighbors->vecinomxpyGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomxpy];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomxpy];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij; 
    }
    //Particles in Cell vecinomxmy
    vecinomxmy = pNeighbors->vecinomxmyGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomxmy];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomxmy];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinomxpz
    vecinomxpz = pNeighbors->vecinomxpzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomxpz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomxpz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinomxmz
    vecinomxmz = pNeighbors->vecinomxmzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomxmz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomxmz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopypz
    vecinopypz = pNeighbors->vecinopypzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopypz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopypz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopymz
    vecinopymz = pNeighbors->vecinopymzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopymz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopymz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinomypz
    vecinomypz = pNeighbors->vecinomypzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomypz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomypz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinomymz
    vecinomymz = pNeighbors->vecinomymzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomymz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomymz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopxpypz
    vecinopxpypz = pNeighbors->vecinopxpypzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopxpypz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopxpypz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopxpymz
    vecinopxpymz = pNeighbors->vecinopxpymzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopxpymz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopxpymz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopxmypz
    vecinopxmypz = pNeighbors->vecinopxmypzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopxmypz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopxmypz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinopxmymz
    vecinopxmymz = pNeighbors->vecinopxmymzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinopxmymz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinopxmymz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinomxpypz
    vecinomxpypz = pNeighbors->vecinomxpypzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomxpypz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomxpypz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinomxpymz
    vecinomxpymz = pNeighbors->vecinomxpymzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomxpymz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomxpymz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinomxmypz
    vecinomxmypz = pNeighbors->vecinomxmypzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomxmypz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomxmypz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    //Particles in Cell vecinomxmymz
    vecinomxmymz = pNeighbors->vecinomxmymzGPU[icel];
    np = pc->countPartInCellNonBonded[vecinomxmymz];
    for(int j=0;j<np;j++){
      int particle = pc->partInCellNonBonded[mNeighborsGPU*j+vecinomxmymz];
      rxij =  (rx - fetch_double(texrxboundaryGPU,particle));
      rxij =  (rxij - int(rxij*invlxGPU + 0.5*((rxij>0)-(rxij<0)))*lxGPU);
      ryij =  (ry - fetch_double(texryboundaryGPU,particle));
      ryij =  (ryij - int(ryij*invlyGPU + 0.5*((ryij>0)-(ryij<0)))*lyGPU);
      rzij =  (rz - fetch_double(texrzboundaryGPU,particle));
      rzij =  (rzij - int(rzij*invlzGPU + 0.5*((rzij>0)-(rzij<0)))*lzGPU);
      r2 = rxij*rxij + ryij*ryij + rzij*rzij;
      f = tex1D(texforceNonBonded1,r2*invcutoff2GPU);
      fx += f * rxij;
      fy += f * ryij;
      fz += f * rzij;
    }
    
  }

  vxboundaryGPU[nboundaryGPU+i] = fx; //Fx, this is the force on the particle.
  vyboundaryGPU[nboundaryGPU+i] = fy;
  vzboundaryGPU[nboundaryGPU+i] = fz;
  
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * mytGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rx - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;
    r = ry - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;
    r = rz - 0.5*dzGPU;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jzdz = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;

    icelx  = jxdx;
    icelx += jy * mxGPU;
    icelx += jz * mxmy;

    icely  = jx;
    icely += jydy * mxGPU;
    icely += jz * mxmy;

    icelz  = jx;
    icelz += jy * mxGPU;
    icelz += jzdz * mxmy;
  }

  np = atomicAdd(&pc->countparticlesincellX[icelx],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[1]=maxNumberPartInCellGPU;
    return;
  }
  pc->partincellX[ncellstGPU*np+icelx] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellY[icely],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[2]=np;
    return;
  }
  pc->partincellY[ncellstGPU*np+icely] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellZ[icelz],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[3]=np;
    return;
  }
  pc->partincellZ[ncellstGPU*np+icelz] = nboundaryGPU+i;

  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;

  //FORCE IN THE X DIRECTION 
  r  = (rx - rxcellGPU[icelx] - dxGPU*0.5);
  r  = (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;

  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  //dlxS = functionDeltaDerived(1.5*r);
  //dlxpS = functionDeltaDerived(1.5*rp);
  //dlxmS = functionDeltaDerived(1.5*rm);
                              
  r =  (ry - rycellGPU[icelx]);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);

  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fx;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fx;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fx;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fx;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fx;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fx;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fx;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fx;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fx;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fx;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fx;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fx;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fx;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fx;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fx;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fx;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fx;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fx;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fx;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fx;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fx;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fx;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fx;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fx;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fx;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fx;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fx;
  
  
  //FORCE IN THE Y DIRECTION
  r =  (rx - rxcellGPU[icely]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icely]);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);

  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fy;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fy;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fy;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fy;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fy;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fy;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fy;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fy;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fy;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fy;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fy;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fy;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fy;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fy;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fy;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fy;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fy;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fy;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fy;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fy;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fy;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fy;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fy;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fy;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fy;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fy;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fy;


  
  //FORCE IN THE Z DIRECTION
  r =  (rx - rxcellGPU[icelz]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  //dlxS = functionDeltaDerived(1.5*r);
  //dlxpS = functionDeltaDerived(1.5*rp);
  //dlxmS = functionDeltaDerived(1.5*rm);

  r =  (ry - rycellGPU[icelz]);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);


  offset = nboundaryGPU;
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fz;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fz;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fz;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fz;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fz;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fz;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fz;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fz;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fz;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fz;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fz;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fz;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fz;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fz;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fz;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fz;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fz;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fz;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fz;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fz;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fz;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fz;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fz;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fz;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fz;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fz;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fz;

  
}











__global__ void addSpreadedForcesStokesLimitBigSystem(double* vxGPU, //fx=S*Fx
						      double* vyGPU, 
						      double* vzGPU,
						      const double* fxboundaryGPU, 
						      const double* fyboundaryGPU, 
						      const double* fzboundaryGPU,
						      const particlesincell* pc,
						      const vecinos* pVecinos){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellstGPU) return;

  int vecino, particle;
  double fx = 0.f;
  double fy = 0.f;
  double fz = 0.f;

  
  //FORCE IN THE X DIRECTION
  //Particles in Cell i
  int np = pc->countparticlesincellX[i];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+i];
    fx -= fxboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[i];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+i];
    fy -= fyboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[i];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+i];
    fz -= fzboundaryGPU[particle+13*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino0
  vecino = pVecinos->vecino0GPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+12*(nboundaryGPU+npGPU)];
  } 
  //Particles in Cell vecino1
  vecino = pVecinos->vecino1GPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+10*(nboundaryGPU+npGPU)];   
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+10*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino2
  vecino = pVecinos->vecino2GPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+4*(nboundaryGPU+npGPU)];    
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+4*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecino3
  vecino = pVecinos->vecino3GPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+22*(nboundaryGPU+npGPU)];
  }  
  //Particles in Cell vecino4
  vecino = pVecinos->vecino4GPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+16*(nboundaryGPU+npGPU)];   
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+16*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecino5
  vecino = pVecinos->vecino5GPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+14*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpy
  vecino = pVecinos->vecinopxpyGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+25*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxmy
  vecino = pVecinos->vecinopxmyGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+19*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinopxpz
  vecino = pVecinos->vecinopxpzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+23*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinopxmz
  vecino = pVecinos->vecinopxmzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+21*(nboundaryGPU+npGPU)]; 
  }
  //Particles in Cell vecinomxpy
  vecino = pVecinos->vecinomxpyGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+7*(nboundaryGPU+npGPU)];   
  }
  //Particles in Cell vecinomxmy
  vecino = pVecinos->vecinomxmyGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpz
  vecino = pVecinos->vecinomxpzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+5*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmz
  vecino = pVecinos->vecinomxmzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+3*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopypz
  vecino = pVecinos->vecinopypzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+17*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopymz
  vecino = pVecinos->vecinopymzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+15*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomypz
  vecino = pVecinos->vecinomypzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+11*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomymz
  vecino = pVecinos->vecinomymzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+9*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpypz
  vecino = pVecinos->vecinopxpypzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+26*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinopxpymz
  vecino = pVecinos->vecinopxpymzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+24*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmypz
  vecino = pVecinos->vecinopxmypzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+20*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinopxmymz
  vecino = pVecinos->vecinopxmymzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+18*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpypz
  vecino = pVecinos->vecinomxpypzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+8*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxpymz
  vecino = pVecinos->vecinomxpymzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+6*(nboundaryGPU+npGPU)];
  }
  //Particles in Cell vecinomxmypz
  vecino = pVecinos->vecinomxmypzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle+2*(nboundaryGPU+npGPU)];    
  }
  //Particles in Cell vecinomxmymz
  vecino = pVecinos->vecinomxmymzGPU[i];
  np = pc->countparticlesincellX[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellX[ncellstGPU*j+vecino];
    fx -= fxboundaryGPU[particle];
  }
  np = pc->countparticlesincellY[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellY[ncellstGPU*j+vecino];
    fy -= fyboundaryGPU[particle];
  }
  np = pc->countparticlesincellZ[vecino];
  for(int j=0;j<np;j++){
    particle = pc->partincellZ[ncellstGPU*j+vecino];
    fz -= fzboundaryGPU[particle];
  }



  vxGPU[i] += fx / volumeGPU;
  vyGPU[i] += fy / volumeGPU;
  vzGPU[i] += fz / volumeGPU;

}










__global__ void kernelConstructWstokesLimitBigSystem(const double *vxGPU, //Stored fx=S*Fx+S*drift_p-S*drift_m
						     const double *vyGPU, 
						     const double *vzGPU, 
						     cufftDoubleComplex *WxZ, 
						     cufftDoubleComplex *WyZ, 
						     cufftDoubleComplex *WzZ, 
						     const double* d_rand,
						     const vecinos* pVecinos,
						     const double noiseFactor){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double wx, wy, wz;
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 

  vecino0 = pVecinos->vecino0GPU[i];
  vecino1 = pVecinos->vecino1GPU[i];
  vecino2 = pVecinos->vecino2GPU[i];
  vecino3 = pVecinos->vecino3GPU[i];
  vecino4 = pVecinos->vecino4GPU[i];
  vecino5 = pVecinos->vecino5GPU[i];

  //Stored fx=S*Fx+S*drift_p-S*drift_m
  wx = vxGPU[i];
  wy = vyGPU[i];
  wz = vzGPU[i];

  //NOISE part
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double fact1 = noiseFactor * fact1GPU;
  double fact4 = noiseFactor * fact4GPU;

  dnoise_sXX = d_rand[vecino3];
  wx += invdxGPU * fact1 * dnoise_sXX;

  dnoise_sYY = d_rand[vecino4 + 3*ncellsGPU];
  wy += invdyGPU * fact1 * dnoise_sYY;

  dnoise_sZZ = d_rand[vecino5 + 5*ncellsGPU];
  wz += invdzGPU * fact1 * dnoise_sZZ;

  dnoise_sXY = d_rand[i + ncellsGPU];
  wx += invdyGPU * fact4 * dnoise_sXY;
  wy += invdxGPU * fact4 * dnoise_sXY;

  dnoise_sXZ = d_rand[i + 2*ncellsGPU];
  wx += invdzGPU * fact4 * dnoise_sXZ;
  wz += invdxGPU * fact4 * dnoise_sXZ;

  dnoise_sYZ = d_rand[i + 4*ncellsGPU];
  wy += invdzGPU * fact4 * dnoise_sYZ;
  wz += invdyGPU * fact4 * dnoise_sYZ;

  dnoise_sXX = d_rand[i];
  wx -= invdxGPU * fact1 * dnoise_sXX;

  dnoise_sYY = d_rand[i + 3*ncellsGPU];
  wy -= invdyGPU * fact1 * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 5*ncellsGPU];
  wz -= invdzGPU * fact1 * dnoise_sZZ;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  wx -= invdyGPU * fact4 * dnoise_sXY;

  dnoise_sXZ = d_rand[vecino0 + 2*ncellsGPU];
  wx -= invdzGPU * fact4 * dnoise_sXZ;

  dnoise_sXY = d_rand[vecino2 + ncellsGPU];
  wy -= invdxGPU * fact4 * dnoise_sXY;

  dnoise_sYZ = d_rand[vecino0 + 4*ncellsGPU];
  wy -= invdzGPU * fact4 * dnoise_sYZ;

  dnoise_sXZ = d_rand[vecino2 + 2*ncellsGPU];
  wz -= invdxGPU * fact4 * dnoise_sXZ;

  dnoise_sYZ = d_rand[vecino1 + 4*ncellsGPU];
  wz -= invdyGPU * fact4 * dnoise_sYZ;


  WxZ[i].x = wx;
  WyZ[i].x = wy;
  WzZ[i].x = wz;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

}










__global__ void updateParticlesStokesLimitBigSystem_1(particlesincell* pc, 
						      int* errorKernel,
						      const double* rxcellGPU,
						      const double* rycellGPU,
						      const double* rzcellGPU,
						      const double* rxboundaryGPU,  //q^{n}
						      const double* ryboundaryGPU, 
						      const double* rzboundaryGPU,
						      double* rxboundaryPredictionGPU, //q^{n+1/2}
						      double* ryboundaryPredictionGPU, 
						      double* rzboundaryPredictionGPU,
						      const double* vxboundaryGPU, //Fx^n
						      const double* vyboundaryGPU, 
						      const double* vzboundaryGPU, 
						      const double* vxGPU, //v^n
						      const double* vyGPU, 
						      const double* vzGPU,
						      const double* d_rand,
						      const vecinos* pVecinos){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;

  double r, rp, rm;
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;

  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * mytGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rx - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;
    r = ry - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;
    r = rz - 0.5*dzGPU;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jzdz = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;

    icelx  = jxdx;
    icelx += jy * mxGPU;
    icelx += jz * mxmy;

    icely  = jx;
    icely += jydy * mxGPU;
    icely += jz * mxmy;

    icelz  = jx;
    icelz += jy * mxGPU;
    icelz += jzdz * mxmy;
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino0 = pVecinos->vecino0GPU[icelx];
  vecino1 = pVecinos->vecino1GPU[icelx];
  vecino2 = pVecinos->vecino2GPU[icelx];
  vecino3 = pVecinos->vecino3GPU[icelx];
  vecino4 = pVecinos->vecino4GPU[icelx];
  vecino5 = pVecinos->vecino5GPU[icelx];
  vecinopxpy = pVecinos->vecinopxpyGPU[icelx];
  vecinopxmy = pVecinos->vecinopxmyGPU[icelx];
  vecinopxpz = pVecinos->vecinopxpzGPU[icelx];
  vecinopxmz = pVecinos->vecinopxmzGPU[icelx];
  vecinomxpy = pVecinos->vecinomxpyGPU[icelx];
  vecinomxmy = pVecinos->vecinomxmyGPU[icelx];
  vecinomxpz = pVecinos->vecinomxpzGPU[icelx];
  vecinomxmz = pVecinos->vecinomxmzGPU[icelx];
  vecinopypz = pVecinos->vecinopypzGPU[icelx];
  vecinopymz = pVecinos->vecinopymzGPU[icelx];
  vecinomypz = pVecinos->vecinomypzGPU[icelx];
  vecinomymz = pVecinos->vecinomymzGPU[icelx];
  vecinopxpypz = pVecinos->vecinopxpypzGPU[icelx];
  vecinopxpymz = pVecinos->vecinopxpymzGPU[icelx];
  vecinopxmypz = pVecinos->vecinopxmypzGPU[icelx];
  vecinopxmymz = pVecinos->vecinopxmymzGPU[icelx];
  vecinomxpypz = pVecinos->vecinomxpypzGPU[icelx];
  vecinomxpymz = pVecinos->vecinomxpymzGPU[icelx];
  vecinomxmypz = pVecinos->vecinomxmypzGPU[icelx];
  vecinomxmymz = pVecinos->vecinomxmymzGPU[icelx];
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  
  r =  (ry - rycellGPU[icelx]);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  
  double v = dlxm * dlym * dlzm * vxGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vxGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vxGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vxGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vxGPU[vecino2] +
    dlxm * dly  * dlzp * vxGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vxGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vxGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vxGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vxGPU[vecinomymz] +
    dlx  * dlym * dlz  * vxGPU[vecino1] +
    dlx  * dlym * dlzp * vxGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vxGPU[vecino0] + 
    dlx  * dly  * dlz  * vxGPU[icelx] + 
    dlx  * dly  * dlzp * vxGPU[vecino5] + 
    dlx  * dlyp * dlzm * vxGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vxGPU[vecino4] + 
    dlx  * dlyp * dlzp * vxGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vxGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vxGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vxGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vxGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vxGPU[vecino3] +
    dlxp * dly  * dlzp * vxGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vxGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vxGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vxGPU[vecinopxpypz];

  

  double rxNew = rx + 0.5 * dtGPU * v;
  if(setExtraMobilityGPU){ 
    rxNew += 0.5*dtGPU*extraMobilityGPU * vxboundaryGPU[nboundaryGPU+i]
      + fact2GPU * d_rand[6*ncellsGPU + 3*npGPU + i];
  }
  rxboundaryPredictionGPU[nboundaryGPU+i] = rxNew;


  //VELOCITY IN THE Y DIRECTION
  vecino0 = pVecinos->vecino0GPU[icely];
  vecino1 = pVecinos->vecino1GPU[icely];
  vecino2 = pVecinos->vecino2GPU[icely];
  vecino3 = pVecinos->vecino3GPU[icely];
  vecino4 = pVecinos->vecino4GPU[icely];
  vecino5 = pVecinos->vecino5GPU[icely];
  vecinopxpy = pVecinos->vecinopxpyGPU[icely];
  vecinopxmy = pVecinos->vecinopxmyGPU[icely];
  vecinopxpz = pVecinos->vecinopxpzGPU[icely];
  vecinopxmz = pVecinos->vecinopxmzGPU[icely];
  vecinomxpy = pVecinos->vecinomxpyGPU[icely];
  vecinomxmy = pVecinos->vecinomxmyGPU[icely];
  vecinomxpz = pVecinos->vecinomxpzGPU[icely];
  vecinomxmz = pVecinos->vecinomxmzGPU[icely];
  vecinopypz = pVecinos->vecinopypzGPU[icely];
  vecinopymz = pVecinos->vecinopymzGPU[icely];
  vecinomypz = pVecinos->vecinomypzGPU[icely];
  vecinomymz = pVecinos->vecinomymzGPU[icely];
  vecinopxpypz = pVecinos->vecinopxpypzGPU[icely];
  vecinopxpymz = pVecinos->vecinopxpymzGPU[icely];
  vecinopxmypz = pVecinos->vecinopxmypzGPU[icely];
  vecinopxmymz = pVecinos->vecinopxmymzGPU[icely];
  vecinomxpypz = pVecinos->vecinomxpypzGPU[icely];
  vecinomxpymz = pVecinos->vecinomxpymzGPU[icely];
  vecinomxmypz = pVecinos->vecinomxmypzGPU[icely];
  vecinomxmymz = pVecinos->vecinomxmymzGPU[icely];  

  r =  (rx - rxcellGPU[icely]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icely]);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);


  v = dlxm * dlym * dlzm * vyGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vyGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vyGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vyGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vyGPU[vecino2] +
    dlxm * dly  * dlzp * vyGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vyGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vyGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vyGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vyGPU[vecinomymz] +
    dlx  * dlym * dlz  * vyGPU[vecino1] +
    dlx  * dlym * dlzp * vyGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vyGPU[vecino0] + 
    dlx  * dly  * dlz  * vyGPU[icely] + 
    dlx  * dly  * dlzp * vyGPU[vecino5] + 
    dlx  * dlyp * dlzm * vyGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vyGPU[vecino4] + 
    dlx  * dlyp * dlzp * vyGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vyGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vyGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vyGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vyGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vyGPU[vecino3] +
    dlxp * dly  * dlzp * vyGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vyGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vyGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vyGPU[vecinopxpypz];
  

  double ryNew = ry + 0.5 * dtGPU * v;
  if(setExtraMobilityGPU){
    ryNew += 0.5*dtGPU*extraMobilityGPU * vyboundaryGPU[nboundaryGPU+i]
      + fact2GPU * d_rand[6*ncellsGPU + 4*npGPU + i];
  }
  ryboundaryPredictionGPU[nboundaryGPU+i] = ryNew;
 
  //VELOCITY IN THE Z DIRECTION
  vecino0 = pVecinos->vecino0GPU[icelz];
  vecino1 = pVecinos->vecino1GPU[icelz];
  vecino2 = pVecinos->vecino2GPU[icelz];
  vecino3 = pVecinos->vecino3GPU[icelz];
  vecino4 = pVecinos->vecino4GPU[icelz];
  vecino5 = pVecinos->vecino5GPU[icelz];
  vecinopxpy = pVecinos->vecinopxpyGPU[icelz];
  vecinopxmy = pVecinos->vecinopxmyGPU[icelz];
  vecinopxpz = pVecinos->vecinopxpzGPU[icelz];
  vecinopxmz = pVecinos->vecinopxmzGPU[icelz];
  vecinomxpy = pVecinos->vecinomxpyGPU[icelz];
  vecinomxmy = pVecinos->vecinomxmyGPU[icelz];
  vecinomxpz = pVecinos->vecinomxpzGPU[icelz];
  vecinomxmz = pVecinos->vecinomxmzGPU[icelz];
  vecinopypz = pVecinos->vecinopypzGPU[icelz];
  vecinopymz = pVecinos->vecinopymzGPU[icelz];
  vecinomypz = pVecinos->vecinomypzGPU[icelz];
  vecinomymz = pVecinos->vecinomymzGPU[icelz];
  vecinopxpypz = pVecinos->vecinopxpypzGPU[icelz];
  vecinopxpymz = pVecinos->vecinopxpymzGPU[icelz];
  vecinopxmypz = pVecinos->vecinopxmypzGPU[icelz];
  vecinopxmymz = pVecinos->vecinopxmymzGPU[icelz];
  vecinomxpypz = pVecinos->vecinomxpypzGPU[icelz];
  vecinomxpymz = pVecinos->vecinomxpymzGPU[icelz];
  vecinomxmypz = pVecinos->vecinomxmypzGPU[icelz];
  vecinomxmymz = pVecinos->vecinomxmymzGPU[icelz];  

  r =  (rx - rxcellGPU[icelz]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icelz]);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);

 
  v = dlxm * dlym * dlzm * vzGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vzGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vzGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vzGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vzGPU[vecino2] +
    dlxm * dly  * dlzp * vzGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vzGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vzGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vzGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vzGPU[vecinomymz] +
    dlx  * dlym * dlz  * vzGPU[vecino1] +
    dlx  * dlym * dlzp * vzGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vzGPU[vecino0] + 
    dlx  * dly  * dlz  * vzGPU[icelz] + 
    dlx  * dly  * dlzp * vzGPU[vecino5] + 
    dlx  * dlyp * dlzm * vzGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vzGPU[vecino4] + 
    dlx  * dlyp * dlzp * vzGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vzGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vzGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vzGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vzGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vzGPU[vecino3] +
    dlxp * dly  * dlzp * vzGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vzGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vzGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vzGPU[vecinopxpypz];



  double rzNew = rz + 0.5 * dtGPU * v;
  if(setExtraMobilityGPU){
    rzNew += 0.5*dtGPU*extraMobilityGPU * vzboundaryGPU[nboundaryGPU+i]
      + fact2GPU * d_rand[6*ncellsGPU + 5*npGPU + i];
  }
  rzboundaryPredictionGPU[nboundaryGPU+i] = rzNew;



  int icel;
  {
    double invdx = double(mxNeighborsGPU)/lxGPU;
    double invdy = double(myNeighborsGPU)/lyGPU;
    double invdz = double(mzNeighborsGPU)/lzGPU;
    rxNew = rxNew - (int(rxNew*invlxGPU + 0.5*((rxNew>0)-(rxNew<0)))) * lxGPU;
    int jx   = int(rxNew * invdx + 0.5*mxNeighborsGPU) % mxNeighborsGPU;
    ryNew = ryNew - (int(ryNew*invlyGPU + 0.5*((ryNew>0)-(ryNew<0)))) * lyGPU;
    int jy   = int(ryNew * invdy + 0.5*myNeighborsGPU) % myNeighborsGPU;
    rzNew = rzNew - (int(rzNew*invlzGPU + 0.5*((rzNew>0)-(rzNew<0)))) * lzGPU;
    int jz   = int(rzNew * invdz + 0.5*mzNeighborsGPU) % mzNeighborsGPU;
    icel  = jx;
    icel += jy * mxNeighborsGPU;
    icel += jz * mxNeighborsGPU * myNeighborsGPU;    
  }
  int np = atomicAdd(&pc->countPartInCellNonBonded[icel],1);
  if(np >= maxNumberPartInCellNonBondedGPU){
    errorKernel[0] = 1;
    errorKernel[4] = 1;
    return;
  }
  pc->partInCellNonBonded[mNeighborsGPU*np+icel] = i;



}
















__global__ void kernelSpreadParticlesDriftBigSystem(const double* rxcellGPU, 
						    const double* rycellGPU, 
						    const double* rzcellGPU,
						    double* fxboundaryGPU,
						    double* fyboundaryGPU,
						    double* fzboundaryGPU,
						    const double* d_rand,
						    particlesincell* pc,
						    int* errorKernel,
						    int sign){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   
  
  double deltaDisplacement = 0.001;
  double f = (temperatureGPU / deltaDisplacement) * sign;
  double fx = f * d_rand[6*ncellsGPU +           i];
  double fy = f * d_rand[6*ncellsGPU +   npGPU + i];
  double fz = f * d_rand[6*ncellsGPU + 2*npGPU + i];

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
   
  double r, rp, rm;

  //Displacement to positive values
  double rxR = rx + sign * 0.5 * deltaDisplacement * d_rand[6*ncellsGPU +           i];
  double ryR = ry + sign * 0.5 * deltaDisplacement * d_rand[6*ncellsGPU +   npGPU + i];
  double rzR = rz + sign * 0.5 * deltaDisplacement * d_rand[6*ncellsGPU + 2*npGPU + i];
  
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;
  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * mytGPU;
    r = rxR;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rxR - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ryR;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;
    r = ryR - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    r = rzR;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;
    r = rzR - 0.5*dzGPU;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jzdz = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;

    icelx  = jxdx;
    icelx += jy * mxGPU;
    icelx += jz * mxmy;

    icely  = jx;
    icely += jydy * mxGPU;
    icely += jz * mxmy;

    icelz  = jx;
    icelz += jy * mxGPU;
    icelz += jzdz * mxmy;
  }


  int np = atomicAdd(&pc->countparticlesincellX[icelx],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[1]=maxNumberPartInCellGPU;
    return;
  }
  pc->partincellX[ncellstGPU*np+icelx] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellY[icely],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[2]=np;
    return;
  }
  pc->partincellY[ncellstGPU*np+icely] = nboundaryGPU+i;
  np = atomicAdd(&pc->countparticlesincellZ[icelz],1);
  if(np >= maxNumberPartInCellGPU){
    errorKernel[0]=1;
    errorKernel[3]=np;
    return;
  }
  pc->partincellZ[ncellstGPU*np+icelz] = nboundaryGPU+i;



  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;

  //THERMAL DRIFT IN THE X DIRECTION
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  //dlx  = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
                              
  r =  (ry - rycellGPU[icelx]);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);

    
  int offset = nboundaryGPU;
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fx;
  offset += nboundaryGPU+npGPU;//1
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fx;
  offset += nboundaryGPU+npGPU;//2
  fxboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fx;
  offset += nboundaryGPU+npGPU;//3
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fx;
  offset += nboundaryGPU+npGPU;//4
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fx;
  offset += nboundaryGPU+npGPU;//5
  fxboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fx;
  offset += nboundaryGPU+npGPU;//6
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fx;
  offset += nboundaryGPU+npGPU;//7
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fx;
  offset += nboundaryGPU+npGPU;//8
  fxboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fx;
  offset += nboundaryGPU+npGPU;//9
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fx;
  offset += nboundaryGPU+npGPU;//10
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fx;
  offset += nboundaryGPU+npGPU;//11
  fxboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fx;
  offset += nboundaryGPU+npGPU;//12
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fx;
  offset += nboundaryGPU+npGPU;//13
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fx;
  offset += nboundaryGPU+npGPU;//14
  fxboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fx;
  offset += nboundaryGPU+npGPU;//15
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fx;
  offset += nboundaryGPU+npGPU;//16
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fx;
  offset += nboundaryGPU+npGPU;//17
  fxboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fx;
  offset += nboundaryGPU+npGPU;//18
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fx;
  offset += nboundaryGPU+npGPU;//19
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fx;
  offset += nboundaryGPU+npGPU;//20
  fxboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fx;
  offset += nboundaryGPU+npGPU;//21
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fx;
  offset += nboundaryGPU+npGPU;//22
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fx;
  offset += nboundaryGPU+npGPU;//23
  fxboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fx;
  offset += nboundaryGPU+npGPU;//24
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fx;
  offset += nboundaryGPU+npGPU;//25
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fx;
  offset += nboundaryGPU+npGPU;//26
  fxboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fx;
  
  
  //FORCE IN THE Y DIRECTION
  r =  (rx - rxcellGPU[icely]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icely]);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);

  offset = nboundaryGPU;
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fy;
  offset += nboundaryGPU+npGPU;//1
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fy;
  offset += nboundaryGPU+npGPU;//2
  fyboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fy;
  offset += nboundaryGPU+npGPU;//3
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fy;
  offset += nboundaryGPU+npGPU;//4
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fy;
  offset += nboundaryGPU+npGPU;//5
  fyboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fy;
  offset += nboundaryGPU+npGPU;//6
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fy;
  offset += nboundaryGPU+npGPU;//7
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fy;
  offset += nboundaryGPU+npGPU;//8
  fyboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fy;
  offset += nboundaryGPU+npGPU;//9
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fy;
  offset += nboundaryGPU+npGPU;//10
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fy;
  offset += nboundaryGPU+npGPU;//11
  fyboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fy;
  offset += nboundaryGPU+npGPU;//12
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fy;
  offset += nboundaryGPU+npGPU;//13
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fy;
  offset += nboundaryGPU+npGPU;//14
  fyboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fy;
  offset += nboundaryGPU+npGPU;//15
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fy;
  offset += nboundaryGPU+npGPU;//16
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fy;
  offset += nboundaryGPU+npGPU;//17
  fyboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fy;
  offset += nboundaryGPU+npGPU;//18
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fy;
  offset += nboundaryGPU+npGPU;//19
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fy;
  offset += nboundaryGPU+npGPU;//20
  fyboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fy;
  offset += nboundaryGPU+npGPU;//21
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fy;
  offset += nboundaryGPU+npGPU;//22
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fy;
  offset += nboundaryGPU+npGPU;//23
  fyboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fy;
  offset += nboundaryGPU+npGPU;//24
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fy;
  offset += nboundaryGPU+npGPU;//25
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fy;
  offset += nboundaryGPU+npGPU;//26
  fyboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fy;


  
  //FORCE IN THE Z DIRECTION
  r =  (rx - rxcellGPU[icelz]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  //dlx = tex1D(texDelta, fabs(r));
  //dlxp = tex1D(texDelta, fabs(rp));
  //dlxm = tex1D(texDelta, fabs(rm));
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icelz]);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  //dly = tex1D(texDelta, fabs(r));
  //dlyp = tex1D(texDelta, fabs(rp));
  //dlym = tex1D(texDelta, fabs(rm));
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  //dlz = tex1D(texDelta, fabs(r));
  //dlzp = tex1D(texDelta, fabs(rp));
  //dlzm = tex1D(texDelta, fabs(rm));
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);


  offset = nboundaryGPU;
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlzp * fz;
  offset += nboundaryGPU+npGPU;//1
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlz  * fz;
  offset += nboundaryGPU+npGPU;//2
  fzboundaryGPU[offset+i] = -dlxp * dlyp * dlzm * fz;
  offset += nboundaryGPU+npGPU;//3
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlzp * fz;
  offset += nboundaryGPU+npGPU;//4
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlz  * fz;
  offset += nboundaryGPU+npGPU;//5
  fzboundaryGPU[offset+i] = -dlxp * dly  * dlzm * fz;
  offset += nboundaryGPU+npGPU;//6
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlzp * fz;
  offset += nboundaryGPU+npGPU;//7
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlz  * fz;
  offset += nboundaryGPU+npGPU;//8
  fzboundaryGPU[offset+i] = -dlxp * dlym * dlzm * fz;
  offset += nboundaryGPU+npGPU;//9
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlzp * fz;
  offset += nboundaryGPU+npGPU;//10
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlz  * fz;
  offset += nboundaryGPU+npGPU;//11
  fzboundaryGPU[offset+i] = -dlx  * dlyp * dlzm * fz;
  offset += nboundaryGPU+npGPU;//12
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlzp * fz;
  offset += nboundaryGPU+npGPU;//13
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlz  * fz;
  offset += nboundaryGPU+npGPU;//14
  fzboundaryGPU[offset+i] = -dlx  * dly  * dlzm * fz;
  offset += nboundaryGPU+npGPU;//15
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlzp * fz;
  offset += nboundaryGPU+npGPU;//16
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlz  * fz;
  offset += nboundaryGPU+npGPU;//17
  fzboundaryGPU[offset+i] = -dlx  * dlym * dlzm * fz;
  offset += nboundaryGPU+npGPU;//18
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlzp * fz;
  offset += nboundaryGPU+npGPU;//19
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlz  * fz;
  offset += nboundaryGPU+npGPU;//20
  fzboundaryGPU[offset+i] = -dlxm * dlyp * dlzm * fz;
  offset += nboundaryGPU+npGPU;//21
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlzp * fz;
  offset += nboundaryGPU+npGPU;//22
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlz  * fz;
  offset += nboundaryGPU+npGPU;//23
  fzboundaryGPU[offset+i] = -dlxm * dly  * dlzm * fz;
  offset += nboundaryGPU+npGPU;//24
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlzp * fz;
  offset += nboundaryGPU+npGPU;//25
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlz  * fz;
  offset += nboundaryGPU+npGPU;//26
  fzboundaryGPU[offset+i] = -dlxm * dlym * dlzm * fz;

  
}













__global__ void kernelConstructWstokesLimitBigSystem_2(const double *vxGPU, //Stored fx=S*Fx+S*drift_p-S*drift_m
						       const double *vyGPU, 
						       const double *vzGPU, 
						       cufftDoubleComplex *WxZ, 
						       cufftDoubleComplex *WyZ, 
						       cufftDoubleComplex *WzZ, 
						       const double* d_rand,
						       const vecinos* pVecinos,
						       const double noiseFactor){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double wx, wy, wz;
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 

  vecino0 = pVecinos->vecino0GPU[i];
  vecino1 = pVecinos->vecino1GPU[i];
  vecino2 = pVecinos->vecino2GPU[i];
  vecino3 = pVecinos->vecino3GPU[i];
  vecino4 = pVecinos->vecino4GPU[i];
  vecino5 = pVecinos->vecino5GPU[i];

  //Stored fx=S*Fx+S*drift_p-S*drift_m
  wx = vxGPU[i];
  wy = vyGPU[i];
  wz = vzGPU[i];

  //NOISE part
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double fact1 = noiseFactor * fact1GPU;
  double fact4 = noiseFactor * fact4GPU;

  dnoise_sXX = d_rand[vecino3];
  wx += invdxGPU * fact1 * dnoise_sXX;

  dnoise_sYY = d_rand[vecino4 + 3*ncellsGPU];
  wy += invdyGPU * fact1 * dnoise_sYY;

  dnoise_sZZ = d_rand[vecino5 + 5*ncellsGPU];
  wz += invdzGPU * fact1 * dnoise_sZZ;

  dnoise_sXY = d_rand[i + ncellsGPU];
  wx += invdyGPU * fact4 * dnoise_sXY;
  wy += invdxGPU * fact4 * dnoise_sXY;

  dnoise_sXZ = d_rand[i + 2*ncellsGPU];
  wx += invdzGPU * fact4 * dnoise_sXZ;
  wz += invdxGPU * fact4 * dnoise_sXZ;

  dnoise_sYZ = d_rand[i + 4*ncellsGPU];
  wy += invdzGPU * fact4 * dnoise_sYZ;
  wz += invdyGPU * fact4 * dnoise_sYZ;

  dnoise_sXX = d_rand[i];
  wx -= invdxGPU * fact1 * dnoise_sXX;

  dnoise_sYY = d_rand[i + 3*ncellsGPU];
  wy -= invdyGPU * fact1 * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 5*ncellsGPU];
  wz -= invdzGPU * fact1 * dnoise_sZZ;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  wx -= invdyGPU * fact4 * dnoise_sXY;

  dnoise_sXZ = d_rand[vecino0 + 2*ncellsGPU];
  wx -= invdzGPU * fact4 * dnoise_sXZ;

  dnoise_sXY = d_rand[vecino2 + ncellsGPU];
  wy -= invdxGPU * fact4 * dnoise_sXY;

  dnoise_sYZ = d_rand[vecino0 + 4*ncellsGPU];
  wy -= invdzGPU * fact4 * dnoise_sYZ;

  dnoise_sXZ = d_rand[vecino2 + 2*ncellsGPU];
  wz -= invdxGPU * fact4 * dnoise_sXZ;

  dnoise_sYZ = d_rand[vecino1 + 4*ncellsGPU];
  wz -= invdyGPU * fact4 * dnoise_sYZ;

  //NOISE part, second random number
  dnoise_sXX = d_rand[6*ncellsGPU + 9*npGPU + vecino3];
  wx += invdxGPU * fact1 * dnoise_sXX;

  dnoise_sYY = d_rand[6*ncellsGPU + 9*npGPU + vecino4 + 3*ncellsGPU];
  wy += invdyGPU * fact1 * dnoise_sYY;

  dnoise_sZZ = d_rand[6*ncellsGPU + 9*npGPU + vecino5 + 5*ncellsGPU];
  wz += invdzGPU * fact1 * dnoise_sZZ;

  dnoise_sXY = d_rand[6*ncellsGPU + 9*npGPU + i + ncellsGPU];
  wx += invdyGPU * fact4 * dnoise_sXY;
  wy += invdxGPU * fact4 * dnoise_sXY;

  dnoise_sXZ = d_rand[6*ncellsGPU + 9*npGPU + i + 2*ncellsGPU];
  wx += invdzGPU * fact4 * dnoise_sXZ;
  wz += invdxGPU * fact4 * dnoise_sXZ;

  dnoise_sYZ = d_rand[6*ncellsGPU + 9*npGPU + i + 4*ncellsGPU];
  wy += invdzGPU * fact4 * dnoise_sYZ;
  wz += invdyGPU * fact4 * dnoise_sYZ;

  dnoise_sXX = d_rand[6*ncellsGPU + 9*npGPU + i];
  wx -= invdxGPU * fact1 * dnoise_sXX;

  dnoise_sYY = d_rand[6*ncellsGPU + 9*npGPU + i + 3*ncellsGPU];
  wy -= invdyGPU * fact1 * dnoise_sYY;

  dnoise_sZZ = d_rand[6*ncellsGPU + 9*npGPU + i + 5*ncellsGPU];
  wz -= invdzGPU * fact1 * dnoise_sZZ;

  dnoise_sXY = d_rand[6*ncellsGPU + 9*npGPU + vecino1 + ncellsGPU];
  wx -= invdyGPU * fact4 * dnoise_sXY;

  dnoise_sXZ = d_rand[6*ncellsGPU + 9*npGPU + vecino0 + 2*ncellsGPU];
  wx -= invdzGPU * fact4 * dnoise_sXZ;

  dnoise_sXY = d_rand[6*ncellsGPU + 9*npGPU + vecino2 + ncellsGPU];
  wy -= invdxGPU * fact4 * dnoise_sXY;

  dnoise_sYZ = d_rand[6*ncellsGPU + 9*npGPU + vecino0 + 4*ncellsGPU];
  wy -= invdzGPU * fact4 * dnoise_sYZ;

  dnoise_sXZ = d_rand[6*ncellsGPU + 9*npGPU + vecino2 + 2*ncellsGPU];
  wz -= invdxGPU * fact4 * dnoise_sXZ;

  dnoise_sYZ = d_rand[6*ncellsGPU + 9*npGPU + vecino1 + 4*ncellsGPU];
  wz -= invdyGPU * fact4 * dnoise_sYZ;



  WxZ[i].x = wx;
  WyZ[i].x = wy;
  WzZ[i].x = wz;

  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

}











__global__ void updateParticlesStokesLimitBigSystem_2(const double* rxcellGPU,
						      const double* rycellGPU,
						      const double* rzcellGPU,
						      double* rxboundaryGPU,  //q^{n}
						      double* ryboundaryGPU, 
						      double* rzboundaryGPU,
						      const double* vxboundaryGPU, //Fx^n
						      const double* vyboundaryGPU, 
						      const double* vzboundaryGPU, 
						      const double* vxGPU, //v^n
						      const double* vyGPU, 
						      const double* vzGPU,
						      const double* d_rand,
						      particlesincell* pc, 
						      const vecinos* pVecinos,
						      int* errorKernel){
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=(npGPU)) return;   

  double rx = fetch_double(texrxboundaryGPU,nboundaryGPU+i);
  double ry = fetch_double(texryboundaryGPU,nboundaryGPU+i);
  double rz = fetch_double(texrzboundaryGPU,nboundaryGPU+i);
  
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;

  double r, rp, rm;
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;
  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;

  int icelx, icely, icelz;

  {
    int mxmy = mxGPU * mytGPU;
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rx - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;
    r = ry - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*mytGPU) % mytGPU;

    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;
    r = rz - 0.5*dzGPU;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jzdz = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;

    icelx  = jxdx;
    icelx += jy * mxGPU;
    icelx += jz * mxmy;

    icely  = jx;
    icely += jydy * mxGPU;
    icely += jz * mxmy;

    icelz  = jx;
    icelz += jy * mxGPU;
    icelz += jzdz * mxmy;
  }
  
  //VELOCITY IN THE X DIRECTION
  vecino0 = pVecinos->vecino0GPU[icelx];
  vecino1 = pVecinos->vecino1GPU[icelx];
  vecino2 = pVecinos->vecino2GPU[icelx];
  vecino3 = pVecinos->vecino3GPU[icelx];
  vecino4 = pVecinos->vecino4GPU[icelx];
  vecino5 = pVecinos->vecino5GPU[icelx];
  vecinopxpy = pVecinos->vecinopxpyGPU[icelx];
  vecinopxmy = pVecinos->vecinopxmyGPU[icelx];
  vecinopxpz = pVecinos->vecinopxpzGPU[icelx];
  vecinopxmz = pVecinos->vecinopxmzGPU[icelx];
  vecinomxpy = pVecinos->vecinomxpyGPU[icelx];
  vecinomxmy = pVecinos->vecinomxmyGPU[icelx];
  vecinomxpz = pVecinos->vecinomxpzGPU[icelx];
  vecinomxmz = pVecinos->vecinomxmzGPU[icelx];
  vecinopypz = pVecinos->vecinopypzGPU[icelx];
  vecinopymz = pVecinos->vecinopymzGPU[icelx];
  vecinomypz = pVecinos->vecinomypzGPU[icelx];
  vecinomymz = pVecinos->vecinomymzGPU[icelx];
  vecinopxpypz = pVecinos->vecinopxpypzGPU[icelx];
  vecinopxpymz = pVecinos->vecinopxpymzGPU[icelx];
  vecinopxmypz = pVecinos->vecinopxmypzGPU[icelx];
  vecinopxmymz = pVecinos->vecinopxmymzGPU[icelx];
  vecinomxpypz = pVecinos->vecinomxpypzGPU[icelx];
  vecinomxpymz = pVecinos->vecinomxpymzGPU[icelx];
  vecinomxmypz = pVecinos->vecinomxmypzGPU[icelx];
  vecinomxmymz = pVecinos->vecinomxmymzGPU[icelx];
  
  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);
  
  r =  (ry - rycellGPU[icelx]);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelx]);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);
  
  double v = dlxm * dlym * dlzm * vxGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vxGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vxGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vxGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vxGPU[vecino2] +
    dlxm * dly  * dlzp * vxGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vxGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vxGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vxGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vxGPU[vecinomymz] +
    dlx  * dlym * dlz  * vxGPU[vecino1] +
    dlx  * dlym * dlzp * vxGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vxGPU[vecino0] + 
    dlx  * dly  * dlz  * vxGPU[icelx] + 
    dlx  * dly  * dlzp * vxGPU[vecino5] + 
    dlx  * dlyp * dlzm * vxGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vxGPU[vecino4] + 
    dlx  * dlyp * dlzp * vxGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vxGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vxGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vxGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vxGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vxGPU[vecino3] +
    dlxp * dly  * dlzp * vxGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vxGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vxGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vxGPU[vecinopxpypz];


  double rxNew = rxboundaryGPU[nboundaryGPU+i] + dtGPU * v;
  if(setExtraMobilityGPU){
    rxNew += dtGPU*extraMobilityGPU * vxboundaryGPU[nboundaryGPU+i]
      + fact2GPU * (d_rand[6*ncellsGPU + 3*npGPU + i] + d_rand[6*ncellsGPU + 6*npGPU + i]);
  }
  rxboundaryGPU[nboundaryGPU+i] = rxNew;


  //VELOCITY IN THE Y DIRECTION
  vecino0 = pVecinos->vecino0GPU[icely];
  vecino1 = pVecinos->vecino1GPU[icely];
  vecino2 = pVecinos->vecino2GPU[icely];
  vecino3 = pVecinos->vecino3GPU[icely];
  vecino4 = pVecinos->vecino4GPU[icely];
  vecino5 = pVecinos->vecino5GPU[icely];
  vecinopxpy = pVecinos->vecinopxpyGPU[icely];
  vecinopxmy = pVecinos->vecinopxmyGPU[icely];
  vecinopxpz = pVecinos->vecinopxpzGPU[icely];
  vecinopxmz = pVecinos->vecinopxmzGPU[icely];
  vecinomxpy = pVecinos->vecinomxpyGPU[icely];
  vecinomxmy = pVecinos->vecinomxmyGPU[icely];
  vecinomxpz = pVecinos->vecinomxpzGPU[icely];
  vecinomxmz = pVecinos->vecinomxmzGPU[icely];
  vecinopypz = pVecinos->vecinopypzGPU[icely];
  vecinopymz = pVecinos->vecinopymzGPU[icely];
  vecinomypz = pVecinos->vecinomypzGPU[icely];
  vecinomymz = pVecinos->vecinomymzGPU[icely];
  vecinopxpypz = pVecinos->vecinopxpypzGPU[icely];
  vecinopxpymz = pVecinos->vecinopxpymzGPU[icely];
  vecinopxmypz = pVecinos->vecinopxmypzGPU[icely];
  vecinopxmymz = pVecinos->vecinopxmymzGPU[icely];
  vecinomxpypz = pVecinos->vecinomxpypzGPU[icely];
  vecinomxpymz = pVecinos->vecinomxpymzGPU[icely];
  vecinomxmypz = pVecinos->vecinomxmypzGPU[icely];
  vecinomxmymz = pVecinos->vecinomxmymzGPU[icely];  

  r =  (rx - rxcellGPU[icely]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icely]);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);


  v = dlxm * dlym * dlzm * vyGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vyGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vyGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vyGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vyGPU[vecino2] +
    dlxm * dly  * dlzp * vyGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vyGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vyGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vyGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vyGPU[vecinomymz] +
    dlx  * dlym * dlz  * vyGPU[vecino1] +
    dlx  * dlym * dlzp * vyGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vyGPU[vecino0] + 
    dlx  * dly  * dlz  * vyGPU[icely] + 
    dlx  * dly  * dlzp * vyGPU[vecino5] + 
    dlx  * dlyp * dlzm * vyGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vyGPU[vecino4] + 
    dlx  * dlyp * dlzp * vyGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vyGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vyGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vyGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vyGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vyGPU[vecino3] +
    dlxp * dly  * dlzp * vyGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vyGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vyGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vyGPU[vecinopxpypz];
  

  double ryNew = ryboundaryGPU[nboundaryGPU+i] + dtGPU * v;
  if(setExtraMobilityGPU){
    ryNew += dtGPU*extraMobilityGPU * vyboundaryGPU[nboundaryGPU+i]
      + fact2GPU * (d_rand[6*ncellsGPU + 4*npGPU + i] + d_rand[6*ncellsGPU + 7*npGPU + i]);
  }
  ryboundaryGPU[nboundaryGPU+i] = ryNew;
 
  //VELOCITY IN THE Z DIRECTION
  vecino0 = pVecinos->vecino0GPU[icelz];
  vecino1 = pVecinos->vecino1GPU[icelz];
  vecino2 = pVecinos->vecino2GPU[icelz];
  vecino3 = pVecinos->vecino3GPU[icelz];
  vecino4 = pVecinos->vecino4GPU[icelz];
  vecino5 = pVecinos->vecino5GPU[icelz];
  vecinopxpy = pVecinos->vecinopxpyGPU[icelz];
  vecinopxmy = pVecinos->vecinopxmyGPU[icelz];
  vecinopxpz = pVecinos->vecinopxpzGPU[icelz];
  vecinopxmz = pVecinos->vecinopxmzGPU[icelz];
  vecinomxpy = pVecinos->vecinomxpyGPU[icelz];
  vecinomxmy = pVecinos->vecinomxmyGPU[icelz];
  vecinomxpz = pVecinos->vecinomxpzGPU[icelz];
  vecinomxmz = pVecinos->vecinomxmzGPU[icelz];
  vecinopypz = pVecinos->vecinopypzGPU[icelz];
  vecinopymz = pVecinos->vecinopymzGPU[icelz];
  vecinomypz = pVecinos->vecinomypzGPU[icelz];
  vecinomymz = pVecinos->vecinomymzGPU[icelz];
  vecinopxpypz = pVecinos->vecinopxpypzGPU[icelz];
  vecinopxpymz = pVecinos->vecinopxpymzGPU[icelz];
  vecinopxmypz = pVecinos->vecinopxmypzGPU[icelz];
  vecinopxmymz = pVecinos->vecinopxmymzGPU[icelz];
  vecinomxpypz = pVecinos->vecinomxpypzGPU[icelz];
  vecinomxpymz = pVecinos->vecinomxpymzGPU[icelz];
  vecinomxmypz = pVecinos->vecinomxmypzGPU[icelz];
  vecinomxmymz = pVecinos->vecinomxmymzGPU[icelz];  

  r =  (rx - rxcellGPU[icelz]);
  r =  (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rp = auxdx * (r - dxGPU);
  rm = auxdx * (r + dxGPU);
  r  = auxdx * r;
  dlx = delta(1.5*r);
  dlxp = delta(1.5*rp);
  dlxm = delta(1.5*rm);

  r =  (ry - rycellGPU[icelz]);
  r =  (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (r - dyGPU);
  rm = auxdy * (r + dyGPU);
  r  = auxdy * r;
  dly = delta(1.5*r);
  dlyp = delta(1.5*rp);
  dlym = delta(1.5*rm);

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  r =  (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (r - dzGPU);
  rm = auxdz * (r + dzGPU);
  r  = auxdz * r;
  dlz = delta(1.5*r);
  dlzp = delta(1.5*rp);
  dlzm = delta(1.5*rm);

 
  v = dlxm * dlym * dlzm * vzGPU[vecinomxmymz] +
    dlxm * dlym * dlz  * vzGPU[vecinomxmy] +
    dlxm * dlym * dlzp * vzGPU[vecinomxmypz] +
    dlxm * dly  * dlzm * vzGPU[vecinomxmz] + 
    dlxm * dly  * dlz  * vzGPU[vecino2] +
    dlxm * dly  * dlzp * vzGPU[vecinomxpz] +
    dlxm * dlyp * dlzm * vzGPU[vecinomxpymz] +
    dlxm * dlyp * dlz  * vzGPU[vecinomxpy] +
    dlxm * dlyp * dlzp * vzGPU[vecinomxpypz] +
    dlx  * dlym * dlzm * vzGPU[vecinomymz] +
    dlx  * dlym * dlz  * vzGPU[vecino1] +
    dlx  * dlym * dlzp * vzGPU[vecinomypz] + 
    dlx  * dly  * dlzm * vzGPU[vecino0] + 
    dlx  * dly  * dlz  * vzGPU[icelz] + 
    dlx  * dly  * dlzp * vzGPU[vecino5] + 
    dlx  * dlyp * dlzm * vzGPU[vecinopymz] + 
    dlx  * dlyp * dlz  * vzGPU[vecino4] + 
    dlx  * dlyp * dlzp * vzGPU[vecinopypz] + 
    dlxp * dlym * dlzm * vzGPU[vecinopxmymz] + 
    dlxp * dlym * dlz  * vzGPU[vecinopxmy] + 
    dlxp * dlym * dlzp * vzGPU[vecinopxmypz] +
    dlxp * dly  * dlzm * vzGPU[vecinopxmz] + 
    dlxp * dly  * dlz  * vzGPU[vecino3] +
    dlxp * dly  * dlzp * vzGPU[vecinopxpz] +
    dlxp * dlyp * dlzm * vzGPU[vecinopxpymz] +
    dlxp * dlyp * dlz  * vzGPU[vecinopxpy] +
    dlxp * dlyp * dlzp * vzGPU[vecinopxpypz];


  double rzNew = rzboundaryGPU[nboundaryGPU+i] + dtGPU * v;
  if(setExtraMobilityGPU){
    rzNew += dtGPU*extraMobilityGPU * vzboundaryGPU[nboundaryGPU+i]
      + fact2GPU * (d_rand[6*ncellsGPU + 5*npGPU + i] + d_rand[6*ncellsGPU + 8*npGPU + i]);           
  }
  rzboundaryGPU[nboundaryGPU+i] = rzNew;

}
