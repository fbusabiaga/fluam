
// Filename: kernelConstructWMHD.cu
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



//In this kernel we construct the vector W
//In the first substep
//  W = u^n + 0.5*dt*nu*L*u^n + Advection(u^n) + (dt/rho)*f^n_{noise}
//
//In the second substep
//  W = u^n + 0.5*dt*nu*L*u^n + Advection(u^{n+1/2}) + (dt/rho)*f^n_{noise}
//with u^{n+1/2} = 0.5 * (u^n + u^{n+1}_{result from first substep})


__global__ void kernelConstructWMHD_1(const double *bxGPU, 
				      const double *byGPU, 
				      const double *bzGPU, 
				      cufftDoubleComplex *vxZ, 
				      cufftDoubleComplex *vyZ, 
				      cufftDoubleComplex *vzZ, 
				      cufftDoubleComplex *WxZ, 
				      cufftDoubleComplex *WyZ, 
				      cufftDoubleComplex *WzZ, 
				      double *d_rand){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double Vx, Vy, Vz;
  double vx, vy, vz;
  double vx0, vx1, vx2, vx3, vx4, vx5;
  double vy0, vy1, vy2, vy3, vy4, vy5;
  double vz0, vz1, vz2, vz3, vz4, vz5;
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 
  int vecinopxmy, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopymz, vecinomypz;
  double vxmxpy,vxmxpz;
  double vypxmy,vymypz;
  double vzpxmz,vzpymz;
  double Bx, By, Bz;
  double bx, by, bz;
  double bx0, bx1, bx2, bx3, bx4, bx5;
  double by0, by1, by2, by3, by4, by5;
  double bz0, bz1, bz2, bz3, bz4, bz5;
  double bxmxpy,bxmxpz;
  double bypxmy,bymypz;
  double bzpxmz,bzpymz;


  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  vecinopymz = tex1Dfetch(texvecinopymzGPU,i);
  vecinomypz = tex1Dfetch(texvecinomypzGPU,i);

  // Read velocities
  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);
  vx0 = fetch_double(texVxGPU,vecino0);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vx5 = fetch_double(texVxGPU,vecino5);
  vy0 = fetch_double(texVyGPU,vecino0);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);
  vy5 = fetch_double(texVyGPU,vecino5);
  vz0 = fetch_double(texVzGPU,vecino0);
  vz1 = fetch_double(texVzGPU,vecino1);
  vz2 = fetch_double(texVzGPU,vecino2);
  vz3 = fetch_double(texVzGPU,vecino3);
  vz4 = fetch_double(texVzGPU,vecino4);
  vz5 = fetch_double(texVzGPU,vecino5);
  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vxmxpz = fetch_double(texVxGPU,vecinomxpz);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);
  vymypz = fetch_double(texVyGPU,vecinomypz);
  vzpxmz = fetch_double(texVzGPU,vecinopxmz);
  vzpymz = fetch_double(texVzGPU,vecinopymz);

  // Read b
  bx = bxGPU[i];
  by = byGPU[i];
  bz = bzGPU[i];
  bx0 = bxGPU[vecino0];
  bx1 = bxGPU[vecino1];
  bx2 = bxGPU[vecino2];
  bx3 = bxGPU[vecino3];
  bx4 = bxGPU[vecino4];
  bx5 = bxGPU[vecino5];
  by0 = byGPU[vecino0];
  by1 = byGPU[vecino1];
  by2 = byGPU[vecino2];
  by3 = byGPU[vecino3];
  by4 = byGPU[vecino4];
  by5 = byGPU[vecino5];
  bz0 = bzGPU[vecino0];
  bz1 = bzGPU[vecino1];
  bz2 = bzGPU[vecino2];
  bz3 = bzGPU[vecino3];
  bz4 = bzGPU[vecino4];
  bz5 = bzGPU[vecino5];
  bxmxpy = bxGPU[vecinomxpy];
  bxmxpz = bxGPU[vecinomxpz];
  bypxmy = byGPU[vecinopxmy];
  bymypz = byGPU[vecinomypz];
  bzpxmz = bzGPU[vecinopxmz];
  bzpymz = bzGPU[vecinopymz];


  //Laplacian part of velocity
  Vx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  Vx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  Vx += invdzGPU * invdzGPU * (vx5 - 2*vx + vx0);
  Vx  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * Vx;
  Vy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  Vy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  Vy += invdzGPU * invdzGPU * (vy5 - 2*vy + vy0);
  Vy  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * Vy;
  Vz  = invdxGPU * invdxGPU * (vz3 - 2*vz + vz2);
  Vz += invdyGPU * invdyGPU * (vz4 - 2*vz + vz1);
  Vz += invdzGPU * invdzGPU * (vz5 - 2*vz + vz0);
  Vz  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * Vz;

  //Laplacian part of b
  Bx  = invdxGPU * invdxGPU * (bx3 - 2*bx + bx2);
  Bx += invdyGPU * invdyGPU * (bx4 - 2*bx + bx1);
  Bx += invdzGPU * invdzGPU * (bx5 - 2*bx + bx0);
  Bx  = 0.5 * dtGPU * diffusionGPU * Bx;
  By  = invdxGPU * invdxGPU * (by3 - 2*by + by2);
  By += invdyGPU * invdyGPU * (by4 - 2*by + by1);
  By += invdzGPU * invdzGPU * (by5 - 2*by + by0);
  By  = 0.5 * dtGPU * diffusionGPU * By;
  Bz  = invdxGPU * invdxGPU * (bz3 - 2*bz + bz2);
  Bz += invdyGPU * invdyGPU * (bz4 - 2*bz + bz1);
  Bz += invdzGPU * invdzGPU * (bz5 - 2*bz + bz0);
  Bz  = 0.5 * dtGPU * diffusionGPU * Bz;

  //Previous Velocity
  Vx += vx;
  Vy += vy;
  Vz += vz;

  //Previous b
  Bx += bx;
  By += by;
  Bz += bz;
  
  // Advection contribution to velocity eq. (\bv \cdot \bna \bv)
  double advX, advY, advZ; 
  advX  = invdxGPU * ((vx3+vx)*(vx3+vx) - (vx+vx2)*(vx+vx2));
  advX += invdyGPU * ((vx4+vx)*(vy3+vy) - (vx+vx1)*(vypxmy+vy1));
  advX += invdzGPU * ((vx5+vx)*(vz3+vz) - (vx+vx0)*(vzpxmz+vz0));
  // advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(vx4+vx) - (vy+vy2)*(vxmxpy+vx2));
  advY += invdyGPU * ((vy4+vy)*(vy4+vy) - (vy+vy1)*(vy+vy1));
  advY += invdzGPU * ((vy5+vy)*(vz4+vz) - (vy+vy0)*(vzpymz+vz0));
  // advY  = 0.25 * dtGPU * advY;
  advZ  = invdxGPU * ((vz3+vz)*(vx5+vx) - (vz+vz2)*(vxmxpz+vx2));
  advZ += invdyGPU * ((vz4+vz)*(vy5+vy) - (vz+vz1)*(vymypz+vy1));
  advZ += invdzGPU * ((vz5+vz)*(vz5+vz) - (vz+vz0)*(vz+vz0));
  // advZ  = 0.25 * dtGPU * advZ;

  // Advection contribution to velocity eq. (-\bb \cdot \bna \bb)
  advX -= invdxGPU * ((bx3+bx)*(bx3+bx) - (bx+bx2)*(bx+bx2));
  advX -= invdyGPU * ((bx4+bx)*(by3+by) - (bx+bx1)*(bypxmy+by1));
  advX -= invdzGPU * ((bx5+bx)*(bz3+bz) - (bx+bx0)*(bzpxmz+bz0));
  advX  = 0.25 * dtGPU * advX;
  advY -= invdxGPU * ((by3+by)*(bx4+bx) - (by+by2)*(bxmxpy+bx2));
  advY -= invdyGPU * ((by4+by)*(by4+by) - (by+by1)*(by+by1));
  advY -= invdzGPU * ((by5+by)*(bz4+bz) - (by+by0)*(bzpymz+bz0));
  advY  = 0.25 * dtGPU * advY;
  advZ -= invdxGPU * ((bz3+bz)*(bx5+bx) - (bz+bz2)*(bxmxpz+bx2));
  advZ -= invdyGPU * ((bz4+bz)*(by5+by) - (bz+bz1)*(bymypz+by1));
  advZ -= invdzGPU * ((bz5+bz)*(bz5+bz) - (bz+bz0)*(bz+bz0));
  advZ  = 0.25 * dtGPU * advZ;

  //advX=0; advY=0; advZ=0;
  Vx -= advX;
  Vy -= advY;
  Vz -= advZ;

  // Advection contribution to b eq. (\bv \cdot \bna \bb)
  advX  = invdxGPU * ((vx3+vx)*(bx3+bx) - (vx+vx2)*(bx+bx2));
  advX += invdyGPU * ((vx4+vx)*(by3+by) - (vx+vx1)*(bypxmy+by1));
  advX += invdzGPU * ((vx5+vx)*(bz3+bz) - (vx+vx0)*(bzpxmz+bz0));
  // advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(bx4+bx) - (vy+vy2)*(bxmxpy+bx2));
  advY += invdyGPU * ((vy4+vy)*(by4+by) - (vy+vy1)*(by+by1));
  advY += invdzGPU * ((vy5+vy)*(bz4+bz) - (vy+vy0)*(bzpymz+bz0));
  // advY  = 0.25 * dtGPU * advY;
  advZ  = invdxGPU * ((vz3+vz)*(bx5+bx) - (vz+vz2)*(bxmxpz+bx2));
  advZ += invdyGPU * ((vz4+vz)*(by5+by) - (vz+vz1)*(bymypz+by1));
  advZ += invdzGPU * ((vz5+vz)*(bz5+bz) - (vz+vz0)*(bz+bz0));
  // advZ  = 0.25 * dtGPU * advZ;

  // Advection contribution to b eq. (-\bb \cdot \bna \bv)
  advX -= invdxGPU * ((bx3+bx)*(vx3+vx) - (bx+bx2)*(vx+vx2));
  advX -= invdyGPU * ((bx4+bx)*(vy3+vy) - (bx+bx1)*(vypxmy+vy1));
  advX -= invdzGPU * ((bx5+bx)*(vz3+vz) - (bx+bx0)*(vzpxmz+vz0));
  advX  = 0.25 * dtGPU * advX;
  advY -= invdxGPU * ((by3+by)*(vx4+vx) - (by+by2)*(vxmxpy+vx2));
  advY -= invdyGPU * ((by4+by)*(vy4+vy) - (by+by1)*(vy+vy1));
  advY -= invdzGPU * ((by5+by)*(vz4+vz) - (by+by0)*(vzpymz+vz0));
  advY  = 0.25 * dtGPU * advY;
  advZ -= invdxGPU * ((bz3+bz)*(vx5+vx) - (bz+bz2)*(vxmxpz+vx2));
  advZ -= invdyGPU * ((bz4+bz)*(vy5+vy) - (bz+bz1)*(vymypz+vy1));
  advZ -= invdzGPU * ((bz5+bz)*(vz5+vz) - (bz+bz0)*(vz+vz0));
  advZ  = 0.25 * dtGPU * advZ;

  //advX=0; advY=0; advZ=0;
  Bx -= advX;
  By -= advY;
  Bz -= advZ;

  // NOISE part on the velocity eq.
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double dnoise_tr;
  dnoise_tr = d_rand[vecino3] + d_rand[vecino3 + 3*ncellsGPU] + d_rand[vecino3 + 5*ncellsGPU];
  dnoise_sXX = d_rand[vecino3] - dnoise_tr/3.;
  Vx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_tr = d_rand[vecino4] + d_rand[vecino4 + 3*ncellsGPU] + d_rand[vecino4 + 5*ncellsGPU];
  dnoise_sYY = d_rand[vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
  Vy += invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_tr = d_rand[vecino5] + d_rand[vecino5 + 3*ncellsGPU] + d_rand[vecino5 + 5*ncellsGPU];
  dnoise_sZZ = d_rand[vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
  Vz += invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[i + ncellsGPU];
  Vx += invdyGPU * fact4GPU * dnoise_sXY;
  Vy += invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[i + 2*ncellsGPU];
  Vx += invdzGPU * fact4GPU * dnoise_sXZ;
  Vz += invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[i + 4*ncellsGPU];
  Vy += invdzGPU * fact4GPU * dnoise_sYZ;
  Vz += invdyGPU * fact4GPU * dnoise_sYZ;

  dnoise_tr = d_rand[i] + d_rand[i + 3*ncellsGPU] + d_rand[i + 5*ncellsGPU];
  dnoise_sXX = d_rand[i] - dnoise_tr/3.;
  Vx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 3*ncellsGPU] - dnoise_tr/3.;
  Vy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 5*ncellsGPU] - dnoise_tr/3.;
  Vz -= invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  Vx -= invdyGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[vecino0 + 2*ncellsGPU];
  Vx -= invdzGPU * fact4GPU * dnoise_sXZ;

  dnoise_sXY = d_rand[vecino2 + ncellsGPU];
  Vy -= invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sYZ = d_rand[vecino0 + 4*ncellsGPU];
  Vy -= invdzGPU * fact4GPU * dnoise_sYZ;

  dnoise_sXZ = d_rand[vecino2 + 2*ncellsGPU];
  Vz -= invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[vecino1 + 4*ncellsGPU];
  Vz -= invdyGPU * fact4GPU * dnoise_sYZ;
  

  vxZ[i].x = Vx;
  vyZ[i].x = Vy;
  vzZ[i].x = Vz;
  vxZ[i].y = 0;
  vyZ[i].y = 0;
  vzZ[i].y = 0;

  WxZ[i].x = Bx;
  WyZ[i].x = By;
  WzZ[i].x = Bz;
  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

}




__global__ void kernelConstructWMHD_2(const double *bxGPU, 
				      const double *byGPU, 
				      const double *bzGPU, 
				      const double *vxPredictionGPU, 
				      const double *vyPredictionGPU, 
				      const double *vzPredictionGPU, 
				      const double *bxPredictionGPU, 
				      const double *byPredictionGPU, 
				      const double *bzPredictionGPU, 
				      cufftDoubleComplex *vxZ, 
				      cufftDoubleComplex *vyZ, 
				      cufftDoubleComplex *vzZ, 
				      cufftDoubleComplex *WxZ, 
				      cufftDoubleComplex *WyZ, 
				      cufftDoubleComplex *WzZ, 
				      double *d_rand){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double Vx, Vy, Vz;
  double vx, vy, vz;
  double vx0, vx1, vx2, vx3, vx4, vx5;
  double vy0, vy1, vy2, vy3, vy4, vy5;
  double vz0, vz1, vz2, vz3, vz4, vz5;
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 
  int vecinopxmy, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopymz, vecinomypz;
  double vxmxpy,vxmxpz;
  double vypxmy,vymypz;
  double vzpxmz,vzpymz;
  double Bx, By, Bz;
  double bx, by, bz;
  double bx0, bx1, bx2, bx3, bx4, bx5;
  double by0, by1, by2, by3, by4, by5;
  double bz0, bz1, bz2, bz3, bz4, bz5;
  double bxmxpy,bxmxpz;
  double bypxmy,bymypz;
  double bzpxmz,bzpymz;

  // Read neighbors
  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  vecinopymz = tex1Dfetch(texvecinopymzGPU,i);
  vecinomypz = tex1Dfetch(texvecinomypzGPU,i);

  // Read velocity
  vx = fetch_double(texVxGPU,i);
  vy = fetch_double(texVyGPU,i);
  vz = fetch_double(texVzGPU,i);
  vx0 = fetch_double(texVxGPU,vecino0);
  vx1 = fetch_double(texVxGPU,vecino1);
  vx2 = fetch_double(texVxGPU,vecino2);
  vx3 = fetch_double(texVxGPU,vecino3);
  vx4 = fetch_double(texVxGPU,vecino4);
  vx5 = fetch_double(texVxGPU,vecino5);
  vy0 = fetch_double(texVyGPU,vecino0);
  vy1 = fetch_double(texVyGPU,vecino1);
  vy2 = fetch_double(texVyGPU,vecino2);
  vy3 = fetch_double(texVyGPU,vecino3);
  vy4 = fetch_double(texVyGPU,vecino4);
  vy5 = fetch_double(texVyGPU,vecino5);
  vz0 = fetch_double(texVzGPU,vecino0);
  vz1 = fetch_double(texVzGPU,vecino1);
  vz2 = fetch_double(texVzGPU,vecino2);
  vz3 = fetch_double(texVzGPU,vecino3);
  vz4 = fetch_double(texVzGPU,vecino4);
  vz5 = fetch_double(texVzGPU,vecino5);
  vxmxpy = fetch_double(texVxGPU,vecinomxpy);
  vxmxpz = fetch_double(texVxGPU,vecinomxpz);
  vypxmy = fetch_double(texVyGPU,vecinopxmy);
  vymypz = fetch_double(texVyGPU,vecinomypz);
  vzpxmz = fetch_double(texVzGPU,vecinopxmz);
  vzpymz = fetch_double(texVzGPU,vecinopymz);

  // Read b
  bx = bxGPU[i];
  by = byGPU[i];
  bz = bzGPU[i];
  bx0 = bxGPU[vecino0];
  bx1 = bxGPU[vecino1];
  bx2 = bxGPU[vecino2];
  bx3 = bxGPU[vecino3];
  bx4 = bxGPU[vecino4];
  bx5 = bxGPU[vecino5];
  by0 = byGPU[vecino0];
  by1 = byGPU[vecino1];
  by2 = byGPU[vecino2];
  by3 = byGPU[vecino3];
  by4 = byGPU[vecino4];
  by5 = byGPU[vecino5];
  bz0 = bzGPU[vecino0];
  bz1 = bzGPU[vecino1];
  bz2 = bzGPU[vecino2];
  bz3 = bzGPU[vecino3];
  bz4 = bzGPU[vecino4];
  bz5 = bzGPU[vecino5];
  bxmxpy = bxGPU[vecinomxpy];
  bxmxpz = bxGPU[vecinomxpz];
  bypxmy = byGPU[vecinopxmy];
  bymypz = byGPU[vecinomypz];
  bzpxmz = bzGPU[vecinopxmz];
  bzpymz = bzGPU[vecinopymz];

  //Laplacian part velocity
  Vx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  Vx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  Vx += invdzGPU * invdzGPU * (vx5 - 2*vx + vx0);
  Vx  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * Vx;
  Vy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  Vy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  Vy += invdzGPU * invdzGPU * (vy5 - 2*vy + vy0);
  Vy  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * Vy;
  Vz  = invdxGPU * invdxGPU * (vz3 - 2*vz + vz2);
  Vz += invdyGPU * invdyGPU * (vz4 - 2*vz + vz1);
  Vz += invdzGPU * invdzGPU * (vz5 - 2*vz + vz0);
  Vz  = 0.5 * dtGPU * (shearviscosityGPU/densfluidGPU) * Vz;

  //Laplacian part b
  Bx  = invdxGPU * invdxGPU * (bx3 - 2*bx + bx2);
  Bx += invdyGPU * invdyGPU * (bx4 - 2*bx + bx1);
  Bx += invdzGPU * invdzGPU * (bx5 - 2*bx + bx0);
  Bx  = 0.5 * dtGPU * diffusionGPU * Bx;
  By  = invdxGPU * invdxGPU * (by3 - 2*by + by2);
  By += invdyGPU * invdyGPU * (by4 - 2*by + by1);
  By += invdzGPU * invdzGPU * (by5 - 2*by + by0);
  By  = 0.5 * dtGPU * diffusionGPU * By;
  Bz  = invdxGPU * invdxGPU * (bz3 - 2*bz + bz2);
  Bz += invdyGPU * invdyGPU * (bz4 - 2*bz + bz1);
  Bz += invdzGPU * invdzGPU * (bz5 - 2*bz + bz0);
  Bz  = 0.5 * dtGPU * diffusionGPU * Bz;

  //Previous Velocity
  Vx += vx;
  Vy += vy;
  Vz += vz;

  //Previous b
  Bx += bx;
  By += by;
  Bz += bz;
 
  // Read velocity midpoint
  vx = vxPredictionGPU[i];
  vy = vyPredictionGPU[i];
  vz = vzPredictionGPU[i];
  vx0 = vxPredictionGPU[vecino0];
  vx1 = vxPredictionGPU[vecino1];
  vx2 = vxPredictionGPU[vecino2];
  vx3 = vxPredictionGPU[vecino3];
  vx4 = vxPredictionGPU[vecino4];
  vx5 = vxPredictionGPU[vecino5];
  vy0 = vyPredictionGPU[vecino0];
  vy1 = vyPredictionGPU[vecino1];
  vy2 = vyPredictionGPU[vecino2];
  vy3 = vyPredictionGPU[vecino3];
  vy4 = vyPredictionGPU[vecino4];
  vy5 = vyPredictionGPU[vecino5];
  vz0 = vzPredictionGPU[vecino0];
  vz1 = vzPredictionGPU[vecino1];
  vz2 = vzPredictionGPU[vecino2];
  vz3 = vzPredictionGPU[vecino3];
  vz4 = vzPredictionGPU[vecino4];
  vz5 = vzPredictionGPU[vecino5];
  vxmxpy = vxPredictionGPU[vecinomxpy];
  vxmxpz = vxPredictionGPU[vecinomxpz];
  vypxmy = vyPredictionGPU[vecinopxmy];
  vymypz = vyPredictionGPU[vecinomypz];
  vzpxmz = vzPredictionGPU[vecinopxmz];
  vzpymz = vzPredictionGPU[vecinopymz];

  // Read b midpoint
  bx = bxPredictionGPU[i];
  by = byPredictionGPU[i];
  bz = bzPredictionGPU[i];
  bx0 = bxPredictionGPU[vecino0];
  bx1 = bxPredictionGPU[vecino1];
  bx2 = bxPredictionGPU[vecino2];
  bx3 = bxPredictionGPU[vecino3];
  bx4 = bxPredictionGPU[vecino4];
  bx5 = bxPredictionGPU[vecino5];
  by0 = byPredictionGPU[vecino0];
  by1 = byPredictionGPU[vecino1];
  by2 = byPredictionGPU[vecino2];
  by3 = byPredictionGPU[vecino3];
  by4 = byPredictionGPU[vecino4];
  by5 = byPredictionGPU[vecino5];
  bz0 = bzPredictionGPU[vecino0];
  bz1 = bzPredictionGPU[vecino1];
  bz2 = bzPredictionGPU[vecino2];
  bz3 = bzPredictionGPU[vecino3];
  bz4 = bzPredictionGPU[vecino4];
  bz5 = bzPredictionGPU[vecino5];
  bxmxpy = bxPredictionGPU[vecinomxpy];
  bxmxpz = bxPredictionGPU[vecinomxpz];
  bypxmy = byPredictionGPU[vecinopxmy];
  bymypz = byPredictionGPU[vecinomypz];
  bzpxmz = bzPredictionGPU[vecinopxmz];
  bzpymz = bzPredictionGPU[vecinopymz];

  // Advection contribution to velocity eq. (\bv \cdot \bna \bv)
  double advX, advY, advZ; 
  advX  = invdxGPU * ((vx3+vx)*(vx3+vx) - (vx+vx2)*(vx+vx2));
  advX += invdyGPU * ((vx4+vx)*(vy3+vy) - (vx+vx1)*(vypxmy+vy1));
  advX += invdzGPU * ((vx5+vx)*(vz3+vz) - (vx+vx0)*(vzpxmz+vz0));
  // advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(vx4+vx) - (vy+vy2)*(vxmxpy+vx2));
  advY += invdyGPU * ((vy4+vy)*(vy4+vy) - (vy+vy1)*(vy+vy1));
  advY += invdzGPU * ((vy5+vy)*(vz4+vz) - (vy+vy0)*(vzpymz+vz0));
  // advY  = 0.25 * dtGPU * advY;
  advZ  = invdxGPU * ((vz3+vz)*(vx5+vx) - (vz+vz2)*(vxmxpz+vx2));
  advZ += invdyGPU * ((vz4+vz)*(vy5+vy) - (vz+vz1)*(vymypz+vy1));
  advZ += invdzGPU * ((vz5+vz)*(vz5+vz) - (vz+vz0)*(vz+vz0));
  // advZ  = 0.25 * dtGPU * advZ;

  // Advection contribution to velocity eq. (-\bb \cdot \bna \bb)
  advX -= invdxGPU * ((bx3+bx)*(bx3+bx) - (bx+bx2)*(bx+bx2));
  advX -= invdyGPU * ((bx4+bx)*(by3+by) - (bx+bx1)*(bypxmy+by1));
  advX -= invdzGPU * ((bx5+bx)*(bz3+bz) - (bx+bx0)*(bzpxmz+bz0));
  advX  = 0.25 * dtGPU * advX;
  advY -= invdxGPU * ((by3+by)*(bx4+bx) - (by+by2)*(bxmxpy+bx2));
  advY -= invdyGPU * ((by4+by)*(by4+by) - (by+by1)*(by+by1));
  advY -= invdzGPU * ((by5+by)*(bz4+bz) - (by+by0)*(bzpymz+bz0));
  advY  = 0.25 * dtGPU * advY;
  advZ -= invdxGPU * ((bz3+bz)*(bx5+bx) - (bz+bz2)*(bxmxpz+bx2));
  advZ -= invdyGPU * ((bz4+bz)*(by5+by) - (bz+bz1)*(bymypz+by1));
  advZ -= invdzGPU * ((bz5+bz)*(bz5+bz) - (bz+bz0)*(bz+bz0));
  advZ  = 0.25 * dtGPU * advZ;

  //advX=0; advY=0; advZ=0;
  Vx -= advX;
  Vy -= advY;
  Vz -= advZ;

  // Advection contribution to b eq. (\bv \cdot \bna \bb)
  advX  = invdxGPU * ((vx3+vx)*(bx3+bx) - (vx+vx2)*(bx+bx2));
  advX += invdyGPU * ((vx4+vx)*(by3+by) - (vx+vx1)*(bypxmy+by1));
  advX += invdzGPU * ((vx5+vx)*(bz3+bz) - (vx+vx0)*(bzpxmz+bz0));
  // advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(bx4+bx) - (vy+vy2)*(bxmxpy+bx2));
  advY += invdyGPU * ((vy4+vy)*(by4+by) - (vy+vy1)*(by+by1));
  advY += invdzGPU * ((vy5+vy)*(bz4+bz) - (vy+vy0)*(bzpymz+bz0));
  // advY  = 0.25 * dtGPU * advY;
  advZ  = invdxGPU * ((vz3+vz)*(bx5+bx) - (vz+vz2)*(bxmxpz+bx2));
  advZ += invdyGPU * ((vz4+vz)*(by5+by) - (vz+vz1)*(bymypz+by1));
  advZ += invdzGPU * ((vz5+vz)*(bz5+bz) - (vz+vz0)*(bz+bz0));
  // advZ  = 0.25 * dtGPU * advZ;

  // Advection contribution to b eq. (-\bb \cdot \bna \bv)
  advX -= invdxGPU * ((bx3+bx)*(vx3+vx) - (bx+bx2)*(vx+vx2));
  advX -= invdyGPU * ((bx4+bx)*(vy3+vy) - (bx+bx1)*(vypxmy+vy1));
  advX -= invdzGPU * ((bx5+bx)*(vz3+vz) - (bx+bx0)*(vzpxmz+vz0));
  advX  = 0.25 * dtGPU * advX;
  advY -= invdxGPU * ((by3+by)*(vx4+vx) - (by+by2)*(vxmxpy+vx2));
  advY -= invdyGPU * ((by4+by)*(vy4+vy) - (by+by1)*(vy+vy1));
  advY -= invdzGPU * ((by5+by)*(vz4+vz) - (by+by0)*(vzpymz+vz0));
  advY  = 0.25 * dtGPU * advY;
  advZ -= invdxGPU * ((bz3+bz)*(vx5+vx) - (bz+bz2)*(vxmxpz+vx2));
  advZ -= invdyGPU * ((bz4+bz)*(vy5+vy) - (bz+bz1)*(vymypz+vy1));
  advZ -= invdzGPU * ((bz5+bz)*(vz5+vz) - (bz+bz0)*(vz+vz0));
  advZ  = 0.25 * dtGPU * advZ;

  //advX=0; advY=0; advZ=0;
  Bx -= advX;
  By -= advY;
  Bz -= advZ;

  // NOISE part velocity eq.
  double dnoise_sXX, dnoise_sXY, dnoise_sXZ;
  double dnoise_sYY, dnoise_sYZ;
  double dnoise_sZZ;
  double dnoise_tr;
  dnoise_tr = d_rand[vecino3] + d_rand[vecino3 + 3*ncellsGPU] + d_rand[vecino3 + 5*ncellsGPU];
  dnoise_sXX = d_rand[vecino3] - dnoise_tr/3.;
  Vx += invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_tr = d_rand[vecino4] + d_rand[vecino4 + 3*ncellsGPU] + d_rand[vecino4 + 5*ncellsGPU];
  dnoise_sYY = d_rand[vecino4 + 3*ncellsGPU] - dnoise_tr/3.;
  Vy += invdyGPU * fact1GPU * dnoise_sYY;
  
  dnoise_tr = d_rand[vecino5] + d_rand[vecino5 + 3*ncellsGPU] + d_rand[vecino5 + 5*ncellsGPU];
  dnoise_sZZ = d_rand[vecino5 + 5*ncellsGPU] - dnoise_tr/3.;
  Vz += invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[i + ncellsGPU];
  Vx += invdyGPU * fact4GPU * dnoise_sXY;
  Vy += invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[i + 2*ncellsGPU];
  Vx += invdzGPU * fact4GPU * dnoise_sXZ;
  Vz += invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[i + 4*ncellsGPU];
  Vy += invdzGPU * fact4GPU * dnoise_sYZ;
  Vz += invdyGPU * fact4GPU * dnoise_sYZ;

  dnoise_tr = d_rand[i] + d_rand[i + 3*ncellsGPU] + d_rand[i + 5*ncellsGPU];
  dnoise_sXX = d_rand[i] - dnoise_tr/3.;
  Vx -= invdxGPU * fact1GPU * dnoise_sXX;

  dnoise_sYY = d_rand[i + 3*ncellsGPU] - dnoise_tr/3.;
  Vy -= invdyGPU * fact1GPU * dnoise_sYY;

  dnoise_sZZ = d_rand[i + 5*ncellsGPU] - dnoise_tr/3.;
  Vz -= invdzGPU * fact1GPU * dnoise_sZZ;

  dnoise_sXY = d_rand[vecino1 + ncellsGPU];
  Vx -= invdyGPU * fact4GPU * dnoise_sXY;

  dnoise_sXZ = d_rand[vecino0 + 2*ncellsGPU];
  Vx -= invdzGPU * fact4GPU * dnoise_sXZ;

  dnoise_sXY = d_rand[vecino2 + ncellsGPU];
  Vy -= invdxGPU * fact4GPU * dnoise_sXY;

  dnoise_sYZ = d_rand[vecino0 + 4*ncellsGPU];
  Vy -= invdzGPU * fact4GPU * dnoise_sYZ;

  dnoise_sXZ = d_rand[vecino2 + 2*ncellsGPU];
  Vz -= invdxGPU * fact4GPU * dnoise_sXZ;

  dnoise_sYZ = d_rand[vecino1 + 4*ncellsGPU];
  Vz -= invdyGPU * fact4GPU * dnoise_sYZ;
  
  vxZ[i].x = Vx;
  vyZ[i].x = Vy;
  vzZ[i].x = Vz;
  vxZ[i].y = 0;
  vyZ[i].y = 0;
  vzZ[i].y = 0;

  WxZ[i].x = Bx;
  WyZ[i].x = By;
  WzZ[i].x = Bz;
  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

}
























__global__ void kernelConstructWMHD_explicit(const double *vxGPU, 
					     const double *vyGPU, 
					     const double *vzGPU, 
					     const double *bxGPU, 
					     const double *byGPU, 
					     const double *bzGPU, 
					     cufftDoubleComplex *vxZ, 
					     cufftDoubleComplex *vyZ, 
					     cufftDoubleComplex *vzZ, 
					     cufftDoubleComplex *WxZ, 
					     cufftDoubleComplex *WyZ, 
					     cufftDoubleComplex *WzZ){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double Vx, Vy, Vz;
  double vx, vy, vz;
  double vx0, vx1, vx2, vx3, vx4, vx5;
  double vy0, vy1, vy2, vy3, vy4, vy5;
  double vz0, vz1, vz2, vz3, vz4, vz5;
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5; 
  int vecinopxmy, vecinopxmz;
  int vecinomxpy, vecinomxpz;
  int vecinopymz, vecinomypz;
  double vxmxpy,vxmxpz;
  double vypxmy,vymypz;
  double vzpxmz,vzpymz;
  double Bx, By, Bz;
  double bx, by, bz;
  double bx0, bx1, bx2, bx3, bx4, bx5;
  double by0, by1, by2, by3, by4, by5;
  double bz0, bz1, bz2, bz3, bz4, bz5;
  double bxmxpy,bxmxpz;
  double bypxmy,bymypz;
  double bzpxmz,bzpymz;


  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);
  vecino3 = tex1Dfetch(texvecino3GPU,i);
  vecino4 = tex1Dfetch(texvecino4GPU,i);
  vecino5 = tex1Dfetch(texvecino5GPU,i);
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU,i);
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU,i);
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU,i);
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU,i);
  vecinopymz = tex1Dfetch(texvecinopymzGPU,i);
  vecinomypz = tex1Dfetch(texvecinomypzGPU,i);

  // Read velocities
  vx = vxGPU[i];
  vy = vyGPU[i];
  vz = vzGPU[i];
  vx0 = vxGPU[vecino0];
  vx1 = vxGPU[vecino1];
  vx2 = vxGPU[vecino2];
  vx3 = vxGPU[vecino3];
  vx4 = vxGPU[vecino4];
  vx5 = vxGPU[vecino5];
  vy0 = vyGPU[vecino0];
  vy1 = vyGPU[vecino1];
  vy2 = vyGPU[vecino2];
  vy3 = vyGPU[vecino3];
  vy4 = vyGPU[vecino4];
  vy5 = vyGPU[vecino5];
  vz0 = vzGPU[vecino0];
  vz1 = vzGPU[vecino1];
  vz2 = vzGPU[vecino2];
  vz3 = vzGPU[vecino3];
  vz4 = vzGPU[vecino4];
  vz5 = vzGPU[vecino5];
  vxmxpy = vxGPU[vecinomxpy];
  vxmxpz = vxGPU[vecinomxpz];
  vypxmy = vyGPU[vecinopxmy];
  vymypz = vyGPU[vecinomypz];
  vzpxmz = vzGPU[vecinopxmz];
  vzpymz = vzGPU[vecinopymz];

  // Read b
  bx = bxGPU[i];
  by = byGPU[i];
  bz = bzGPU[i];
  bx0 = bxGPU[vecino0];
  bx1 = bxGPU[vecino1];
  bx2 = bxGPU[vecino2];
  bx3 = bxGPU[vecino3];
  bx4 = bxGPU[vecino4];
  bx5 = bxGPU[vecino5];
  by0 = byGPU[vecino0];
  by1 = byGPU[vecino1];
  by2 = byGPU[vecino2];
  by3 = byGPU[vecino3];
  by4 = byGPU[vecino4];
  by5 = byGPU[vecino5];
  bz0 = bzGPU[vecino0];
  bz1 = bzGPU[vecino1];
  bz2 = bzGPU[vecino2];
  bz3 = bzGPU[vecino3];
  bz4 = bzGPU[vecino4];
  bz5 = bzGPU[vecino5];
  bxmxpy = bxGPU[vecinomxpy];
  bxmxpz = bxGPU[vecinomxpz];
  bypxmy = byGPU[vecinopxmy];
  bymypz = byGPU[vecinomypz];
  bzpxmz = bzGPU[vecinopxmz];
  bzpymz = bzGPU[vecinopymz];


  //Laplacian part of velocity
  Vx  = invdxGPU * invdxGPU * (vx3 - 2*vx + vx2);
  Vx += invdyGPU * invdyGPU * (vx4 - 2*vx + vx1);
  Vx += invdzGPU * invdzGPU * (vx5 - 2*vx + vx0);
  Vx  = dtGPU * (shearviscosityGPU/densfluidGPU) * Vx;
  Vy  = invdxGPU * invdxGPU * (vy3 - 2*vy + vy2);
  Vy += invdyGPU * invdyGPU * (vy4 - 2*vy + vy1);
  Vy += invdzGPU * invdzGPU * (vy5 - 2*vy + vy0);
  Vy  = dtGPU * (shearviscosityGPU/densfluidGPU) * Vy;
  Vz  = invdxGPU * invdxGPU * (vz3 - 2*vz + vz2);
  Vz += invdyGPU * invdyGPU * (vz4 - 2*vz + vz1);
  Vz += invdzGPU * invdzGPU * (vz5 - 2*vz + vz0);
  Vz  = dtGPU * (shearviscosityGPU/densfluidGPU) * Vz;

  //Laplacian part of b
  Bx  = invdxGPU * invdxGPU * (bx3 - 2*bx + bx2);
  Bx += invdyGPU * invdyGPU * (bx4 - 2*bx + bx1);
  Bx += invdzGPU * invdzGPU * (bx5 - 2*bx + bx0);
  Bx  = dtGPU * diffusionGPU * Bx;
  By  = invdxGPU * invdxGPU * (by3 - 2*by + by2);
  By += invdyGPU * invdyGPU * (by4 - 2*by + by1);
  By += invdzGPU * invdzGPU * (by5 - 2*by + by0);
  By  = dtGPU * diffusionGPU * By;
  Bz  = invdxGPU * invdxGPU * (bz3 - 2*bz + bz2);
  Bz += invdyGPU * invdyGPU * (bz4 - 2*bz + bz1);
  Bz += invdzGPU * invdzGPU * (bz5 - 2*bz + bz0);
  Bz  = dtGPU * diffusionGPU * Bz;

  //Previous Velocity
  Vx += vx;
  Vy += vy;
  Vz += vz;

  //Previous b
  Bx += bx;
  By += by;
  Bz += bz;
  
  // Advection contribution to velocity eq. (-\bv \cdot \bna \bv)
  double advX, advY, advZ; 
  advX  = invdxGPU * ((vx3+vx)*(vx3+vx) - (vx+vx2)*(vx    +vx2));
  advX += invdyGPU * ((vx4+vx)*(vy3+vy) - (vx+vx1)*(vypxmy+vy1));
  advX += invdzGPU * ((vx5+vx)*(vz3+vz) - (vx+vx0)*(vzpxmz+vz0));
  // advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(vx4+vx) - (vy+vy2)*(vxmxpy+vx2));
  advY += invdyGPU * ((vy4+vy)*(vy4+vy) - (vy+vy1)*(vy    +vy1));
  advY += invdzGPU * ((vy5+vy)*(vz4+vz) - (vy+vy0)*(vzpymz+vz0));
  // advY  = 0.25 * dtGPU * advY;
  advZ  = invdxGPU * ((vz3+vz)*(vx5+vx) - (vz+vz2)*(vxmxpz+vx2));
  advZ += invdyGPU * ((vz4+vz)*(vy5+vy) - (vz+vz1)*(vymypz+vy1));
  advZ += invdzGPU * ((vz5+vz)*(vz5+vz) - (vz+vz0)*(vz    +vz0));
  // advZ  = 0.25 * dtGPU * advZ;

  // Advection contribution to velocity eq. (\bb \cdot \bna \bb)
  advX -= invdxGPU * ((bx3+bx)*(bx3+bx) - (bx+bx2)*(bx    +bx2));
  advX -= invdyGPU * ((bx4+bx)*(by3+by) - (bx+bx1)*(bypxmy+by1));
  advX -= invdzGPU * ((bx5+bx)*(bz3+bz) - (bx+bx0)*(bzpxmz+bz0));
  advX  = 0.25 * dtGPU * advX;
  advY -= invdxGPU * ((by3+by)*(bx4+bx) - (by+by2)*(bxmxpy+bx2));
  advY -= invdyGPU * ((by4+by)*(by4+by) - (by+by1)*(by    +by1));
  advY -= invdzGPU * ((by5+by)*(bz4+bz) - (by+by0)*(bzpymz+bz0));
  advY  = 0.25 * dtGPU * advY;
  advZ -= invdxGPU * ((bz3+bz)*(bx5+bx) - (bz+bz2)*(bxmxpz+bx2));
  advZ -= invdyGPU * ((bz4+bz)*(by5+by) - (bz+bz1)*(bymypz+by1));
  advZ -= invdzGPU * ((bz5+bz)*(bz5+bz) - (bz+bz0)*(bz    +bz0));
  advZ  = 0.25 * dtGPU * advZ;

  //advX=0; advY=0; advZ=0;
  Vx -= advX;
  Vy -= advY;
  Vz -= advZ;

  // Advection contribution to b eq. (\bb \cdot \bna \bv = \bna \cdot (vb) )
  advX  = invdxGPU * ((vx3+vx)*(bx3+bx) - (vx+vx2)*(bx    +bx2));
  advX += invdyGPU * ((vx4+vx)*(by3+by) - (vx+vx1)*(bypxmy+by1));
  advX += invdzGPU * ((vx5+vx)*(bz3+bz) - (vx+vx0)*(bzpxmz+bz0));
  // advX  = 0.25 * dtGPU * advX;
  advY  = invdxGPU * ((vy3+vy)*(bx4+bx) - (vy+vy2)*(bxmxpy+bx2));
  advY += invdyGPU * ((vy4+vy)*(by4+by) - (vy+vy1)*(by    +by1));
  advY += invdzGPU * ((vy5+vy)*(bz4+bz) - (vy+vy0)*(bzpymz+bz0));
  // advY  = 0.25 * dtGPU * advY;
  advZ  = invdxGPU * ((vz3+vz)*(bx5+bx) - (vz+vz2)*(bxmxpz+bx2));
  advZ += invdyGPU * ((vz4+vz)*(by5+by) - (vz+vz1)*(bymypz+by1));
  advZ += invdzGPU * ((vz5+vz)*(bz5+bz) - (vz+vz0)*(bz    +bz0));
  // advZ  = 0.25 * dtGPU * advZ;
  
  // Advection contribution to b eq. (-\bv \cdot \bna \bb = -\bna \cdot (bv) )
  advX -= invdxGPU * ((bx3+bx)*(vx3+vx) - (bx+bx2)*(vx    +vx2));
  advX -= invdyGPU * ((bx4+bx)*(vy3+vy) - (bx+bx1)*(vypxmy+vy1));
  advX -= invdzGPU * ((bx5+bx)*(vz3+vz) - (bx+bx0)*(vzpxmz+vz0));
  advX  = 0.25 * dtGPU * advX;
  advY -= invdxGPU * ((by3+by)*(vx4+vx) - (by+by2)*(vxmxpy+vx2));
  advY -= invdyGPU * ((by4+by)*(vy4+vy) - (by+by1)*(vy    +vy1));
  advY -= invdzGPU * ((by5+by)*(vz4+vz) - (by+by0)*(vzpymz+vz0));
  advY  = 0.25 * dtGPU * advY;
  advZ -= invdxGPU * ((bz3+bz)*(vx5+vx) - (bz+bz2)*(vxmxpz+vx2));
  advZ -= invdyGPU * ((bz4+bz)*(vy5+vy) - (bz+bz1)*(vymypz+vy1));
  advZ -= invdzGPU * ((bz5+bz)*(vz5+vz) - (bz+bz0)*(vz    +vz0));
  advZ  = 0.25 * dtGPU * advZ;
  
  // advX=0; advY=0; advZ=0;
  Bx += advX;
  By += advY;
  Bz += advZ;

  vxZ[i].x = Vx;
  vyZ[i].x = Vy;
  vzZ[i].x = Vz;
  vxZ[i].y = 0;
  vyZ[i].y = 0;
  vzZ[i].y = 0;

  WxZ[i].x = Bx;
  WyZ[i].x = By;
  WzZ[i].x = Bz;
  WxZ[i].y = 0;
  WyZ[i].y = 0;
  WzZ[i].y = 0;

}
