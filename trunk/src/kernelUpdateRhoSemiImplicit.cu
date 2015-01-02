// Filename: kernelUpdateRhoSemiImplicit.cu
//
// Copyright (c) 2010-2015, Florencio Balboa Usabiaga
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



__global__ void kernelUpdateRhoSemiImplicit(cufftDoubleComplex *vxZ,
					    const prefactorsFourier *pF,
					    const double omega1){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double omega1 = 1.70710678118654752;
  //double omega1 = 0.292893218813452476;

  //Find mode
  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  //Construct L
  double L;
  L = -((pF->gradKx[kx].y) * (pF->gradKx[kx].y)) - 
    ((pF->gradKy[ky].y) * (pF->gradKy[ky].y)) -
    ((pF->gradKz[kz].y) * (pF->gradKz[kz].y));

  //Construct denominator
  //double denominator = 1 - pow((omega1 * dtGPU),2) * pressurea1GPU * L ;
  double denominator = 1 - (omega1*dtGPU)*(omega1*dtGPU) * pressurea1GPU * L;

  vxZ[i].x = vxZ[i].x / denominator;
  vxZ[i].y = vxZ[i].y / denominator;
  
  
  
}








__global__ void kernelUpdateRhoSemiImplicit_2(cufftDoubleComplex *vxZ,
					      const prefactorsFourier *pF,
					      const double omega4){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  //double omega4 = 1.70710678118654752;
  //double omega4 = 0.292893218813452476;

  //Find mode
  int kx, ky, kz;
  kz = i / (mxGPU*myGPU);
  ky = (i % (mxGPU*myGPU)) / mxGPU;
  kx = i % mxGPU;
  
  //Construct L
  double L;
  L = -((pF->gradKx[kx].y) * (pF->gradKx[kx].y)) - 
    ((pF->gradKy[ky].y) * (pF->gradKy[ky].y)) -
    ((pF->gradKz[kz].y) * (pF->gradKz[kz].y));

  //Construct denominator
  //double denominator = 1 - pow((omega4*dtGPU),2) * pressurea1GPU * L;
  double denominator = 1 - (omega4*dtGPU)*(omega4*dtGPU) * pressurea1GPU * L;

  vxZ[i].x = vxZ[i].x / denominator;
  vxZ[i].y = vxZ[i].y / denominator;

  
}

