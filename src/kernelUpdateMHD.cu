// Filename: kernelUpdateMHD.cu
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


__global__ void kernelUpdateMHD(cufftDoubleComplex *vxZ, 
				cufftDoubleComplex *vyZ,
				cufftDoubleComplex *vzZ, 
				cufftDoubleComplex *WxZ, 
				cufftDoubleComplex *WyZ,
				cufftDoubleComplex *WzZ, 
				prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

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

  //Construct GG
  double GG;
  GG = L;

  //Construct denominator
  double denominator = 1 - 0.5 * dtGPU * shearviscosityGPU * L / densfluidGPU;
  double denominatorB = 1 - 0.5 * dtGPU * diffusionGPU * L ;

  //Construct GW
  cufftDoubleComplex GW, GB;
  GW.x = pF->gradKx[kx].y * vxZ[i].x + pF->gradKy[ky].y * vyZ[i].x + pF->gradKz[kz].y * vzZ[i].x;
  GW.y = pF->gradKx[kx].y * vxZ[i].y + pF->gradKy[ky].y * vyZ[i].y + pF->gradKz[kz].y * vzZ[i].y;

  GB.x = pF->gradKx[kx].y * WxZ[i].x + pF->gradKy[ky].y * WyZ[i].x + pF->gradKz[kz].y * WzZ[i].x;
  GB.y = pF->gradKx[kx].y * WxZ[i].y + pF->gradKy[ky].y * WyZ[i].y + pF->gradKz[kz].y * WzZ[i].y;
  
  if(i==0){
    //vxZ[i].x = WxZ[i].x;
    //vxZ[i].y = WxZ[i].y;
    //vyZ[i].x = WyZ[i].x;
    //vyZ[i].y = WyZ[i].y;
    //vzZ[i].x = WzZ[i].x;
    //vzZ[i].y = WzZ[i].y;
  }
  else{
    vxZ[i].x = (vxZ[i].x + pF->gradKx[kx].y * GW.x / GG) / denominator;
    vxZ[i].y = (vxZ[i].y + pF->gradKx[kx].y * GW.y / GG) / denominator;
    vyZ[i].x = (vyZ[i].x + pF->gradKy[ky].y * GW.x / GG) / denominator;
    vyZ[i].y = (vyZ[i].y + pF->gradKy[ky].y * GW.y / GG) / denominator;
    vzZ[i].x = (vzZ[i].x + pF->gradKz[kz].y * GW.x / GG) / denominator;
    vzZ[i].y = (vzZ[i].y + pF->gradKz[kz].y * GW.y / GG) / denominator;

    WxZ[i].x = (WxZ[i].x + pF->gradKx[kx].y * GB.x / GG) / denominatorB;
    WxZ[i].y = (WxZ[i].y + pF->gradKx[kx].y * GB.y / GG) / denominatorB;
    WyZ[i].x = (WyZ[i].x + pF->gradKy[ky].y * GB.x / GG) / denominatorB;
    WyZ[i].y = (WyZ[i].y + pF->gradKy[ky].y * GB.y / GG) / denominatorB;
    WzZ[i].x = (WzZ[i].x + pF->gradKz[kz].y * GB.x / GG) / denominatorB;
    WzZ[i].y = (WzZ[i].y + pF->gradKz[kz].y * GB.y / GG) / denominatorB;

    // WxZ[i].x = WxZ[i].x / denominatorB;
    // WxZ[i].y = WxZ[i].y / denominatorB;   
    // WyZ[i].x = WyZ[i].x / denominatorB;
    // WyZ[i].y = WyZ[i].y / denominatorB;   
    // WzZ[i].x = WzZ[i].x / denominatorB;
    // WzZ[i].y = WzZ[i].y / denominatorB;   
  }
  
}















__global__ void filterTwoThirds(cufftDoubleComplex *vxZ,
				cufftDoubleComplex *vyZ,
				cufftDoubleComplex *vzZ,
				prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double pi = 4. * atan(1.);

  // Find mode
  int nx, ny, nz;
  nz = i / (mxGPU*myGPU);
  ny = (i % (mxGPU*myGPU)) / mxGPU;
  nx = i % mxGPU;

  // Find mode in intervale (-pi/L, pi/L)
  double kx = pF->gradKx[nx].y;
  double ky = pF->gradKx[ny].y;
  double kz = pF->gradKx[nz].y;
  double k = kx*kx + ky*ky + kz*kz;  

  // Set k_max to the standard 2/3 rule
  double k_x_max = (2.0/3.0) * pi * invdxGPU; // right value is 2*pi/3.0 * invdxGPU
  double k_y_max = (2.0/3.0) * pi * invdyGPU;
  double k_z_max = (2.0/3.0) * pi * invdzGPU;

  // For k>m_max set modes to zero 
  if((abs(kx) >= k_x_max) || (abs(ky) >= k_y_max) || (abs(kz) >= k_z_max) || (k >= k_x_max*k_y_max)){
    vxZ[i].x = 0;
    vxZ[i].y = 0;
    vyZ[i].x = 0;
    vyZ[i].y = 0;
    vzZ[i].x = 0;
    vzZ[i].y = 0;
  }

  return;
}




__global__ void filterExponential(cufftDoubleComplex *vxZ,
				  cufftDoubleComplex *vyZ,
				  cufftDoubleComplex *vzZ,
				  prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  double pi = 4. * atan(1.);

  // Find mode
  int nx, ny, nz;
  nz = i / (mxGPU*myGPU);
  ny = (i % (mxGPU*myGPU)) / mxGPU;
  nx = i % mxGPU;

  // Find mode in intervale (-pi/L, pi/L)
  double kx = pF->gradKx[nx].y;
  double ky = pF->gradKx[ny].y;
  double kz = pF->gradKx[nz].y;
  double k = kx*kx + ky*ky + kz*kz;  

  // Set k_max to the standard 2/3 rule
  double k_x_max = pi * invdxGPU; // right value is 2*pi/3.0 * invdxGPU
  double k_y_max = pi * invdyGPU;
  double k_z_max = pi * invdzGPU;

  double factor = exp(-36.0 * pow(k/(k_x_max*k_y_max), 36));

  // Scale all modes
  vxZ[i].x *= factor;
  vxZ[i].y *= factor;
  vyZ[i].x *= factor;
  vyZ[i].y *= factor;
  vzZ[i].x *= factor;
  vzZ[i].y *= factor;
  
  return;
}
