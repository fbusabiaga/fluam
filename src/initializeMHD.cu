// Filename: initializeBeltramiFlow.cu
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




__global__ void initializeBeltramiFlow(cufftDoubleComplex *vxZ,
				       cufftDoubleComplex *vyZ,
				       cufftDoubleComplex *vzZ,
				       prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   
  
  double pi=4*atan(1.0);

  // Define lambda
  double lambda = 2*pi;
  
  //Find mode
  int nx, ny, nz;
  nz = i / (mxGPU*myGPU);
  ny = (i % (mxGPU*myGPU)) / mxGPU;
  nx = i % mxGPU;

  double invsqrt2 = 1.0 / sqrt(2.0);
  double a=1e+05;

  double kx0=2*pi*invlxGPU;
  double ky0=2*pi*invlyGPU;
  double kz0= 2*pi*invlzGPU;
  double kx = kx0 * nx;
  double ky = ky0 * ny;
  double kz = kz0 * nz;
  
  double k = sqrt( kx*kx + ky*ky + kz*kz );
  double kproj = sqrt( k*k - kz*kz );
  
  double v2x, v2y, v2z; //First solenoidal mode
  double v3x, v3y, v3z; //Second solenoidal mode

  if(fabs(k-lambda)<=1e-6){
    if(nx==0 && ny==0){//kx=0 and ky=0 is a singular limit
      v2x=1; v2y=0; v2z=0;
      v3x=0; v3y=1; v3z=0;      
    }
    else{
      v2x=  kx*kz / (k*kproj);
      v2y=  ky*kz / (k*kproj);
      v2z= -kproj / k;
      v3x= -ky / kproj;
      v3y=  kx / kproj;
      v3z=  0;
    }
    vxZ[i].x =  v2x * invsqrt2 * a; //Real part
    vyZ[i].x =  v2y * invsqrt2 * a;
    vzZ[i].x =  v2z * invsqrt2 * a;
    vxZ[i].y =  v3x * invsqrt2 * a; //Imaginary part
    vyZ[i].y =  v3y * invsqrt2 * a;
    vzZ[i].y =  v3z * invsqrt2 * a;
  }
  else{
    vxZ[i].x = 0;
    vyZ[i].x = 0;
    vzZ[i].x = 0;
    vxZ[i].y = 0;
    vyZ[i].y = 0;
    vzZ[i].y = 0;
  }
  
  
}









__global__ void initializeDiscreteBeltramiFlow(cufftDoubleComplex *vxZ,
					       cufftDoubleComplex *vyZ,
					       cufftDoubleComplex *vzZ,
					       prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   
  
  double pi=4*atan(1.0);//3.141592654;

  //Find mode
  int nx, ny, nz;
  nz = i / (mxGPU*myGPU);
  ny = (i % (mxGPU*myGPU)) / mxGPU;
  nx = i % mxGPU;

  double invsqrt2 = 1.0 / sqrt(2.0);
  double a=ncellsGPU * invsqrt2;

  double kx0=2*pi*invlxGPU;
  double ky0=2*pi*invlyGPU;
  double kz0=2*pi*invlzGPU;
  //double kx = (2/dxGPU) * sin(0.5*kx0*dxGPU*nx);
  //double ky = (2/dyGPU) * sin(0.5*ky0*dyGPU*ny);
  //double kz = (2/dzGPU) * sin(0.5*kz0*dzGPU*nz);
  double kx = (2/dxGPU) * sin(pi*nx/mxGPU);
  double ky = (2/dyGPU) * sin(pi*ny/myGPU);
  double kz = (2/dzGPU) * sin(pi*nz/mzGPU);
  
  double k = sqrt( kx*kx + ky*ky + kz*kz );
  double kproj = sqrt( kx*kx + ky*ky );
  double kContinuoum=sqrt(kx0*kx0*nx*nx + ky0*ky0*ny*ny + kz0*kz0*nz*nz);
  
  double v2x, v2y, v2z; //First solenoidal mode
  double v3x, v3y, v3z; //Second solenoidal mode


  //if(abs(kContinuoum-lambdaGPU)<=1e-6){
  if(abs(kContinuoum-1)<=1e-6 || abs(kContinuoum-1.41421356237)<=1e-6){
    if(nx==0 && ny==0){//kx=0 and ky=0 is a singular limit
      v2x=1; v2y=0; v2z=0;
      v3x=0; v3y=1; v3z=0;      
    }
    else{
      v2x=  kx*kz / (k*kproj);
      v2y=  ky*kz / (k*kproj);
      v2z= -kproj / k;
      v3x= -ky / kproj;
      v3y=  kx / kproj;
      v3z=  0;
    }
    vxZ[i].x =  v2x * invsqrt2 * a; //Real part
    vyZ[i].x =  v2y * invsqrt2 * a;
    vzZ[i].x =  v2z * invsqrt2 * a;
    vxZ[i].y =  v3x * invsqrt2 * a; //Imaginary part
    vyZ[i].y =  v3y * invsqrt2 * a;
    vzZ[i].y =  v3z * invsqrt2 * a;
  }
  else{
    vxZ[i].x = 0;
    vyZ[i].x = 0;
    vzZ[i].x = 0;
    vxZ[i].y = 0;
    vyZ[i].y = 0;
    vzZ[i].y = 0;
  }
  
}



__global__ void initializeMHD(cufftDoubleComplex *vxZ,
			      cufftDoubleComplex *vyZ,
			      cufftDoubleComplex *vzZ,
			      cufftDoubleComplex *WxZ,
			      cufftDoubleComplex *WyZ,
			      cufftDoubleComplex *WzZ,
			      prefactorsFourier *pF){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   
  
  double pi=4*atan(1.0);

  // Define lambda
  // double lambda = 2*pi;
  
  //Find mode
  int nx, ny, nz;
  nz = i / (mxGPU*myGPU);
  ny = (i % (mxGPU*myGPU)) / mxGPU;
  nx = i % mxGPU;

  double invsqrt2 = 1.0 / sqrt(2.0);
  double a=1e+05;

  double kx0=2*pi*invlxGPU;
  double ky0=2*pi*invlyGPU;
  double kz0=0;;
  double kx = kx0 * nx;
  double ky = ky0 * ny;
  double kz = kz0 * nz;
  
  double k = sqrt( kx*kx + ky*ky + kz*kz );
  double kproj = sqrt( k*k - kz*kz );
  
  double v2x, v2y; //First solenoidal mode
  double v3x, v3y; //Second solenoidal mode

  if(fabs(k-pressurea1GPU)<=1e-6){
    if(nx==0 && ny==0){//kx=0 and ky=0 is a singular limit
      v2x=1; v2y=0;
      v3x=0; v3y=1;
    }
    else{
      v2x=  kx*kz / (k*kproj);
      v2y=  ky*kz / (k*kproj);
      // v2z=  0;
      v3x= -ky / kproj;
      v3y=  kx / kproj;
      // v3z=  0;
    }
    // Velocity
    vxZ[i].x =  0;//v2x * invsqrt2 * a; //Real part
    vyZ[i].x =  0;//v2y * invsqrt2 * a;
    vzZ[i].x =  0;//v2z * invsqrt2 * a;
    vxZ[i].y =  0;//v3x * invsqrt2 * a; //Imaginary part
    vyZ[i].y =  0;//v3y * invsqrt2 * a;
    vzZ[i].y =  0;//v3z * invsqrt2 * a;
    // b
    WxZ[i].x =  0; //v2x * invsqrt2 * a; //Real part
    WyZ[i].x =  0; //v2y * invsqrt2 * a;
    WzZ[i].x =  0; //v2z * invsqrt2 * a;
    WxZ[i].y =  0; //v3x * invsqrt2 * a; //Imaginary part
    WyZ[i].y =  0; //v3y * invsqrt2 * a;
    WzZ[i].y =  0; //v3z * invsqrt2 * a;
  }
  else{
    vxZ[i].x = 0;
    vyZ[i].x = 0;
    vzZ[i].x = 0;
    vxZ[i].y = 0;
    vyZ[i].y = 0;
    vzZ[i].y = 0;
    WxZ[i].x = 0;
    WyZ[i].x = 0;
    WzZ[i].x = 0;
    WxZ[i].y = 0;
    WyZ[i].y = 0;
    WzZ[i].y = 0;
  }
  
  
}
