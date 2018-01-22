// Filename: createCellsIncompressibleGPU.cu
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


#define GPUVARIABLES 1


bool createCellsQuasi2DGPU(){
  //Raul added. Upload saffman variables to gpu
  cutilSafeCall(cudaMemcpyToSymbol(saffmanCutOffWaveNumberGPU,&saffmanCutOffWaveNumber, sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(saffmanLayerWidthGPU,&saffmanLayerWidth, sizeof(double)));

  cutilSafeCall(cudaMemcpyToSymbol(mxGPU,&mx,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(myGPU,&my,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mzGPU,&mz,sizeof(int)));

  cutilSafeCall(cudaMemcpyToSymbol(mxtGPU,&mxt,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mytGPU,&myt,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(mztGPU,&mzt,sizeof(int)));

  cutilSafeCall(cudaMemcpyToSymbol(ncellsGPU,&ncells,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(ncellstGPU,&ncellst,sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(lxGPU,&lx,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(lyGPU,&ly,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(lzGPU,&lz,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(dtGPU,&dt,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(volumeGPU,&cVolume,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(shearviscosityGPU,&shearviscosity,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(temperatureGPU,&temperature,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(thermostatGPU,&thermostat,sizeof(bool)));

  cutilSafeCall(cudaMemcpyToSymbol(densfluidGPU,&densfluid,sizeof(double)));

  // Radius and kernel
  double GaussianVariance;
  // For 3D
  // double GaussianVariance = pow(hydrodynamicRadius / (1.0 * sqrt(3.1415926535897932385)), 2);
  // For 2D
  if(stokesLimit2D){
    GaussianVariance = pow(hydrodynamicRadius * 0.66556976637237890625, 2);
  }
  // For quasi-2D disks
  // double GaussianVariance = pow(hydrodynamicRadius * 9.0*sqrt(3.1415926535897932385)/16.0, 2);
  // For quasi-2D spheres
  if(quasi2D){
    GaussianVariance = pow(hydrodynamicRadius / sqrt(3.1415926535897932385), 2);
  }
  
  int kernelWidth = int(3.0 * hydrodynamicRadius * mx / lx) + 1;
  cutilSafeCall(cudaMemcpyToSymbol(GaussianVarianceGPU,&GaussianVariance,sizeof(double)));
  if (kernelWidth > mx/2){
    kernelWidth = mx/2;
  }
  cout << "kernelWidth = " << kernelWidth << endl;
  cout << "GaussianVariance = " << GaussianVariance << endl;
  cutilSafeCall(cudaMemcpyToSymbol(kernelWidthGPU,&kernelWidth,sizeof(int)));
  double deltaRFD = 1e-05 * hydrodynamicRadius;
  cutilSafeCall(cudaMemcpyToSymbol(deltaRFDGPU,&deltaRFD,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(nDriftGPU,&nDrift,sizeof(int)));

  cutilSafeCall(cudaMalloc((void**)&vxGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vyGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&vzGPU,ncells*sizeof(double)));
 
  cutilSafeCall(cudaMalloc((void**)&rxcellGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rycellGPU,ncells*sizeof(double)));
  cutilSafeCall(cudaMalloc((void**)&rzcellGPU,ncells*sizeof(double)));

  // FACT1 for quasi-2D or stokesLimit2D
  double fact1 = sqrt(1.0 * temperature  / (shearviscosity * dt * lx * ly)) * ncells;
  cutilSafeCall(cudaMemcpyToSymbol(fact1GPU,&fact1,sizeof(double)));

  fact1 = lx/double(mx);
  double fact2 = ly/double(my);
  double fact3 = lz/double(mz);
  cutilSafeCall(cudaMemcpyToSymbol(dxGPU,&fact1,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(dyGPU,&fact2,sizeof(double)));
  cutilSafeCall(cudaMemcpyToSymbol(dzGPU,&fact3,sizeof(double)));

  fact1 = double(mx)/lx;
  fact2 = double(my)/ly;
  fact3 = double(mz)/lz;
  cutilSafeCall(cudaMemcpyToSymbol(invdxGPU,&fact1,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invdyGPU,&fact2,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invdzGPU,&fact3,sizeof(double)));  
  fact1 = 1./dt;
  cutilSafeCall(cudaMemcpyToSymbol(invdtGPU,&fact1,sizeof(double)));
  fact1 = 1./lx;
  fact2 = 1./ly;
  fact3 = 1./lz;
  cutilSafeCall(cudaMemcpyToSymbol(invlxGPU,&fact1,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invlyGPU,&fact2,sizeof(double)));  
  cutilSafeCall(cudaMemcpyToSymbol(invlzGPU,&fact3,sizeof(double)));

 
  bool auxbool = 0;
  cutilSafeCall(cudaMemcpyToSymbol(setparticlesGPU,&auxbool,sizeof(bool)));
  cutilSafeCall(cudaMemcpyToSymbol(setboundaryGPU,&auxbool,sizeof(bool)));

  long long auxulonglong = 0;
  cutilSafeCall(cudaMalloc((void**)&stepGPU,sizeof(long long)));
  cutilSafeCall(cudaMemcpy(stepGPU,&auxulonglong,sizeof(long long),cudaMemcpyHostToDevice));

  //Factors for the update in fourier space
  cutilSafeCall(cudaMalloc((void**)&gradKx,     mx*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&gradKy,     my*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&gradKz,     mz*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKx,      mx*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKy,      my*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&expKz,      mz*sizeof(cufftDoubleComplex)));

  cutilSafeCall(cudaMalloc((void**)&pF,sizeof(prefactorsFourier)));

  cutilSafeCall(cudaMalloc((void**)&vxZ,ncells*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&vyZ,ncells*sizeof(cufftDoubleComplex)));
  cutilSafeCall(cudaMalloc((void**)&vzZ,ncells*sizeof(cufftDoubleComplex))); 

  cout << "CREATE CELLS GPU :              DONE" << endl;

  return 1;
}
