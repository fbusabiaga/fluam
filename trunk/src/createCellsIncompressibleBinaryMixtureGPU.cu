// Filename: createCellsIncompressibleBinaryMixtureGPU.cu
//
// Copyright (c) 2010-2012, Florencio Balboa Usabiaga
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


bool createCellsIncompressibleBinaryMixtureGPU(){
  //Number of cells to constant memory
  cudaMemcpyToSymbol(mxGPU,&mx,sizeof(int));
  cudaMemcpyToSymbol(myGPU,&my,sizeof(int));
  cudaMemcpyToSymbol(mzGPU,&mz,sizeof(int));
  cudaMemcpyToSymbol(ncellsGPU,&ncells,sizeof(int));

  cudaMemcpyToSymbol(mxtGPU,&mxt,sizeof(int));
  cudaMemcpyToSymbol(mytGPU,&myt,sizeof(int));
  cudaMemcpyToSymbol(mztGPU,&mzt,sizeof(int));
  cudaMemcpyToSymbol(ncellstGPU,&ncellst,sizeof(int));

  //Simulation box size to constant memory
  cudaMemcpyToSymbol(lxGPU,&lx,sizeof(double));
  cudaMemcpyToSymbol(lyGPU,&ly,sizeof(double));
  cudaMemcpyToSymbol(lzGPU,&lz,sizeof(double));

  //Time step to constant memory
  cudaMemcpyToSymbol(dtGPU,&dt,sizeof(double));

  //Volume cell to constant memory
  cudaMemcpyToSymbol(volumeGPU,&cVolume,sizeof(double));

  //Viscosity and temperature to constant memory
  cudaMemcpyToSymbol(shearviscosityGPU,&shearviscosity,sizeof(double));
  cudaMemcpyToSymbol(temperatureGPU,&temperature,sizeof(double));
  cudaMemcpyToSymbol(thermostatGPU,&thermostat,sizeof(bool));

  //Mass diffusion coefficient to constant memory
  cudaMemcpyToSymbol(diffusionGPU,&diffusion,sizeof(double));
  cudaMemcpyToSymbol(massSpecies0GPU,&massSpecies0,sizeof(double));
  cudaMemcpyToSymbol(massSpecies1GPU,&massSpecies1,sizeof(double));

  //Fluid density to constant memory
  cudaMemcpyToSymbol(densfluidGPU,&densfluid,sizeof(double));

  double fact1, fact4, fact5;
  //FACT1 DIFFERENT FOR INCOMPRESSIBLE
  fact1 = sqrt((4.*temperature*shearviscosity*dt)/(cVolume*densfluid*densfluid));
  //FACT4 DIFFERENT FOR INCOMPRESSIBLE
  fact4 = sqrt((2.*temperature*shearviscosity*dt)/(cVolume*densfluid*densfluid));
  fact5 = sqrt(1./(dt*cVolume));
  cudaMemcpyToSymbol(gradTemperatureGPU,&gradTemperature,sizeof(double));
  
  //Prefactor for stochastic force to constant memory
  cudaMemcpyToSymbol(fact1GPU,&fact1,sizeof(double));
  cudaMemcpyToSymbol(fact4GPU,&fact4,sizeof(double));
  cudaMemcpyToSymbol(fact5GPU,&fact5,sizeof(double));

  //Cell size to constant memory
  fact1 = lx/double(mx);
  double fact2 = ly/double(my);
  double fact3 = lz/double(mz);
  cudaMemcpyToSymbol(dxGPU,&fact1,sizeof(double));
  cudaMemcpyToSymbol(dyGPU,&fact2,sizeof(double));
  cudaMemcpyToSymbol(dzGPU,&fact3,sizeof(double));

  //Inverse cell size to cosntant memory
  fact1 = double(mx)/lx;
  fact2 = double(my)/ly;
  fact3 = double(mz)/lz;
  cudaMemcpyToSymbol(invdxGPU,&fact1,sizeof(double));  
  cudaMemcpyToSymbol(invdyGPU,&fact2,sizeof(double));  
  cudaMemcpyToSymbol(invdzGPU,&fact3,sizeof(double));  

  //Inverse time step to constant memory
  fact1 = 1./dt;
  cudaMemcpyToSymbol(invdtGPU,&fact1,sizeof(double));

  //Inverse box size to constant memory
  fact1 = 1./lx;
  fact2 = 1./ly;
  fact3 = 1./lz;
  cudaMemcpyToSymbol(invlxGPU,&fact1,sizeof(double));  
  cudaMemcpyToSymbol(invlyGPU,&fact2,sizeof(double));  
  cudaMemcpyToSymbol(invlzGPU,&fact3,sizeof(double));

  //Some options to constant memory
  bool auxbool = 0;
  cudaMemcpyToSymbol(setparticlesGPU,&auxbool,sizeof(bool));
  cudaMemcpyToSymbol(setboundaryGPU,&auxbool,sizeof(bool));
  








  //Step to global memory
  long long auxulonglong = 0;
  cudaMalloc((void**)&stepGPU,sizeof(long long));
  cudaMemcpy(stepGPU,&auxulonglong,sizeof(long long),cudaMemcpyHostToDevice);

  //Fluid velocity and velocity prediction to
  //global memory
  cudaMalloc((void**)&vxGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vyGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vzGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vxPredictionGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vyPredictionGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vzPredictionGPU,ncells*sizeof(double));

  //Concentration to global memory
  cudaMalloc((void**)&cGPU,ncells*sizeof(double));
  cudaMalloc((void**)&cPredictionGPU,ncells*sizeof(double));

  //Centers cells to global memory
  cudaMalloc((void**)&rxcellGPU,ncells*sizeof(double));
  cudaMalloc((void**)&rycellGPU,ncells*sizeof(double));
  cudaMalloc((void**)&rzcellGPU,ncells*sizeof(double));

  //List of neighbors cells to global memory
  cudaMalloc((void**)&vecino0GPU,ncells*sizeof(int));
  cudaMalloc((void**)&vecino1GPU,ncells*sizeof(int));
  cudaMalloc((void**)&vecino2GPU,ncells*sizeof(int));
  cudaMalloc((void**)&vecino3GPU,ncells*sizeof(int));
  cudaMalloc((void**)&vecino4GPU,ncells*sizeof(int));
  cudaMalloc((void**)&vecino5GPU,ncells*sizeof(int));
  cudaMalloc((void**)&vecinopxpyGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinopxmyGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinopxpzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinopxmzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinomxpyGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinomxmyGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinomxpzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinomxmzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinopypzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinopymzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinomypzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinomymzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinopxpypzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinopxpymzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinopxmypzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinopxmymzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinomxpypzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinomxpymzGPU,ncells*sizeof(int));
  cudaMalloc((void**)&vecinomxmypzGPU,ncells*sizeof(int)); 
  cudaMalloc((void**)&vecinomxmymzGPU,ncells*sizeof(int)); 

  //Factors for the update in fourier space to global memory
  cudaMalloc((void**)&gradKx,     mx*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&gradKy,     my*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&gradKz,     mz*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&expKx,      mx*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&expKy,      my*sizeof(cufftDoubleComplex));  
  cudaMalloc((void**)&expKz,      mz*sizeof(cufftDoubleComplex));

  cudaMalloc((void**)&pF,sizeof(prefactorsFourier));

  //Complex velocity field to global memory
  cudaMalloc((void**)&vxZ,ncells*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&vyZ,ncells*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&vzZ,ncells*sizeof(cufftDoubleComplex)); 

  //Complex concentration to global memory
  cudaMalloc((void**)&cZ,ncells*sizeof(cufftDoubleComplex));


  cout << "CREATE CELLS GPU :              DONE" << endl;

  return 1;
}
