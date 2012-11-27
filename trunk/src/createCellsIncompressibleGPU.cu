// Filename: createCellsIncompressibleGPU.cu
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


bool createCellsIncompressibleGPU(){
  cudaMemcpyToSymbol(mxGPU,&mx,sizeof(int));
  cudaMemcpyToSymbol(myGPU,&my,sizeof(int));
  cudaMemcpyToSymbol(mzGPU,&mz,sizeof(int));

  cudaMemcpyToSymbol(mxtGPU,&mxt,sizeof(int));
  cudaMemcpyToSymbol(mytGPU,&myt,sizeof(int));
  cudaMemcpyToSymbol(mztGPU,&mzt,sizeof(int));



  cudaMemcpyToSymbol(ncellsGPU,&ncells,sizeof(int));
  cudaMemcpyToSymbol(ncellstGPU,&ncellst,sizeof(int));
  cudaMemcpyToSymbol(lxGPU,&lx,sizeof(double));
  cudaMemcpyToSymbol(lyGPU,&ly,sizeof(double));
  cudaMemcpyToSymbol(lzGPU,&lz,sizeof(double));
  cudaMemcpyToSymbol(dtGPU,&dt,sizeof(double));
  cudaMemcpyToSymbol(volumeGPU,&cVolume,sizeof(double));
  cudaMemcpyToSymbol(shearviscosityGPU,&shearviscosity,sizeof(double));
  cudaMemcpyToSymbol(temperatureGPU,&temperature,sizeof(double));
  cudaMemcpyToSymbol(thermostatGPU,&thermostat,sizeof(bool));

  cudaMemcpyToSymbol(densfluidGPU,&densfluid,sizeof(double));

  cudaMalloc((void**)&vxGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vyGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vzGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vxPredictionGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vyPredictionGPU,ncells*sizeof(double));
  cudaMalloc((void**)&vzPredictionGPU,ncells*sizeof(double));

 
  cudaMalloc((void**)&rxcellGPU,ncells*sizeof(double));
  cudaMalloc((void**)&rycellGPU,ncells*sizeof(double));
  cudaMalloc((void**)&rzcellGPU,ncells*sizeof(double));

  //FACT1 DIFFERENT FOR INCOMPRESSIBLE
  double fact1 = sqrt((4.*temperature*shearviscosity*dt)/(cVolume*densfluid*densfluid));
  //FACT4 DIFFERENT FOR INCOMPRESSIBLE
  double fact4 = sqrt((2.*temperature*shearviscosity*dt)/(cVolume*densfluid*densfluid));
  double fact5 = sqrt(1./(dt*cVolume));

  cudaMemcpyToSymbol(fact1GPU,&fact1,sizeof(double));
  cudaMemcpyToSymbol(fact4GPU,&fact4,sizeof(double));
  cudaMemcpyToSymbol(fact5GPU,&fact5,sizeof(double));


  fact1 = lx/double(mx);
  double fact2 = ly/double(my);
  double fact3 = lz/double(mz);
  cudaMemcpyToSymbol(dxGPU,&fact1,sizeof(double));
  cudaMemcpyToSymbol(dyGPU,&fact2,sizeof(double));
  cudaMemcpyToSymbol(dzGPU,&fact3,sizeof(double));

  fact1 = double(mx)/lx;
  fact2 = double(my)/ly;
  fact3 = double(mz)/lz;
  cudaMemcpyToSymbol(invdxGPU,&fact1,sizeof(double));  
  cudaMemcpyToSymbol(invdyGPU,&fact2,sizeof(double));  
  cudaMemcpyToSymbol(invdzGPU,&fact3,sizeof(double));  
  fact1 = 1./dt;
  cudaMemcpyToSymbol(invdtGPU,&fact1,sizeof(double));
  fact1 = 1./lx;
  fact2 = 1./ly;
  fact3 = 1./lz;
  cudaMemcpyToSymbol(invlxGPU,&fact1,sizeof(double));  
  cudaMemcpyToSymbol(invlyGPU,&fact2,sizeof(double));  
  cudaMemcpyToSymbol(invlzGPU,&fact3,sizeof(double));

 
  bool auxbool = 0;
  cudaMemcpyToSymbol(setparticlesGPU,&auxbool,sizeof(bool));
  cudaMemcpyToSymbol(setboundaryGPU,&auxbool,sizeof(bool));


  long long auxulonglong = 0;
  cudaMalloc((void**)&stepGPU,sizeof(long long));
  cudaMemcpy(stepGPU,&auxulonglong,sizeof(long long),cudaMemcpyHostToDevice);


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


  //Factors for the update in fourier space
  cudaMalloc((void**)&gradKx,     mx*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&gradKy,     my*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&gradKz,     mz*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&expKx,      mx*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&expKy,      my*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&expKz,      mz*sizeof(cufftDoubleComplex));

  cudaMalloc((void**)&pF,sizeof(prefactorsFourier));

  //cudaMalloc((void**)&WxZ,ncells*sizeof(cufftDoubleComplex));
  //cudaMalloc((void**)&WyZ,ncells*sizeof(cufftDoubleComplex));
  //cudaMalloc((void**)&WzZ,ncells*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&vxZ,ncells*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&vyZ,ncells*sizeof(cufftDoubleComplex));
  cudaMalloc((void**)&vzZ,ncells*sizeof(cufftDoubleComplex)); 

  if(quasiNeutrallyBuoyant){
    cudaMalloc((void**)&advXGPU,ncells*sizeof(double));
    cudaMalloc((void**)&advYGPU,ncells*sizeof(double));
    cudaMalloc((void**)&advZGPU,ncells*sizeof(double));
  }

  cout << "CREATE CELLS GPU :              DONE" << endl;

  return 1;
}
