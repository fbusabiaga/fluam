// Filename: createCellsGPU.cu
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


bool createCellsGPU(){
  
  cudaMemcpyToSymbol(mxGPU,&mx,sizeof(int));
  cudaMemcpyToSymbol(myGPU,&my,sizeof(int));
  cudaMemcpyToSymbol(mzGPU,&mz,sizeof(int));

  cudaMemcpyToSymbol(mxtGPU,&mxt,sizeof(int));
  cudaMemcpyToSymbol(mytGPU,&myt,sizeof(int));
  cudaMemcpyToSymbol(mztGPU,&mzt,sizeof(int));

  int aux = (mxt) * (myt);
  cudaMemcpyToSymbol(mxmytGPU,&aux,sizeof(int));


  cudaMemcpyToSymbol(ncellsGPU,&ncells,sizeof(int));
  cudaMemcpyToSymbol(ncellstGPU,&ncellst,sizeof(int));
  cudaMemcpyToSymbol(lxGPU,&lx,sizeof(double));
  cudaMemcpyToSymbol(lyGPU,&ly,sizeof(double));
  cudaMemcpyToSymbol(lzGPU,&lz,sizeof(double));
  cudaMemcpyToSymbol(dtGPU,&dt,sizeof(double));
  cudaMemcpyToSymbol(volumeGPU,&cVolume,sizeof(double));
  cudaMemcpyToSymbol(shearviscosityGPU,&shearviscosity,sizeof(double));
  cudaMemcpyToSymbol(bulkviscosityGPU,&bulkviscosity,sizeof(double));
  cudaMemcpyToSymbol(temperatureGPU,&temperature,sizeof(double));
  cudaMemcpyToSymbol(pressurea0GPU,&pressurea0,sizeof(double));
  cudaMemcpyToSymbol(pressurea1GPU,&pressurea1,sizeof(double));
  cudaMemcpyToSymbol(pressurea2GPU,&pressurea2,sizeof(double));
  cudaMemcpyToSymbol(thermostatGPU,&thermostat,sizeof(bool));

  cudaMemcpyToSymbol(densfluidGPU,&densfluid,sizeof(double));

  cudaMalloc((void**)&densityGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vxGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vyGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vzGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&densityPredictionGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vxPredictionGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vyPredictionGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vzPredictionGPU,ncellst*sizeof(double));

 
  cudaMalloc((void**)&dmGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&dpxGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&dpyGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&dpzGPU,ncellst*sizeof(double));

  cudaMalloc((void**)&rxcellGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&rycellGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&rzcellGPU,ncellst*sizeof(double));

  double fact1 = sqrt((4.*temperature*shearviscosity)/(dt*cVolume));
  double fact2 = sqrt((2.*temperature*bulkviscosity)/(3.*dt*cVolume));
  double fact3 = bulkviscosity - 2. * shearviscosity/3.;
  double fact4 = sqrt((2.*temperature*shearviscosity)/(dt*cVolume));
  double fact5 = sqrt(1./(dt*cVolume));

  cudaMemcpyToSymbol(fact1GPU,&fact1,sizeof(double));
  cudaMemcpyToSymbol(fact2GPU,&fact2,sizeof(double));
  cudaMemcpyToSymbol(fact3GPU,&fact3,sizeof(double));
  cudaMemcpyToSymbol(fact4GPU,&fact4,sizeof(double));
  cudaMemcpyToSymbol(fact5GPU,&fact5,sizeof(double));


  fact1 = lx/double(mx);
  fact2 = ly/double(my);
  fact3 = lz/double(mz);
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


  cudaMalloc((void**)&vecino0GPU,ncellst*sizeof(int));
  cudaMalloc((void**)&vecino1GPU,ncellst*sizeof(int));
  cudaMalloc((void**)&vecino2GPU,ncellst*sizeof(int));
  cudaMalloc((void**)&vecino3GPU,ncellst*sizeof(int));
  cudaMalloc((void**)&vecino4GPU,ncellst*sizeof(int));
  cudaMalloc((void**)&vecino5GPU,ncellst*sizeof(int));
  cudaMalloc((void**)&vecinopxpyGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinopxmyGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinopxpzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinopxmzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinomxpyGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinomxmyGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinomxpzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinomxmzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinopypzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinopymzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinomypzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinomymzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinopxpypzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinopxpymzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinopxmypzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinopxmymzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinomxpypzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinomxpymzGPU,ncellst*sizeof(int));
  cudaMalloc((void**)&vecinomxmypzGPU,ncellst*sizeof(int)); 
  cudaMalloc((void**)&vecinomxmymzGPU,ncellst*sizeof(int)); 



  if(particlesWall){
    cudaMalloc((void**)&ghostIndexGPU,ncells*sizeof(int));
    cudaMalloc((void**)&realIndexGPU,ncellst*sizeof(int));
    cudaMalloc((void**)&ghostToPIGPU,(ncellst-ncells)*sizeof(int));
    cudaMalloc((void**)&ghostToGhostGPU,(ncellst-ncells)*sizeof(int));

    fact1 = ly + 2*ly/double(my);
    fact2 = 1. / (ly + 2*ly/double(my));
    cudaMemcpyToSymbol(lyGPU,&fact1,sizeof(double)); 
    cudaMemcpyToSymbol(invlyGPU,&fact2,sizeof(double));   
  }
  

  
  cout << "CREATE CELLS GPU :              DONE" << endl;

  return 1;
}
