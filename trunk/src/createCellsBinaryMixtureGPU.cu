// Filename: createCellsBinaryMixtureGPU.cu
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


//#define GPUVARIABLES 1


bool createCellsBinaryMixtureGPU(){
  cudaMemcpyToSymbol(mxGPU,&mx,sizeof(int));
  cudaMemcpyToSymbol(myGPU,&my,sizeof(int));
  cudaMemcpyToSymbol(mzGPU,&mz,sizeof(int));
  int aux;
  aux = mx+2;
  cudaMemcpyToSymbol(mxtGPU,&aux,sizeof(int));
  aux = my+2;
  cudaMemcpyToSymbol(mytGPU,&aux,sizeof(int));
  aux = mz+2;
  cudaMemcpyToSymbol(mztGPU,&aux,sizeof(int));
  aux = (mx+2) * (my+2);
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

  cudaMalloc((void**)&densityGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vxGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vyGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vzGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&densityPredictionGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vxPredictionGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vyPredictionGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&vzPredictionGPU,ncellst*sizeof(double));

  cudaMalloc((void**)&cGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&cPredictionGPU,ncellst*sizeof(double));
  cudaMalloc((void**)&dcGPU,ncellst*sizeof(double));

  cudaMemcpyToSymbol(diffusionGPU,&diffusion,sizeof(double));
  cudaMemcpyToSymbol(massSpecies0GPU,&massSpecies0,sizeof(double));
  cudaMemcpyToSymbol(massSpecies1GPU,&massSpecies1,sizeof(double));


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

  //
  cudaMalloc((void**)&ghostIndexGPU,ncells*sizeof(int));
  cudaMalloc((void**)&realIndexGPU,ncellst*sizeof(int));
  cudaMalloc((void**)&ghostToPIGPU,(ncellst-ncells)*sizeof(int));
  cudaMalloc((void**)&ghostToGhostGPU,(ncellst-ncells)*sizeof(int));

  bool auxbool = 0;
  cudaMemcpyToSymbol(setparticlesGPU,&auxbool,sizeof(bool));
  cudaMemcpyToSymbol(setboundaryGPU,&auxbool,sizeof(bool));


  long long auxulonglong = 0;
  cudaMalloc((void**)&stepGPU,sizeof(long long));
  cudaMemcpy(stepGPU,&auxulonglong,sizeof(long long),cudaMemcpyHostToDevice);




  

  
  cout << "CREATE CELLS GPU :              DONE" << endl;

  return 1;
}
