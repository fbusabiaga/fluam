// Filename: initializeFluidIncompressibleGPU.cu
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


bool initializeFluidIncompressibleGPU(){
  
  cutilSafeCall(cudaMemcpy(vxGPU,cvx,ncells*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(vyGPU,cvy,ncells*sizeof(double),cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy(vzGPU,cvz,ncells*sizeof(double),cudaMemcpyHostToDevice));

  if(quasiNeutrallyBuoyant || quasiNeutrallyBuoyant2D || quasiNeutrallyBuoyant4pt2D || stokesLimit){
    cutilSafeCall(cudaMemcpy(rxcellGPU,crx,ncells*sizeof(double),cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(rycellGPU,cry,ncells*sizeof(double),cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(rzcellGPU,crz,ncells*sizeof(double),cudaMemcpyHostToDevice));
  }

  if(incompressibleBinaryMixture || incompressibleBinaryMixtureMidPoint)
    cutilSafeCall(cudaMemcpy(cGPU,c,ncells*sizeof(double),cudaMemcpyHostToDevice));

  
  cout << "INITIALIZE FLUID GPU :          DONE" << endl;

  return 1;
}