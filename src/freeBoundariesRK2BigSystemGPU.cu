// Filename: freeBoundariesRK2GPU.cu
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


bool freeBoundariesRK2BigSystemGPU(){

  cutilSafeCall(cudaUnbindTexture(texrxboundaryGPU));    
  cutilSafeCall(cudaUnbindTexture(texryboundaryGPU));    
  cutilSafeCall(cudaUnbindTexture(texrzboundaryGPU));

  cutilSafeCall(cudaFree(rxboundaryGPU));
  cutilSafeCall(cudaFree(ryboundaryGPU));
  cutilSafeCall(cudaFree(rzboundaryGPU));
  cutilSafeCall(cudaFree(rxboundaryPredictionGPU));
  cutilSafeCall(cudaFree(ryboundaryPredictionGPU));
  cutilSafeCall(cudaFree(rzboundaryPredictionGPU));
  cutilSafeCall(cudaFree(vxboundaryGPU));
  cutilSafeCall(cudaFree(vyboundaryGPU));
  cutilSafeCall(cudaFree(vzboundaryGPU));
  cutilSafeCall(cudaFree(vxboundaryPredictionGPU));
  cutilSafeCall(cudaFree(vyboundaryPredictionGPU));
  cutilSafeCall(cudaFree(vzboundaryPredictionGPU));
  cutilSafeCall(cudaFree(fxboundaryGPU));
  cutilSafeCall(cudaFree(fyboundaryGPU));
  cutilSafeCall(cudaFree(fzboundaryGPU));


  if(setparticles){
    cutilSafeCall(cudaFree(countPartInCellNonBonded));
    cutilSafeCall(cudaFree(partInCellNonBonded));

    cutilSafeCall(cudaFree(neighbor0GPU));
    cutilSafeCall(cudaFree(neighbor1GPU));
    cutilSafeCall(cudaFree(neighbor2GPU));
    cutilSafeCall(cudaFree(neighbor3GPU));
    cutilSafeCall(cudaFree(neighbor4GPU));
    cutilSafeCall(cudaFree(neighbor5GPU));
    cutilSafeCall(cudaFree(neighborpxpyGPU));
    cutilSafeCall(cudaFree(neighborpxmyGPU));
    cutilSafeCall(cudaFree(neighborpxpzGPU));
    cutilSafeCall(cudaFree(neighborpxmzGPU));
    cutilSafeCall(cudaFree(neighbormxpyGPU));
    cutilSafeCall(cudaFree(neighbormxmyGPU));
    cutilSafeCall(cudaFree(neighbormxpzGPU));
    cutilSafeCall(cudaFree(neighbormxmzGPU));
    cutilSafeCall(cudaFree(neighborpypzGPU));
    cutilSafeCall(cudaFree(neighborpymzGPU));
    cutilSafeCall(cudaFree(neighbormypzGPU));
    cutilSafeCall(cudaFree(neighbormymzGPU));
    cutilSafeCall(cudaFree(neighborpxpypzGPU));
    cutilSafeCall(cudaFree(neighborpxpymzGPU));
    cutilSafeCall(cudaFree(neighborpxmypzGPU));
    cutilSafeCall(cudaFree(neighborpxmymzGPU));
    cutilSafeCall(cudaFree(neighbormxpypzGPU));
    cutilSafeCall(cudaFree(neighbormxpymzGPU));
    cutilSafeCall(cudaFree(neighbormxmypzGPU));
    cutilSafeCall(cudaFree(neighbormxmymzGPU));
    cutilSafeCall(cudaFree(pNeighbors));
  }

  freeErrorArray();
  cutilSafeCall(cudaFree(pc));
  freeDelta();

  if(setparticles){
    cutilSafeCall(cudaUnbindTexture(texforceNonBonded1));
    cutilSafeCall(cudaFreeArray(forceNonBonded1));
  }


  cout << "FREE BOUNDARIES GPU :           DONE" << endl; 

  return 1;
}
