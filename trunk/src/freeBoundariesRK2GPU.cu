// Filename: freeBoundariesRK2GPU.cu
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


bool freeBoundariesRK2GPU(){

  cudaUnbindTexture(texrxboundaryGPU);    
  cudaUnbindTexture(texryboundaryGPU);    
  cudaUnbindTexture(texrzboundaryGPU);
  //cudaUnbindTexture(texfxboundaryGPU);
  //cudaUnbindTexture(texfyboundaryGPU);
  //cudaUnbindTexture(texfzboundaryGPU);

  cudaFree(rxboundaryGPU);
  cudaFree(ryboundaryGPU);
  cudaFree(rzboundaryGPU);
  cudaFree(rxboundaryPredictionGPU);
  cudaFree(ryboundaryPredictionGPU);
  cudaFree(rzboundaryPredictionGPU);
  cudaFree(vxboundaryGPU);
  cudaFree(vyboundaryGPU);
  cudaFree(vzboundaryGPU);
  cudaFree(vxboundaryPredictionGPU);
  cudaFree(vyboundaryPredictionGPU);
  cudaFree(vzboundaryPredictionGPU);
  cudaFree(fxboundaryGPU);
  cudaFree(fyboundaryGPU);
  cudaFree(fzboundaryGPU);

  cudaUnbindTexture(texCountParticlesInCellX);
  cudaUnbindTexture(texCountParticlesInCellY);
  cudaUnbindTexture(texCountParticlesInCellZ);
  cudaUnbindTexture(texPartInCellX);
  cudaUnbindTexture(texPartInCellY);
  cudaUnbindTexture(texPartInCellZ);
  cudaUnbindTexture(texCountParticlesInCellNonBonded);
  cudaUnbindTexture(texPartInCellNonBonded);



  if(setparticles){
    cudaUnbindTexture(texCountParticlesInCellNonBonded);
    cudaFree(countPartInCellNonBonded);

    cudaUnbindTexture(texPartInCellNonBonded);
    cudaFree(partInCellNonBonded);

    cudaUnbindTexture(texneighbor0GPU);
    cudaUnbindTexture(texneighbor1GPU);
    cudaUnbindTexture(texneighbor2GPU);
    cudaUnbindTexture(texneighbor3GPU);
    cudaUnbindTexture(texneighbor4GPU);
    cudaUnbindTexture(texneighbor5GPU);
    cudaUnbindTexture(texneighborpxpyGPU);
    cudaUnbindTexture(texneighborpxmyGPU);
    cudaUnbindTexture(texneighborpxpzGPU);
    cudaUnbindTexture(texneighborpxmzGPU);
    cudaUnbindTexture(texneighbormxpyGPU);
    cudaUnbindTexture(texneighbormxmyGPU);
    cudaUnbindTexture(texneighbormxpzGPU);
    cudaUnbindTexture(texneighbormxmzGPU);
    cudaUnbindTexture(texneighborpypzGPU);
    cudaUnbindTexture(texneighborpymzGPU);
    cudaUnbindTexture(texneighbormypzGPU);
    cudaUnbindTexture(texneighbormymzGPU);
    cudaUnbindTexture(texneighborpxpypzGPU);
    cudaUnbindTexture(texneighborpxpymzGPU);
    cudaUnbindTexture(texneighborpxmypzGPU);
    cudaUnbindTexture(texneighborpxmymzGPU);
    cudaUnbindTexture(texneighbormxpypzGPU);
    cudaUnbindTexture(texneighbormxpymzGPU);
    cudaUnbindTexture(texneighbormxmypzGPU);
    cudaUnbindTexture(texneighbormxmymzGPU);
    cudaFree(neighbor0GPU);
    cudaFree(neighbor1GPU);
    cudaFree(neighbor2GPU);
    cudaFree(neighbor3GPU);
    cudaFree(neighbor4GPU);
    cudaFree(neighbor5GPU);
    cudaFree(neighborpxpyGPU);
    cudaFree(neighborpxmyGPU);
    cudaFree(neighborpxpzGPU);
    cudaFree(neighborpxmzGPU);
    cudaFree(neighbormxpyGPU);
    cudaFree(neighbormxmyGPU);
    cudaFree(neighbormxpzGPU);
    cudaFree(neighbormxmzGPU);
    cudaFree(neighborpypzGPU);
    cudaFree(neighborpymzGPU);
    cudaFree(neighbormypzGPU);
    cudaFree(neighbormymzGPU);
    cudaFree(neighborpxpypzGPU);
    cudaFree(neighborpxpymzGPU);
    cudaFree(neighborpxmypzGPU);
    cudaFree(neighborpxmymzGPU);
    cudaFree(neighbormxpypzGPU);
    cudaFree(neighbormxpymzGPU);
    cudaFree(neighbormxmypzGPU);
    cudaFree(neighbormxmymzGPU);
  }

  freeErrorArray();
  cudaFree(pc);
  freeDelta();

  if(setparticles){
    cudaUnbindTexture(texforceNonBonded1);
    cudaFreeArray(forceNonBonded1);
  }

  //No-slip Test
  //cudaFree(saveForceX);
  //cudaFree(saveForceY);
  //cudaFree(saveForceZ);

  cout << "FREE BOUNDARIES GPU :           DONE" << endl; 

  return 1;
}
