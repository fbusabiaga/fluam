// Filename: freeMemoryGPU.cu
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


bool freeMemoryGPU(){
  if(thermostat == 1){
    cudaFree(d_rand);
  }
  //cudaFree(massGPU);
  cudaFree(densityGPU);
  cudaUnbindTexture(texVxGPU);
  cudaUnbindTexture(texVyGPU);
  cudaUnbindTexture(texVzGPU);    
  cudaFree(vxGPU);
  cudaFree(vyGPU);
  cudaFree(vzGPU);
  cudaFree(densityPredictionGPU);
  cudaFree(vxPredictionGPU);
  cudaFree(vyPredictionGPU);
  cudaFree(vzPredictionGPU);

  //cudaFree(fxGPU);
  //cudaFree(fyGPU);
  //cudaFree(fzGPU);

  cudaFree(dmGPU);
  cudaFree(dpxGPU);
  cudaFree(dpyGPU);
  cudaFree(dpzGPU);

  cudaFree(rxcellGPU);
  cudaFree(rycellGPU);
  cudaFree(rzcellGPU);

  if((setboundary==1) || (setparticles==1)){
    cudaUnbindTexture(texrxboundaryGPU);    
    cudaUnbindTexture(texryboundaryGPU);    
    cudaUnbindTexture(texrzboundaryGPU);
    cudaUnbindTexture(texfxboundaryGPU);
    cudaUnbindTexture(texfyboundaryGPU);
    cudaUnbindTexture(texfzboundaryGPU);
    cudaFree(rxboundaryGPU);
    cudaFree(ryboundaryGPU);
    cudaFree(rzboundaryGPU);
    cudaFree(vxboundaryGPU);
    cudaFree(vyboundaryGPU);
    cudaFree(vzboundaryGPU);
    cudaFree(fxboundaryGPU);
    cudaFree(fyboundaryGPU);
    cudaFree(fzboundaryGPU);
    
    //freeDelta();

    cudaUnbindTexture(texCountParticlesInCellX);
    cudaUnbindTexture(texCountParticlesInCellY);
    cudaUnbindTexture(texCountParticlesInCellZ);
    cudaUnbindTexture(texPartInCellX);
    cudaUnbindTexture(texPartInCellY);
    cudaUnbindTexture(texPartInCellZ);
    cudaUnbindTexture(texCountParticlesInCellNonBonded);
    cudaUnbindTexture(texPartInCellNonBonded);
  }

  if(setparticles == 1){
    cudaFree(countPartInCellNonBonded);
    cudaFree(partInCellNonBonded);
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
    cudaUnbindTexture(texforceNonBonded1);
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
    cudaFreeArray(forceNonBonded1);
    cudaFree(saveForceX);
    cudaFree(saveForceY);
    cudaFree(saveForceZ);
  }
  
  cudaUnbindTexture(texvecino0GPU);
  cudaUnbindTexture(texvecino1GPU);
  cudaUnbindTexture(texvecino2GPU);
  cudaUnbindTexture(texvecino3GPU);
  cudaUnbindTexture(texvecino4GPU);
  cudaUnbindTexture(texvecino5GPU);
  cudaUnbindTexture(texvecinopxpyGPU);
  cudaUnbindTexture(texvecinopxmyGPU);
  cudaUnbindTexture(texvecinopxpzGPU);
  cudaUnbindTexture(texvecinopxmzGPU);
  cudaUnbindTexture(texvecinomxpyGPU);
  cudaUnbindTexture(texvecinomxmyGPU);
  cudaUnbindTexture(texvecinomxpzGPU);
  cudaUnbindTexture(texvecinomxmzGPU);
  cudaUnbindTexture(texvecinopypzGPU);
  cudaUnbindTexture(texvecinopymzGPU);
  cudaUnbindTexture(texvecinomypzGPU);
  cudaUnbindTexture(texvecinomymzGPU);
  cudaUnbindTexture(texvecinopxpypzGPU);
  cudaUnbindTexture(texvecinopxpymzGPU);
  cudaUnbindTexture(texvecinopxmypzGPU);
  cudaUnbindTexture(texvecinopxmymzGPU);
  cudaUnbindTexture(texvecinomxpypzGPU);
  cudaUnbindTexture(texvecinomxpymzGPU);
  cudaUnbindTexture(texvecinomxmypzGPU);
  cudaUnbindTexture(texvecinomxmymzGPU);

  cudaFree(vecino0GPU);
  cudaFree(vecino1GPU);
  cudaFree(vecino2GPU);
  cudaFree(vecino3GPU);
  cudaFree(vecino4GPU);
  cudaFree(vecino5GPU);

  cudaFree(vecinopxpyGPU);
  cudaFree(vecinopxmyGPU);
  cudaFree(vecinopxpzGPU);
  cudaFree(vecinopxmzGPU);
  cudaFree(vecinomxpyGPU);
  cudaFree(vecinomxmyGPU);
  cudaFree(vecinomxpzGPU);
  cudaFree(vecinomxmzGPU);  
  cudaFree(vecinopypzGPU);
  cudaFree(vecinopymzGPU);
  cudaFree(vecinomypzGPU);
  cudaFree(vecinomymzGPU);
  cudaFree(vecinopxpypzGPU);
  cudaFree(vecinopxpymzGPU);
  cudaFree(vecinopxmypzGPU);
  cudaFree(vecinopxmymzGPU);
  cudaFree(vecinomxpypzGPU);
  cudaFree(vecinomxpymzGPU);
  cudaFree(vecinomxmypzGPU);
  cudaFree(vecinomxmymzGPU);

  cudaFree(countparticlesincellX);
  cudaFree(countparticlesincellY);
  cudaFree(countparticlesincellZ);
  cudaFree(partincellX);
  cudaFree(partincellY);
  cudaFree(partincellZ);
  cudaFree(errorKernel);
  cudaFree(stepGPU);

  if(setCheckVelocity==1){
    cudaFree(rxCheckGPU);
    cudaFree(ryCheckGPU);
    cudaFree(rzCheckGPU);
    cudaFree(vxCheckGPU);
    cudaFree(vyCheckGPU);
    cudaFree(vzCheckGPU);
  }


  if(!freeOtherFluidVariablesGPU()) return 0;

  cout << "FREE MEMORY GPU :               DONE" << endl; 


  return 1;
}

