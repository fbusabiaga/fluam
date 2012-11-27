// Filename: freeMemoryRK3GPU.cu
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


bool freeMemoryRK3GPU(){
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

  cudaFree(dmGPU);
  cudaFree(dpxGPU);
  cudaFree(dpyGPU);
  cudaFree(dpzGPU);

  cudaFree(rxcellGPU);
  cudaFree(rycellGPU);
  cudaFree(rzcellGPU);

  
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
  cudaFree(stepGPU);

  cout << "FREE MEMORY GPU :               DONE" << endl; 


  return 1;
}

