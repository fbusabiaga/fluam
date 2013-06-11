// Filename: initGhostIndexParticlesWallGPU.cu
//
// Copyright (c) 2010-2013, Florencio Balboa Usabiaga
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


__global__ void   kernelInitGhostIndexParticlesWallGPU(int* ghostIndexGPU);
__global__ void   kernelInitRealIndexParticlesWallGPU(int* ghostIndexGPU);
__global__ void   kernelGhostToPeriodicImageParticlesWall(int *ghostToPIGPU, 
							  int *ghostToGhostGPU, 
							  int *realIndexGPU, 
							  int* ghostIndexGPU);

bool initGhostIndexParticlesWallGPU(){
  //Init ghost index
  int threadsPerBlock = 128;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncells/threadsPerBlock) < 60) threadsPerBlock = 32;
  int numBlocks = (ncells-1)/threadsPerBlock + 1;

  kernelInitGhostIndexParticlesWallGPU<<<numBlocks,threadsPerBlock>>>(ghostIndexGPU);

  //init real index
  threadsPerBlock = 128;
  if((ncellst/threadsPerBlock) < 60) threadsPerBlock = 64;
  if((ncellst/threadsPerBlock) < 60) threadsPerBlock = 32;
  numBlocks = (ncellst-1)/threadsPerBlock + 1;
  
  kernelInitRealIndexParticlesWallGPU<<<numBlocks,threadsPerBlock>>>(realIndexGPU);

  
  kernelGhostToPeriodicImageParticlesWall<<<1,1>>>(ghostToPIGPU,ghostToGhostGPU,realIndexGPU,ghostIndexGPU);


  cout << "INIT GHOST INDEX :              DONE" << endl;
  return 1;
}


__global__ void   kernelInitGhostIndexParticlesWallGPU(int* ghostIndexGPU){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  int j, fx, fy, fz;
  fx = i % mxGPU;
  fy = (i % (mxGPU*myGPU)) / mxGPU;
  fz = i / (mxGPU*myGPU);
  //go to index tanking into account ghost cells
  fy++;
  j = fx + fy*mxtGPU + fz*mxtGPU*mytGPU;

  ghostIndexGPU[i] = j;

  return;

}



__global__ void   kernelInitRealIndexParticlesWallGPU(int* realIndexGPU){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellstGPU) return;   
  
  int j, fx, fy, fz;
  fx = i % mxtGPU;
  fy = (i % (mxmytGPU)) / mxtGPU;
  fz = i / (mxmytGPU);

  //go to real index from ghost cells
  fy--;
  
  fy = moduGPU(fy,myGPU);
  j = fx + fy*mxGPU + fz*mxGPU*myGPU;

  realIndexGPU[i] = j;

  return;

}


__global__ void kernelGhostToPeriodicImageParticlesWall(int *ghostToPIGPU, 
							int *ghostToGhostGPU, 
							int *realIndexGPU, 
							int *ghostIndexGPU){

  int fy, j;  
  j = 0;
  for(int i=0;i<ncellstGPU;i++){
    
    //fx = i % mxtGPU;
    fy = (i % (mxmytGPU)) / mxtGPU;
    //fz = i / (mxmytGPU);
    

    if( (fy==0)||(fy==(mytGPU-1)) ){
      ghostToPIGPU[j] = ghostIndexGPU[ realIndexGPU[i] ];
      ghostToGhostGPU[j] = i;
      j++;
    }
  }

  return;
}
