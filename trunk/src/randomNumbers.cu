// Filename: randomNumbers.cu
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



curandGenerator_t gen;
double *dRand;

bool initializeRandomNumbersGPU(size_t numberRandom, int seed){
  unsigned long long seedLong = seed;

  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,seedLong);
  
  cudaMalloc((void**)&dRand,numberRandom*sizeof(double));

  return 1;
}


bool generateRandomNumbers(size_t numberRandom){
  curandGenerateNormalDouble(gen,dRand,numberRandom,0,1);
  return 1;
}


bool freeRandomNumbersGPU(){
  cudaFree(dRand);
  curandDestroyGenerator(gen);
  return 1;
}
