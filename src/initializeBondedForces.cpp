// Filename: initializeBondedForces.cu
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



#include <cstring>
#include <math.h>
#include "header.h"
#include "particles.h"
#include "fluid.h"
#include "parameters.h"
#include "cells.h"


bool initializeBondedForces(){
  
  int index1, index2;
  int trashInt;
  double trashDouble;

  if(bondedForcesOldVersion){
    initializeBondedForcesOldVersion();
    return 1;
  }
  

  //OPEN FILE
  ifstream file(bondedForcesFile.c_str());

  //Number of particle-particle bonds
  file >> nbondsParticleParticle;
  
  //Allocate memory
  bondsParticleParticle = new int [np];
  bondsParticleParticleOffset = new int [np];

  //Initially no particle has bonds
  for(int i=0;i<np;i++)
    bondsParticleParticle[i] = 0;

  //Information bonds particle-particle
  for(int i=0;i<nbondsParticleParticle;i++){  
    file >> index1 >> index2 >> trashDouble >> trashDouble;   
    bondsParticleParticle[index1]++;
    bondsParticleParticle[index2]++;
  }
  




  //Number of particle-fixedPoints bonds
  file >> nbondsParticleFixedPoint;

  //Allocate memory
  bondsParticleFixedPoint = new int [np];
  bondsParticleFixedPointOffset = new int [np];

  //Initially no particle has bonds
  for(int i=0;i<np;i++)
    bondsParticleFixedPoint[i] = 0;

  //Information bonds particle-fixedPoint
  for(int i=0;i<nbondsParticleFixedPoint;i++){  
    file >> index1 >> trashDouble >> trashDouble >> trashDouble >> trashDouble >> trashDouble;
    bondsParticleFixedPoint[index1]++;
  }
  
  //Important, lear how to rewind a file
  //CLOSE FILE 
  file.close();
  
  //Allocate memory
  bondsIndexParticleParticle = new int [nbondsParticleParticle * 2];
  kSpringParticleParticle = new double [nbondsParticleParticle * 2];
  r0ParticleParticle = new double [nbondsParticleParticle * 2];

  //bondsIndexParticleFixedPoint = new int [nbondsParticleFixedPoint];  
  kSpringParticleFixedPoint = new double [nbondsParticleFixedPoint];
  r0ParticleFixedPoint = new double [nbondsParticleFixedPoint];
  rxFixedPoint = new double [nbondsParticleFixedPoint];
  ryFixedPoint = new double [nbondsParticleFixedPoint];
  rzFixedPoint = new double [nbondsParticleFixedPoint];

  //Compute offset
  bondsParticleParticleOffset[0] = 0;
  bondsParticleFixedPointOffset[0] = 0;
  for(int i=1; i<np; i++){
    bondsParticleParticleOffset[i] = bondsParticleParticleOffset[i-1] + bondsParticleParticle[i-1];
    bondsParticleFixedPointOffset[i] = bondsParticleFixedPointOffset[i-1] + bondsParticleFixedPoint[i-1];
  }
  
  //Create tmp offset
  int *tmpOffset = new int [np];
  for(int i=0;i<np;i++){
    tmpOffset[i] = 0;
  }

  //OPEN THE FILE AGAIN
  file.open(bondedForcesFile.c_str());

  //Number of particle-particle bonds
  file >> nbondsParticleParticle;

  //Information bonds particle-particle
  double k, r0;
  for(int i=0;i<nbondsParticleParticle;i++){  
    file >> index1 >> index2 >> k >> r0;
    
    // Data for particle index1
    bondsIndexParticleParticle[bondsParticleParticleOffset[index1]+tmpOffset[index1]] = index2;
    kSpringParticleParticle[   bondsParticleParticleOffset[index1]+tmpOffset[index1]] = k;
    r0ParticleParticle[        bondsParticleParticleOffset[index1]+tmpOffset[index1]] = r0;

    // Data for particle index2
    bondsIndexParticleParticle[bondsParticleParticleOffset[index2]+tmpOffset[index2]] = index1;
    kSpringParticleParticle[   bondsParticleParticleOffset[index2]+tmpOffset[index2]] = k;
    r0ParticleParticle[        bondsParticleParticleOffset[index2]+tmpOffset[index2]] = r0;

    // Increase tmpOffset
    tmpOffset[index1]++;
    tmpOffset[index2]++;
  }

  // Reset tmp offset to zero for particle-fixed point interactions
  for(int i=0;i<np;i++){
    tmpOffset[i] = 0;
  }

  //Number of particle-fixedPoints bonds
  file >> nbondsParticleFixedPoint;

  //Information bonds particle-fixedPoint
  for(int i=0;i<nbondsParticleFixedPoint;i++){  
    file >> index1 >> kSpringParticleFixedPoint[bondsParticleFixedPointOffset[index1]+tmpOffset[index1]]
	 >> r0ParticleFixedPoint[               bondsParticleFixedPointOffset[index1]+tmpOffset[index1]]
	 >> rxFixedPoint[                       bondsParticleFixedPointOffset[index1]+tmpOffset[index1]]
	 >> ryFixedPoint[                       bondsParticleFixedPointOffset[index1]+tmpOffset[index1]]
	 >> rzFixedPoint[                       bondsParticleFixedPointOffset[index1]+tmpOffset[index1]];

    // Increase tmpOffset
    tmpOffset[index1]++;
  }

  //CLOSE FILE
  file.close();

  // Free tmpOffset
  delete[] tmpOffset;
  
  cout << "INITALIZE BONDED FORCES :       DONE " << endl;

  return 1;
}















bool freeBondedForces(){

  delete[] bondsParticleParticle;
  delete[] bondsParticleFixedPoint;

  delete[] bondsParticleParticleOffset;
  delete[] bondsParticleFixedPointOffset;


  delete[] bondsIndexParticleParticle;
  delete[] kSpringParticleParticle;
  delete[] r0ParticleParticle;
  

  //delete[] bondsIndexParticleFixedPoint;
  delete[] kSpringParticleFixedPoint;
  delete[] r0ParticleFixedPoint;
  delete[] rxFixedPoint;
  delete[] ryFixedPoint;
  delete[] rzFixedPoint;
  
}
