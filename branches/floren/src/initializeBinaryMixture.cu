// Filename: initializeBinaryMixture.cu
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


#include "header.h"
#include "fluid.h"
#include "cells.h"
#include "headerOtherFluidVariables.h"

bool initializeBinaryMixture(){
  if(!setBinaryMixture) return 1; //Don't use binary Mixture
  if((diffusion==0) || (massSpecies0==0) || (massSpecies1==0)){
    cout << "diffusion=0 or massSpecies0=0 or massSpecies1=0" << endl;
    return 0;
  }
  
  c = new double [ncells];
  

  

  cout << "INITIALIZE BINARY MIXTURE :    DONE" << endl;
  return 1;
}
