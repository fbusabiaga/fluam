// Filename: createCellsBinaryMixture.cpp
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


//#define GLOBALS_MOVE 1

#include "header.h"
#include "cells.h"
#include "headerOtherFluidVariables.h"


bool createCellsBinaryMixture(){

  int fx, fy, fz;
  int** neighbour;
  double dx, dy, dz, vol, areax, areay, areaz;
  

  ncells = mx * my * mz;

  if(incompressibleBinaryMixture || incompressibleBinaryMixtureMidPoint){
    mxt = mx ;
    myt = my ;
    mzt = mz ;
    ncellst = mxt * myt * mzt;
  }
  else{
    //With ghost cells
    mxt = mx + 2;
    myt = my + 2;
    mzt = mz + 2;
    ncellst = mxt * myt * mzt;
  }

  
  
    
  dx = lx/double(mx);
  dy = ly/double(my);
  dz = lz/double(mz);
  
  cVolume = dx * dy * dz;
  
  crx = new double [ncellst];
  cry = new double [ncellst];
  crz = new double [ncellst];
  cvx = new double [ncellst];
  cvy = new double [ncellst];
  cvz = new double [ncellst];
  cDensity = new double [ncellst];
  c = new double [ncellst];
  
  /*for(int j=0;j<ncells;j++){
    int fx, fy, fz;    
    int i;
    fx = j % mx;
    fy = (j % (my*mx)) / mx;
    fz = j / (mx*my);
    //go to index tanking into account ghost cells
    fx++;
    fy++;
    fz++;
    i = fx + fy*mxt + fz*mxt*myt;
    
    crx[i] = (modu(i,mx) + 0.5) * dx - 0.5 * lx;
    cry[i] = (modu(i,mx*my)/mx + 0.5) * dy - 0.5 * ly;
    crz[i] = (i/(mx*my) + 0.5) * dz - 0.5 * lz;
    
    
  }
  */

  cout << "CREATE CELLS :                  DONE" << endl;
  
  return 1;
}
