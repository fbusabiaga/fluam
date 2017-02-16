// Filename: simpleCubic.cpp
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


#include "header.h"
#include "cells.h"
#include "particles.h"

void simpleCubic(const int dimension){
  double dx, dy, dz;
  int nx, ny, nz, n;
    
  if(dimension == 3){
    sigma = pow(lx*ly*lz/(np),1./3.);
    nz = (int(lz/sigma) ? int(lz/sigma) : 1);
  }
  else{
    sigma = sqrt(lx*ly/np);
    nz = 1;
  }
  nx = (int(lx/sigma) ? int(lx/sigma) : 1);
  ny = (int(ly/sigma) ? int(ly/sigma) : 1);
    
  while((nx*ny*nz)<np){
    if((nx*ny*nz)<np) nx++;
    if((nx*ny*nz)<np) ny++;
    if((nx*ny*nz)<np) nz++;
  }

  dx = lx/double(nx);
  dy = ly/double(ny);
  dz = lz/double(nz);
  
  n = 0;  
  for (int i=0;i<=nz-1;i++){
    for(int j=0;j<=ny-1;j++){
      for(int k=0;k<=nx-1;k++){
	if(n<np){
	  n = n + 1;
	  rxParticle[n-1] = (k + 0.5) * dx - lx/2.;
	  ryParticle[n-1] = (j + 0.5) * dy - ly/2.;
	  rzParticle[n-1] = (i + 0.5) * dz - lz/2.;
	}
      }
    }
  } 
}

