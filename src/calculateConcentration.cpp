// Filename: calculateConcentration.cpp
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
#include "particles.h"
#include "headerOtherFluidVariables.h"
#include "cells.h"
#include "fluid.h"

// Includes from TestHydroGrid.c
extern "C" {
#include "HydroGrid.h"
}
// #include "hydroAnalysis.h"

bool callHydroGrid(const int option,
                   const string outputname,
                   double *c,
                   const double *y_avg,
                   const int mx,
                   const int my,
                   const double lx,
                   const double ly,
                   const double dt,
                   const int samplefreq,
                   const int step);

bool calculateConcentration(int option, long long step){

  // Declare static variables
  static int Npout, Npin, savefreq, npout, npin;
  static double dx, dy, inverse_volume_cell, ly_green_fraction;
  static double *y_init, *c, *y_av, *density; 
  static int nCells[3];
  static double systemLength[3];
  static double heatCapacity[1];

  if(option == 0){
    // Init variables
    Npout = 0, Npin = 0, savefreq = -1;

    npout = np;
    npin = 1;
    if((Npout>0) && (Npout<=np)) npout=Npout;
    if((Npin>0) && (Npin<=np)) npin=Npin;

    ly_green_fraction = 1.0 / 6.0;
    dx = lx / mxHydroGrid;
    dy = ly / myHydroGrid;
    inverse_volume_cell = 1.0 / (dx * dy);
    y_init = new double [np];
    c = new double [mxHydroGrid*myHydroGrid * 2]; // First half green, second half red
    density = new double [mxHydroGrid*myHydroGrid];
    y_av = new double [mxHydroGrid*myHydroGrid]; 
    double t;
 
    // Initialize HydroGrid
    if(1){
      ifstream fileinput ("hydroGridOptions.nml");
      string word, wordfile;
      while(!fileinput.eof()){
        getline(fileinput,word);
        wordfile += word + "\n";
      }
      fileinput.close();
      string fileOutName = outputname + ".hydroGridOptions.nml";
      ofstream fileout(fileOutName.c_str());
      fileout << wordfile << endl;
      fileout.close();
      
      nCells[0] = mxHydroGrid;
      nCells[1] = myHydroGrid;
      nCells[2] = 1;
      systemLength[0] = lx;
      systemLength[1] = ly;
      systemLength[2] = 1;   
      createHydroAnalysis_C(nCells,
                            3 /*nSpecies*/,
                            2 /*nVelocityDimensions*/,
                            1 /*isSingleFluid*/,
                            systemLength,
                            heatCapacity /*heatCapacity*/,
                            dt*samplefreq /*time step*/,
                            0 /*nPassiveScalars*/,
                            1 /*structFactMultiplier*/,
                            0 /*project2D*/);
    }
    for(int i=0;i<np;i++){
      y_init[i] = ryParticle[i]; // Save the initial coordinate
    }
  }
  else if(option == 1){
    // Set concentration to zero
    for(int i=0; i < mxHydroGrid*myHydroGrid; i++){
      c[i] = 0;
      y_av[i] = 0;
    }
    
    // Loop over particles and save as concentration
    for(int i=npin-1;i<npout;i++){
      double x = rxParticle[i] - (int(rxParticle[i]/lx + 0.5*((rxParticle[i]>0)-(rxParticle[i]<0)))) * lx;
      double y = ryParticle[i] - (int(ryParticle[i]/ly + 0.5*((ryParticle[i]>0)-(ryParticle[i]<0)))) * ly;  
      int jx   = int(x / dx + 0.5*mxHydroGrid) % mxHydroGrid;
      int jy   = int(y / dy + 0.5*myHydroGrid) % myHydroGrid;
      int icel = jx + jy * mxHydroGrid;

      // Is particle green or red
      if((y_init[i] < -ly * ly_green_fraction) or (y_init[i] > ly * ly_green_fraction)){ // Particle is red
        c[mxHydroGrid*myHydroGrid+icel] += 1.0;
      }
      else{ // Particle is green
        c[icel] += 1.0;
      }
      density[icel] += 1.0;
      y_av[icel] += y_init[i];
    }  

    for(int i=0; i < mxHydroGrid*myHydroGrid; i++){
      if(c[i]>0) {
        y_av[i] = y_av[i] / density[i]; // Average initial y coordinate for particles in this cell
      } else {
        y_av[i] = - (ly/2 + dy); // No particles in this cell so no information
      }   
      density[i]     = inverse_volume_cell*density[i];
      c[i]           = inverse_volume_cell*c[i];
      c[mxHydroGrid*myHydroGrid+i] = inverse_volume_cell*c[mxHydroGrid*myHydroGrid+i];
    }

    // Call HydroGrid to update data
    if(1){
      // updateHydroAnalysisIsothermal_C(NULL /*velocities*/, c /*densities*/);
      updateHydroAnalysisMixture_C(NULL /*velocities*/, density /*densities*/, c /*concentrations*/);
    }
  }
  else if(option == 3){
    // Call Hydrogrid to print data
    if(1){
      writeToFiles_C(step); // Write to files
    }
  }
  else if(option == 2){
    // Free HydroGrid
    if(1){
      writeToFiles_C(-1); // Write to files
      destroyHydroAnalysis_C();
    }
    delete[] c;
    delete[] density;
    delete[] y_init;
    delete[] y_av; 
  }

  return 1;
}
