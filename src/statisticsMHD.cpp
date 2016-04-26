#include "header.h"
#include "cells.h"
#include "headerOtherFluidVariables.h"


static ofstream fileStatisticsMHD;

bool statisticsMHD(int index){

  double energyFluid = 0;
  double energyB = 0;
  double totalEnergy = 0;

  // Add energy of each cell
  for(int i=0; i<ncells; i++){
    energyFluid += cvx[i]*cvx[i] + cvy[i]*cvy[i] + cvz[i]*cvz[i];
    energyB     += cbx[i]*cbx[i] + cby[i]*cby[i] + cbz[i]*cbz[i];
    // energyFluid += cvx[i]*cvx[i] + cvy[i]*cvy[i];
    // energyB += cbx[i]*cbx[i] + cby[i]*cby[i];
  }

  // Scale energy with volume cell
  double dv = lx * ly / (1.0 * mx * my);
  energyFluid *= dv;
  energyB *= dv;
  totalEnergy = energyFluid + energyB;

  if(index == 0){
    string savefile;
    savefile = outputname +  ".statisticsMHD.dat";
    fileStatisticsMHD.open(savefile.c_str());
    fileStatisticsMHD << "# Columns: step, time, energy fluid, energy b, total energy "  << endl;
    fileStatisticsMHD << step << "  " << currentTime << "  " << energyFluid << "  " << energyB << "  " << totalEnergy << endl;   
  }
  else if(index == 1){
    fileStatisticsMHD << step << "  " << currentTime << "  " << energyFluid << "  " << energyB << "  " << totalEnergy << endl;
  }
  else if(index == 2){
    fileStatisticsMHD.close();
  }


  return 1;
}
