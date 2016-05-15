#include "header.h"
#include "cells.h"
#include "headerOtherFluidVariables.h"


static ofstream fileStatisticsMHD;
static ofstream fileJacobianV_1_N, fileJacobianV_2_N, fileJacobianV_4_N;
static ofstream fileJacobianB_1_N, fileJacobianB_2_N, fileJacobianB_4_N;

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

  // Compute Jacobian and integrate it in regions around x=0,
  // Components {xx, xy, yx, yy}
  double dx = lx / mx;
  double dy = ly / my;
  double Jv[4], Jb[4];
  double Jcount_1_N[4]={}; // for |x| < 1/N
  double Jcount_2_N[4]={}; // for |x| < 1/N
  double Jcount_4_N[4]={}; // for |x| < 1/N
  double Jv_1_N[4]={};     // for |x| < 1/N
  double Jv_2_N[4]={};     // for |x| < 2/N
  double Jv_4_N[4]={};     // for |x| < 4/N
  double Jb_1_N[4]={};     // for |x| < 1/N
  double Jb_2_N[4]={};     // for |x| < 2/N
  double Jb_4_N[4]={};     // for |x| < 4/N
  double distance_1_N = 1.0 / max(kernelWidthN, 1);
  double distance_2_N = 2.0 / max(kernelWidthN, 1);
  double distance_4_N = 4.0 / max(kernelWidthN, 1);
  // Loop to compute Jv and Jb
  double max=0;
  double min= 100;
  for(int i=0; i<ncells; i++){
    int fz = i/(mx*my);
    int fy = (i % (mx*my))/mx;
    int fx = i % mx;     
    int fzp1 = ((fz+1) % mz) ;
    int fyp1 = ((fy+1) % my) ;
    int fxp1 = ((fx+1) % mx);      
    int fzm1 = ((fz-1+mz) % mz) ;
    int fym1 = ((fy-1+my) % my) ;
    int fxm1 = ((fx-1+mx) % mx);      

    int vecino5 = fzp1*mx*my + fy*mx   + fx;
    int vecino4 = fz*mx*my   + fyp1*mx + fx;
    int vecino3 = fz*mx*my   + fy*mx   + fxp1;    
    int vecino2 = fz*mx*my   + fy*mx   + fxm1;
    int vecino1 = fz*mx*my   + fym1*mx + fx;
    int vecino0 = fzm1*mx*my + fy*mx   + fx;    
    
    // Diagonal elements defined in the cell centers
    if(fabs(crx[i] - 0) <= distance_4_N){
      // Jv
      Jv[0] = (cvx[i]-cvx[vecino2]) / dx;
      Jv[3] = (cvy[i]-cvy[vecino1]) / dy;
      Jv_1_N[0] += Jv[0];
      Jv_1_N[3] += Jv[3];
      Jv_2_N[0] += Jv[0];
      Jv_2_N[3] += Jv[3];
      Jv_4_N[0] += Jv[0];
      Jv_4_N[3] += Jv[3];
      Jcount_1_N[0] += 1.0; 
      Jcount_1_N[3] += 1.0; 
      Jcount_2_N[0] += 1.0; 
      Jcount_2_N[3] += 1.0; 
      Jcount_4_N[0] += 1.0; 
      Jcount_4_N[3] += 1.0; 
      // Jb
      Jb[0] = (cbx[i]-cbx[vecino2]) / dx;
      Jb[3] = (cby[i]-cby[vecino1]) / dy;
      Jb_1_N[0] += Jb[0];
      Jb_1_N[3] += Jb[3];
      Jb_2_N[0] += Jb[0];
      Jb_2_N[3] += Jb[3];
      Jb_4_N[0] += Jb[0];
      Jb_4_N[3] += Jb[3];
    }
    else if(fabs(crx[i] - 0) <= distance_2_N){
      // Jv
      Jv[0] = (cvx[i]-cvx[vecino2]) / dx;
      Jv[3] = (cvy[i]-cvy[vecino1]) / dy;
      Jv_1_N[0] += Jv[0];
      Jv_1_N[3] += Jv[3];
      Jv_2_N[0] += Jv[0];
      Jv_2_N[3] += Jv[3];
      Jcount_1_N[0] += 1.0; 
      Jcount_1_N[3] += 1.0; 
      Jcount_2_N[0] += 1.0; 
      Jcount_2_N[3] += 1.0; 
      // Jb
      Jb[0] = (cbx[i]-cbx[vecino2]) / dx;
      Jb[3] = (cby[i]-cby[vecino1]) / dy;
      Jb_1_N[0] += Jb[0];
      Jb_1_N[3] += Jb[3];
      Jb_2_N[0] += Jb[0];
      Jb_2_N[3] += Jb[3];
    }
    else if(fabs(crx[i] - 0) <= distance_1_N){
      // Jv
      Jv[0] = (cvx[i]-cvx[vecino2]) / dx;
      Jv[3] = (cvy[i]-cvy[vecino1]) / dy;
      Jv_1_N[0] += Jv[0];
      Jv_1_N[3] += Jv[3];
      Jcount_1_N[0] += 1.0; 
      Jcount_1_N[3] += 1.0; 
      // Jb
      Jb[0] = (cbx[i]-cbx[vecino2]) / dx;
      Jb[3] = (cby[i]-cby[vecino1]) / dy;
      Jb_1_N[0] += Jb[0];
      Jb_1_N[3] += Jb[3];
    }
    // Off-diagonal elements defined on the cell corners (in 2D)
    if(fabs(crx[i]+0.5*dx - 0) <= distance_4_N){
      // Jv
      Jv[1] = (cvx[vecino4] - cvx[i]) / dy;
      Jv[2] = (cvy[vecino3] - cvy[i]) / dx;
      Jv_1_N[1] += Jv[1];
      Jv_1_N[2] += Jv[2];
      Jv_2_N[1] += Jv[1];
      Jv_2_N[2] += Jv[2];
      Jv_4_N[1] += Jv[1];
      Jv_4_N[2] += Jv[2];
      Jcount_1_N[1] += 1.0; 
      Jcount_1_N[2] += 1.0; 
      Jcount_2_N[1] += 1.0; 
      Jcount_2_N[2] += 1.0; 
      Jcount_4_N[1] += 1.0; 
      Jcount_4_N[2] += 1.0; 
      // Jb
      Jb[1] = (cbx[vecino4] - cbx[i]) / dy;
      Jb[2] = (cby[vecino3] - cby[i]) / dx;
      Jb_1_N[1] += Jb[1];
      Jb_1_N[2] += Jb[2];
      Jb_2_N[1] += Jb[1];
      Jb_2_N[2] += Jb[2];
      Jb_4_N[1] += Jb[1];
      Jb_4_N[2] += Jb[2];
    }
    else if(fabs(crx[i]+0.5*dx - 0) <= distance_2_N){
      // Jv
      Jv[1] = (cvx[vecino4] - cvx[i]) / dy;
      Jv[2] = (cvy[vecino3] - cvy[i]) / dx;
      Jv_1_N[1] += Jv[1];
      Jv_1_N[2] += Jv[2];
      Jv_2_N[1] += Jv[1];
      Jv_2_N[2] += Jv[2];
      Jcount_1_N[1] += 1.0; 
      Jcount_1_N[2] += 1.0; 
      Jcount_2_N[1] += 1.0; 
      Jcount_2_N[2] += 1.0; 
      // Jb
      Jb[1] = (cbx[vecino4] - cbx[i]) / dy;
      Jb[2] = (cby[vecino3] - cby[i]) / dx;
      Jb_1_N[1] += Jb[1];
      Jb_1_N[2] += Jb[2];
      Jb_2_N[1] += Jb[1];
      Jb_2_N[2] += Jb[2];
    }
    if(fabs(crx[i]+0.5*dx - 0) <= distance_1_N){
      // Jv
      Jv[1] = (cvx[vecino4] - cvx[i]) / dy;
      Jv[2] = (cvy[vecino3] - cvy[i]) / dx;
      Jv_1_N[1] += Jv[1];
      Jv_1_N[2] += Jv[2];
      Jcount_1_N[1] += 1.0; 
      Jcount_1_N[2] += 1.0; 
      // Jb
      Jb[1] = (cbx[vecino4] - cbx[i]) / dy;
      Jb[2] = (cby[vecino3] - cby[i]) / dx;
      Jb_1_N[1] += Jb[1];
      Jb_1_N[2] += Jb[2];
    }
  }
  // Normalize Jacobians
  for(int i=0; i<4; i++){
    Jv_1_N[i] /= Jcount_1_N[i];
    Jv_2_N[i] /= Jcount_2_N[i];
    Jv_4_N[i] /= Jcount_4_N[i];
    Jb_1_N[i] /= Jcount_1_N[i];
    Jb_2_N[i] /= Jcount_2_N[i];
    Jb_4_N[i] /= Jcount_4_N[i];
  }
  


  if(index == 0){
    string savefile;
    savefile = outputname +  ".statisticsMHD.dat";
    fileStatisticsMHD.open(savefile.c_str());
    fileStatisticsMHD << "# Columns: step, time, energy fluid, energy b, total energy "  << endl;
    fileStatisticsMHD << step << "  " << currentTime << "  " << energyFluid << "  " << energyB << "  " << totalEnergy << endl;   
    // Jv
    savefile = outputname +  ".jacobianV_1_N.dat";
    fileJacobianV_1_N.open(savefile.c_str());
    fileJacobianV_1_N << "# Columns: step, time, Jv_xx, Jv_xy, Jv_yx, Jv_yy" << endl;
    fileJacobianV_1_N << step << "  " << currentTime << "  " << Jv_1_N[0] << "  " << Jv_1_N[1] << "  " << Jv_1_N[2] << "  " << Jv_1_N[3] << endl;
    savefile = outputname +  ".jacobianV_2_N.dat";
    fileJacobianV_2_N.open(savefile.c_str());
    fileJacobianV_2_N << "# Columns: step, time, Jv_xx, Jv_xy, Jv_yx, Jv_yy" << endl;
    fileJacobianV_2_N << step << "  " << currentTime << "  " << Jv_2_N[0] << "  " << Jv_2_N[1] << "  " << Jv_2_N[2] << "  " << Jv_2_N[3] << endl;
    savefile = outputname +  ".jacobianV_4_N.dat";
    fileJacobianV_4_N.open(savefile.c_str());
    fileJacobianV_4_N << "# Columns: step, time, Jv_xx, Jv_xy, Jv_yx, Jv_yy" << endl;
    fileJacobianV_4_N << step << "  " << currentTime << "  " << Jv_4_N[0] << "  " << Jv_4_N[1] << "  " << Jv_4_N[2] << "  " << Jv_4_N[3] << endl;
    // Jb
    savefile = outputname +  ".jacobianB_1_N.dat";
    fileJacobianB_1_N.open(savefile.c_str());
    fileJacobianB_1_N << "# Columns: step, time, Jb_xx, Jb_xy, Jb_yx, Jb_yy" << endl;
    fileJacobianB_1_N << step << "  " << currentTime << "  " << Jb_1_N[0] << "  " << Jb_1_N[1] << "  " << Jb_1_N[2] << "  " << Jb_1_N[3] << endl;
    savefile = outputname +  ".jacobianB_2_N.dat";
    fileJacobianB_2_N.open(savefile.c_str());
    fileJacobianB_2_N << "# Columns: step, time, Jb_xx, Jb_xy, Jb_yx, Jb_yy" << endl;
    fileJacobianB_2_N << step << "  " << currentTime << "  " << Jb_2_N[0] << "  " << Jb_2_N[1] << "  " << Jb_2_N[2] << "  " << Jb_2_N[3] << endl;
    savefile = outputname +  ".jacobianB_4_N.dat";
    fileJacobianB_4_N.open(savefile.c_str());
    fileJacobianB_4_N << "# Columns: step, time, Jb_xx, Jb_xy, Jb_yx, Jb_yy" << endl;
    fileJacobianB_4_N << step << "  " << currentTime << "  " << Jb_4_N[0] << "  " << Jb_4_N[1] << "  " << Jb_4_N[2] << "  " << Jb_4_N[3] << endl;
  }
  else if(index == 1){
    fileStatisticsMHD << step << "  " << currentTime << "  " << energyFluid << "  " << energyB << "  " << totalEnergy << endl;
    // Jv
    fileJacobianV_1_N << step << "  " << currentTime << "  " << Jv_1_N[0] << "  " << Jv_1_N[1] << "  " << Jv_1_N[2] << "  " << Jv_1_N[3] << endl;
    fileJacobianV_2_N << step << "  " << currentTime << "  " << Jv_2_N[0] << "  " << Jv_2_N[1] << "  " << Jv_2_N[2] << "  " << Jv_2_N[3] << endl;
    fileJacobianV_4_N << step << "  " << currentTime << "  " << Jv_4_N[0] << "  " << Jv_4_N[1] << "  " << Jv_4_N[2] << "  " << Jv_4_N[3] << endl;
    // Jb
    fileJacobianB_1_N << step << "  " << currentTime << "  " << Jb_1_N[0] << "  " << Jb_1_N[1] << "  " << Jb_1_N[2] << "  " << Jb_1_N[3] << endl;
    fileJacobianB_2_N << step << "  " << currentTime << "  " << Jb_2_N[0] << "  " << Jb_2_N[1] << "  " << Jb_2_N[2] << "  " << Jb_2_N[3] << endl;
    fileJacobianB_4_N << step << "  " << currentTime << "  " << Jb_4_N[0] << "  " << Jb_4_N[1] << "  " << Jb_4_N[2] << "  " << Jb_4_N[3] << endl;
  }
  else if(index == 2){
    fileStatisticsMHD.close();
    fileJacobianV_1_N.close();
    fileJacobianV_2_N.close();
    fileJacobianV_4_N.close();
    fileJacobianB_1_N.close();
    fileJacobianB_2_N.close();
    fileJacobianB_4_N.close();
  }


  return 1;
}
