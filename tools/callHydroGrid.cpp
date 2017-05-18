
#include <stdlib.h> 
#include <sstream>
#include <iostream>
#include <fstream>
// #include "visit_writer.h"
// #include "visit_writer.c"
using namespace std;


extern "C" {
#include "HydroGrid.h"
}



/*int main(int argc, char* argv[]){




  cout << "# End" << endl;
  }*/


bool callHydroGrid(const int option,
                   const string outputname,
                   double *c,
                   double *density,
                   const double *y_avg,
                   const int mx,
                   const int my,
                   const double lx,
                   const double ly,
                   const double dt,
                   const int samplefreq,
                   const int step){

  // Define variables
  static int nCells[3];
  static double systemLength[3];
  static double heatCapacity[1];

  // static double *v = new double [mx*my * 3];

  if(option == 0){
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

    nCells[0] = mx;
    nCells[1] = my;
    nCells[2] = 1;
    systemLength[0] = lx;
    systemLength[1] = ly;
    systemLength[2] = 1;   // 0
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
  else if(option == 1){
    updateHydroAnalysisMixture_C(NULL /*velocities*/, density /*densities*/, c /*concentrations*/);
  }
  else if(option == 2){
    writeToFiles_C(-1); // Write to files
    destroyHydroAnalysis_C();
  }
  else if(option == 3){
    writeToFiles_C(step); // Write to files
  }
  return 0;
}
