// Filename: saveFunctionsSchemeStokesLimit.cu
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


// Uncomment the appropiate line to use HydroGrid
#undef HydroGrid 
// #define HydroGrid 1


#ifdef HydroGrid
extern "C" {
#include "HydroGrid.h"
}
void calculateConcentration(string outputname,
                            double lx,       // Domain x length
                            double ly,       // Domain y length
                            int green_start, // Start of "green" particles
                            int green_end,   // End of "green" particles
                            int mx,          // Grid size x
                            int my,          // Grid size y
                            int step,        // Step of simulation
                            double dt,       // Time interval between successive snapshots (calls to updateHydroGrid)
                            int np,          // Number of particles
                            int option,      // option = 0 (initialize), 1 (update), 2 (save), 3 (finalize)
                            double *x_array, 
                            double *y_array);
#endif

bool saveFunctionsSchemeStokesLimit(int index, long long step, int samplefreq){

  if(index==0){
    if(!saveSeed()) return 0;
    if(setparticles)
      if(samplefreq > 0){
        if(!saveParticles(index,step)) return 0;
      }
#ifdef HydroGrid
    if((quasi2D or stokesLimit2D) and sampleHydroGrid > 0){
      calculateConcentration(outputname,
                             lx,                   // Domain x length
                             ly,                   // Domain y length
                             greenStart,           // Start of "green" particles
                             greenEnd,             // End of "green" particles
                             mxHydroGrid,          // Grid size x
                             myHydroGrid,          // Grid size y
                             step,                 // Step of simulation
                             dt * sampleHydroGrid, // Time interval between successive snapshots (calls to updateHydroGrid)
                             np,                   // Number of particles
                             0,                    // option = 0 (initialize), 1 (update), 2 (save), 3 (finalize)
                             rxParticle, 
                             ryParticle); 
    }
#endif  
    if(!saveTime(index)) return 0;
  }
  // Use save functions
  else if(index==1){
    if(setparticles)
      if(samplefreq > 0){
        if(!saveParticles(index,step)) return 0;
      }
  }
  // Close save functions
  else if(index==2){
    if(!saveTime(index)) return 0;
    if(setparticles)
      if(samplefreq > 0){
        if(!saveParticles(index,step)) return 0;
      }
#ifdef HydroGrid
    if((quasi2D or stokesLimit2D) and sampleHydroGrid > 0){
      calculateConcentration(outputname,
                             lx,                   // Domain x length
                             ly,                   // Domain y length
                             greenStart,           // Start of "green" particles
                             greenEnd,             // End of "green" particles
                             mxHydroGrid,          // Grid size x
                             myHydroGrid,          // Grid size y
                             step,                 // Step of simulation
                             dt * sampleHydroGrid, // Time interval between successive snapshots (calls to updateHydroGrid)
                             np,                   // Number of particles
                             3,                    // option = 0 (initialize), 1 (update), 2 (save), 3 (finalize)
                             rxParticle, 
                             ryParticle); 
    }
#endif  
  }
  // Update HydroGrid
  else if(index==3){
#ifdef HydroGrid
    if((quasi2D or stokesLimit2D) and sampleHydroGrid > 0){
      // The first step is special, because correlations
      if(step == 0){
        calculateConcentration(outputname,
                               lx,                   // Domain x length
                               ly,                   // Domain y length
                               greenStart,           // Start of "green" particles
                               greenEnd,             // End of "green" particles
                               mxHydroGrid,          // Grid size x
                               myHydroGrid,          // Grid size y
                               step,                 // Step of simulation
                               dt * sampleHydroGrid, // Time interval between successive snapshots (calls to updateHydroGrid)
                               np,                   // Number of particles
                               1,                    // option = 0 (initialize), 1 (update), 2 (save), 3 (finalize)
                               rxParticle, 
                               ryParticle); 
        calculateConcentration(outputname,
                               lx,                   // Domain x length
                               ly,                   // Domain y length
                               greenStart,           // Start of "green" particles
                               greenEnd,             // End of "green" particles
                               mxHydroGrid,          // Grid size x
                               myHydroGrid,          // Grid size y
                               step,                 // Step of simulation
                               dt * sampleHydroGrid, // Time interval between successive snapshots (calls to updateHydroGrid)
                               np,                   // Number of particles
                               2,                    // option = 0 (initialize), 1 (update), 2 (save), 3 (finalize)
                               rxParticle, 
                               ryParticle); 
      }
      calculateConcentration(outputname,
                             lx,                   // Domain x length
                             ly,                   // Domain y length
                             greenStart,           // Start of "green" particles
                             greenEnd,             // End of "green" particles
                             mxHydroGrid,          // Grid size x
                             myHydroGrid,          // Grid size y
                             step,                 // Step of simulation
                             dt * sampleHydroGrid, // Time interval between successive snapshots (calls to updateHydroGrid)
                             np,                   // Number of particles
                             1,                    // option = 0 (initialize), 1 (update), 2 (save), 3 (finalize)
                             rxParticle, 
                             ryParticle); 
    }
#endif  
  }
  // Print HydroGrid
  else if(index==4){
#ifdef HydroGrid
    if((quasi2D or stokesLimit2D) and sampleHydroGrid > 0){
      calculateConcentration(outputname,
                             lx,                   // Domain x length
                             ly,                   // Domain y length
                             greenStart,           // Start of "green" particles
                             greenEnd,             // End of "green" particles
                             mxHydroGrid,          // Grid size x
                             myHydroGrid,          // Grid size y
                             step,                 // Step of simulation
                             dt * sampleHydroGrid, // Time interval between successive snapshots (calls to updateHydroGrid)
                             np,                   // Number of particles
                             2,                    // option = 0 (initialize), 1 (update), 2 (save), 3 (finalize)
                             rxParticle, 
                             ryParticle); 
    }
#endif  
  }
  else{
    cout << "SAVE FUNCTIONS ERROR, INDEX !=0,1,2 " << endl;
    return 0;
  }
  
  return 1;
}
