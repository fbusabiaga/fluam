// Filename: fluid.h
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


#ifdef GLOBALS_FLUID
#define EXTERN_FLUID 
#else
#define EXTERN_FLUID extern
#endif

EXTERN_FLUID double densfluid;
EXTERN_FLUID double shearviscosity;
EXTERN_FLUID double bulkviscosity;
EXTERN_FLUID double temperature;
EXTERN_FLUID double pressurea0;
EXTERN_FLUID double pressurea1;
EXTERN_FLUID double pressurea2;
EXTERN_FLUID double densityConst, dDensity, omega;
EXTERN_FLUID int initfluid;
//EXTERN_FLUID double concentration;
//Raul added. Saffman cut off wave number for quasi2D kernel and viscosity measure for quasi2D run scheme.
EXTERN_FLUID double saffmanCutOffWaveNumber;
EXTERN_FLUID double viscosityMeasureAmplitude;
EXTERN_FLUID double saffmanLayerWidth;
