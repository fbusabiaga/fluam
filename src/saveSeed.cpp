// Filename: saveSeed.cpp
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

bool saveSeed(){
  string NombreSeed;
  NombreSeed =  outputname + ".seed";
  ofstream fileSaveSeed(NombreSeed.c_str());
  fileSaveSeed << seed << endl;
  fileSaveSeed.close();
  
  return 1;
}
