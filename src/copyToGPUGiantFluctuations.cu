// Filename: copyToGPUGiantFluctuations.cu
//
// Copyright (c) 2010-2012, Florencio Balboa Usabiaga
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


bool copyToGPUGiantFluctuations(){
  cudaMemcpyToSymbol(cWall0GPU,&cWall0,sizeof(double));
  cudaMemcpyToSymbol(cWall1GPU,&cWall1,sizeof(double));
  cudaMemcpyToSymbol(densityWall0GPU,&densityWall0,sizeof(double));
  cudaMemcpyToSymbol(densityWall1GPU,&densityWall1,sizeof(double));
  cudaMemcpyToSymbol(vxWall0GPU,&vxWall0,sizeof(double));
  cudaMemcpyToSymbol(vxWall1GPU,&vxWall1,sizeof(double));
  cudaMemcpyToSymbol(vyWall0GPU,&vyWall0,sizeof(double));
  cudaMemcpyToSymbol(vyWall1GPU,&vyWall1,sizeof(double));
  cudaMemcpyToSymbol(vzWall0GPU,&vzWall0,sizeof(double));
  cudaMemcpyToSymbol(vzWall1GPU,&vzWall1,sizeof(double));
  cudaMemcpyToSymbol(soretCoefficientGPU,&soretCoefficient,sizeof(double));
  cudaMemcpyToSymbol(gradTemperatureGPU,&gradTemperature,sizeof(double));

  cout << "COPY TO GPU :                   DONE" << endl;

  return 1;
}
