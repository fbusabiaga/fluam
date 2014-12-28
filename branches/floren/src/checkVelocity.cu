// Filename: checkVelocity.cu
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


int threadsPerBlockCheckVelocity = 128;
int numBlocksCheckVelocity;

__global__ void	kernelCheckVelocity(double *vxGPU, 
				    double *vyGPU, 
				    double *vzGPU,
				    double *rxcellGPU, 
				    double *rycellGPU, 
				    double *rzcellGPU,
				    double *rxCheckGPU, 
				    double *ryCheckGPU, 
				    double *rzCheckGPU,
				    double* vxCheckGPU, 
				    double* vyCheckGPU, 
				    double* vzCheckGPU,
				    int nCheck){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=nCheck) return; 

  double v;
  int icelx, icely, icelz; 

  double rx = rxCheckGPU[i];
  double ry = ryCheckGPU[i];
  double rz = rzCheckGPU[i];
  int mxmy = mxGPU * myGPU;
  double r, rp, rm;    
  {
    r = rx;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jx   = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;
    r = rx - 0.5*dxGPU;
    r = r - (int(r*invlxGPU + 0.5*((r>0)-(r<0)))) * lxGPU;
    int jxdx = int(r * invdxGPU + 0.5*mxGPU) % mxGPU;

    r = ry;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jy   = int(r * invdyGPU + 0.5*myGPU) % myGPU;
    r = ry - 0.5*dyGPU;
    r = r - (int(r*invlyGPU + 0.5*((r>0)-(r<0)))) * lyGPU;
    int jydy = int(r * invdyGPU + 0.5*myGPU) % myGPU;

    r = rz;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jz   = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;
    r = rz - 0.5*dzGPU;
    r = r - (int(r*invlzGPU + 0.5*((r>0)-(r<0)))) * lzGPU;
    int jzdz = int(r * invdzGPU + 0.5*mzGPU) % mzGPU;

    icelx  = jxdx;
    icelx += jy * mxGPU;
    icelx += jz * mxmy;

    icely  = jx;
    icely += jydy * mxGPU;
    icely += jz * mxmy;

    icelz  = jx;
    icelz += jy * mxGPU;
    icelz += jzdz * mxmy;
  }
  
  int vecino0, vecino1, vecino2, vecino3, vecino4, vecino5;
  int vecinopxpy, vecinopxmy, vecinopxpz, vecinopxmz;
  int vecinomxpy, vecinomxmy, vecinomxpz, vecinomxmz;
  int vecinopypz, vecinopymz, vecinomypz, vecinomymz;
  int vecinopxpypz, vecinopxpymz, vecinopxmypz, vecinopxmymz;
  int vecinomxpypz, vecinomxpymz, vecinomxmypz, vecinomxmymz;


  double dlx, dlxp, dlxm;
  double dly, dlyp, dlym;
  double dlz, dlzp, dlzm;

  //FORCE IN THE X DIRECTION
  //vecino0 = in->vecino0GPU[icelx];
  vecino0 = tex1Dfetch(texvecino0GPU, icelx);
  //vecino1 = in->vecino1GPU[icelx];
  vecino1 = tex1Dfetch(texvecino1GPU, icelx);
  //vecino2 = in->vecino2GPU[icelx];
  vecino2 = tex1Dfetch(texvecino2GPU, icelx);
  //vecino3 = in->vecino3GPU[icelx];
  vecino3 = tex1Dfetch(texvecino3GPU, icelx);
  //vecino4 = in->vecino4GPU[icelx];
  vecino4 = tex1Dfetch(texvecino4GPU, icelx);
  //vecino5 = in->vecino5GPU[icelx];
  vecino5 = tex1Dfetch(texvecino5GPU, icelx);
  //vecinopxpy = in->vecinopxpyGPU[icelx];
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelx);
  //vecinopxmy = in->vecinopxmyGPU[icelx];  
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelx);
  //vecinopxpz = in->vecinopxpzGPU[icelx];
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelx);
  //vecinopxmz = in->vecinopxmzGPU[icelx];
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelx);
  //vecinomxpy = in->vecinomxpyGPU[icelx];
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelx);
  //vecinomxmy = in->vecinomxmyGPU[icelx];
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelx);
  //vecinomxpz = in->vecinomxpzGPU[icelx];
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelx);
  //vecinomxmz = in->vecinomxmzGPU[icelx];
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelx);
  //vecinopypz = in->vecinopypzGPU[icelx];
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelx);
  //vecinopymz = in->vecinopymzGPU[icelx];
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelx);
  //vecinomypz = in->vecinomypzGPU[icelx];
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelx);
  //vecinomymz = in->vecinomymzGPU[icelx];
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelx);
  //vecinopxpypz = in->vecinopxpypzGPU[icelx];
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelx);
  //vecinopxpymz = in->vecinopxpymzGPU[icelx];
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelx);
  //vecinopxmypz = in->vecinopxmypzGPU[icelx];
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelx);
  //vecinopxmymz = in->vecinopxmymzGPU[icelx];
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelx);
  //vecinomxpypz = in->vecinomxpypzGPU[icelx];
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelx);
  //vecinomxpymz = in->vecinomxpymzGPU[icelx];
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelx);
  //vecinomxmypz = in->vecinomxmypzGPU[icelx];
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelx);
  //vecinomxmymz = in->vecinomxmymzGPU[icelx];
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelx);
  
  double auxdx = invdxGPU/1.5;
  double auxdy = invdyGPU/1.5;
  double auxdz = invdzGPU/1.5;

  r =  (rx - rxcellGPU[icelx] - dxGPU*0.5);
  rp = (rx - rxcellGPU[vecino3] - dxGPU*0.5);
  rm = (rx - rxcellGPU[vecino2] - dxGPU*0.5);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = tex1D(texDelta, fabs(r));
  dlxp = tex1D(texDelta, fabs(rp));
  dlxm = tex1D(texDelta, fabs(rm));

  r =  (ry - rycellGPU[icelx]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = tex1D(texDelta, fabs(r));
  dlyp = tex1D(texDelta, fabs(rp));
  dlym = tex1D(texDelta, fabs(rm));

  r =  (rz - rzcellGPU[icelx]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = tex1D(texDelta, fabs(r));
  dlzp = tex1D(texDelta, fabs(rp));
  dlzm = tex1D(texDelta, fabs(rm));

  v = dlxm * dlym * dlzm * fetch_double(texVxGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVxGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVxGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVxGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVxGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVxGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVxGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVxGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVxGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVxGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVxGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVxGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVxGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVxGPU,icelx) + 
    dlx  * dly  * dlzp * fetch_double(texVxGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVxGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVxGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVxGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVxGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVxGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVxGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVxGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVxGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVxGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVxGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVxGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVxGPU,vecinopxpypz);

  vxCheckGPU[i] = v;

  
  
  //FORCE IN THE Y DIRECTION
  //vecino0 = in->vecino0GPU[icelx];
  vecino0 = tex1Dfetch(texvecino0GPU, icely);
  //vecino1 = in->vecino1GPU[icelx];
  vecino1 = tex1Dfetch(texvecino1GPU, icely);
  //vecino2 = in->vecino2GPU[icelx];
  vecino2 = tex1Dfetch(texvecino2GPU, icely);
  //vecino3 = in->vecino3GPU[icelx];
  vecino3 = tex1Dfetch(texvecino3GPU, icely);
  //vecino4 = in->vecino4GPU[icelx];
  vecino4 = tex1Dfetch(texvecino4GPU, icely);
  //vecino5 = in->vecino5GPU[icelx];
  vecino5 = tex1Dfetch(texvecino5GPU, icely);
  //vecinopxpy = in->vecinopxpyGPU[icelx];
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icely);
  //vecinopxmy = in->vecinopxmyGPU[icelx];  
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icely);
  //vecinopxpz = in->vecinopxpzGPU[icelx];
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icely);
  //vecinopxmz = in->vecinopxmzGPU[icelx];
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icely);
  //vecinomxpy = in->vecinomxpyGPU[icelx];
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icely);
  //vecinomxmy = in->vecinomxmyGPU[icelx];
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icely);
  //vecinomxpz = in->vecinomxpzGPU[icelx];
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icely);
  //vecinomxmz = in->vecinomxmzGPU[icelx];
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icely);
  //vecinopypz = in->vecinopypzGPU[icelx];
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icely);
  //vecinopymz = in->vecinopymzGPU[icelx];
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icely);
  //vecinomypz = in->vecinomypzGPU[icelx];
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icely);
  //vecinomymz = in->vecinomymzGPU[icelx];
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icely);
  //vecinopxpypz = in->vecinopxpypzGPU[icelx];
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icely);
  //vecinopxpymz = in->vecinopxpymzGPU[icelx];
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icely);
  //vecinopxmypz = in->vecinopxmypzGPU[icelx];
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icely);
  //vecinopxmymz = in->vecinopxmymzGPU[icelx];
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icely);
  //vecinomxpypz = in->vecinomxpypzGPU[icelx];
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icely);
  //vecinomxpymz = in->vecinomxpymzGPU[icelx];
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icely);
  //vecinomxmypz = in->vecinomxmypzGPU[icelx];
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icely);
  //vecinomxmymz = in->vecinomxmymzGPU[icelx];
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icely);


  r =  (rx - rxcellGPU[icely]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  dlx = tex1D(texDelta, fabs(r));
  dlxp = tex1D(texDelta, fabs(rp));
  dlxm = tex1D(texDelta, fabs(rm));


  r =  (ry - rycellGPU[icely] - dyGPU*0.5);
  rp = (ry - rycellGPU[vecino4] - dyGPU*0.5);
  rm = (ry - rycellGPU[vecino1] - dyGPU*0.5); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  dly = tex1D(texDelta, fabs(r));
  dlyp = tex1D(texDelta, fabs(rp));
  dlym = tex1D(texDelta, fabs(rm));

  r =  (rz - rzcellGPU[icely]);
  rp = (rz - rzcellGPU[vecino5]);
  rm = (rz - rzcellGPU[vecino0]);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  dlz = tex1D(texDelta, fabs(r));
  dlzp = tex1D(texDelta, fabs(rp));
  dlzm = tex1D(texDelta, fabs(rm));

  v = dlxm * dlym * dlzm * fetch_double(texVyGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVyGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVyGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVyGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVyGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVyGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVyGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVyGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVyGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVyGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVyGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVyGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVyGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVyGPU,icely) + 
    dlx  * dly  * dlzp * fetch_double(texVyGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVyGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVyGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVyGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVyGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVyGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVyGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVyGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVyGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVyGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVyGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVyGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVyGPU,vecinopxpypz);

  vyCheckGPU[i] = v;

  //FORCE IN THE Z DIRECTION
  //vecino0 = in->vecino0GPU[icelx];
  vecino0 = tex1Dfetch(texvecino0GPU, icelz);
  //vecino1 = in->vecino1GPU[icelx];
  vecino1 = tex1Dfetch(texvecino1GPU, icelz);
  //vecino2 = in->vecino2GPU[icelx];
  vecino2 = tex1Dfetch(texvecino2GPU, icelz);
  //vecino3 = in->vecino3GPU[icelx];
  vecino3 = tex1Dfetch(texvecino3GPU, icelz);
  //vecino4 = in->vecino4GPU[icelx];
  vecino4 = tex1Dfetch(texvecino4GPU, icelz);
  //vecino5 = in->vecino5GPU[icelx];
  vecino5 = tex1Dfetch(texvecino5GPU, icelz);
  //vecinopxpy = in->vecinopxpyGPU[icelx];
  vecinopxpy = tex1Dfetch(texvecinopxpyGPU, icelz);
  //vecinopxmy = in->vecinopxmyGPU[icelx];  
  vecinopxmy = tex1Dfetch(texvecinopxmyGPU, icelz);
  //vecinopxpz = in->vecinopxpzGPU[icelx];
  vecinopxpz = tex1Dfetch(texvecinopxpzGPU, icelz);
  //vecinopxmz = in->vecinopxmzGPU[icelx];
  vecinopxmz = tex1Dfetch(texvecinopxmzGPU, icelz);
  //vecinomxpy = in->vecinomxpyGPU[icelx];
  vecinomxpy = tex1Dfetch(texvecinomxpyGPU, icelz);
  //vecinomxmy = in->vecinomxmyGPU[icelx];
  vecinomxmy = tex1Dfetch(texvecinomxmyGPU, icelz);
  //vecinomxpz = in->vecinomxpzGPU[icelx];
  vecinomxpz = tex1Dfetch(texvecinomxpzGPU, icelz);
  //vecinomxmz = in->vecinomxmzGPU[icelx];
  vecinomxmz = tex1Dfetch(texvecinomxmzGPU, icelz);
  //vecinopypz = in->vecinopypzGPU[icelx];
  vecinopypz = tex1Dfetch(texvecinopypzGPU, icelz);
  //vecinopymz = in->vecinopymzGPU[icelx];
  vecinopymz = tex1Dfetch(texvecinopymzGPU, icelz);
  //vecinomypz = in->vecinomypzGPU[icelx];
  vecinomypz = tex1Dfetch(texvecinomypzGPU, icelz);
  //vecinomymz = in->vecinomymzGPU[icelx];
  vecinomymz = tex1Dfetch(texvecinomymzGPU, icelz);
  //vecinopxpypz = in->vecinopxpypzGPU[icelx];
  vecinopxpypz = tex1Dfetch(texvecinopxpypzGPU, icelz);
  //vecinopxpymz = in->vecinopxpymzGPU[icelx];
  vecinopxpymz = tex1Dfetch(texvecinopxpymzGPU, icelz);
  //vecinopxmypz = in->vecinopxmypzGPU[icelx];
  vecinopxmypz = tex1Dfetch(texvecinopxmypzGPU, icelz);
  //vecinopxmymz = in->vecinopxmymzGPU[icelx];
  vecinopxmymz = tex1Dfetch(texvecinopxmymzGPU, icelz);
  //vecinomxpypz = in->vecinomxpypzGPU[icelx];
  vecinomxpypz = tex1Dfetch(texvecinomxpypzGPU, icelz);
  //vecinomxpymz = in->vecinomxpymzGPU[icelx];
  vecinomxpymz = tex1Dfetch(texvecinomxpymzGPU, icelz);
  //vecinomxmypz = in->vecinomxmypzGPU[icelx];
  vecinomxmypz = tex1Dfetch(texvecinomxmypzGPU, icelz);
  //vecinomxmymz = in->vecinomxmymzGPU[icelx];
  vecinomxmymz = tex1Dfetch(texvecinomxmymzGPU, icelz);


  r =  (rx - rxcellGPU[icelz]);
  rp = (rx - rxcellGPU[vecino3]);
  rm = (rx - rxcellGPU[vecino2]);
  r =  auxdx * (r - int(r*invlxGPU + 0.5*((r>0)-(r<0)))*lxGPU);
  rm = auxdx * (rm - int(rm*invlxGPU + 0.5*((rm>0)-(rm<0)))*lxGPU);
  rp = auxdx * (rp - int(rp*invlxGPU + 0.5*((rp>0)-(rp<0)))*lxGPU);
  //dlx = delta(r);
  //dlxp = delta(rp);
  //dlxm = delta(rm);
  dlx = tex1D(texDelta, fabs(r));
  dlxp = tex1D(texDelta, fabs(rp));
  dlxm = tex1D(texDelta, fabs(rm));

  r =  (ry - rycellGPU[icelz]);
  rp = (ry - rycellGPU[vecino4]);
  rm = (ry - rycellGPU[vecino1]); 
  r =  auxdy * (r - int(r*invlyGPU + 0.5*((r>0)-(r<0)))*lyGPU);
  rp = auxdy * (rp - int(rp*invlyGPU + 0.5*((rp>0)-(rp<0)))*lyGPU);
  rm = auxdy * (rm - int(rm*invlyGPU + 0.5*((rm>0)-(rm<0)))*lyGPU);
  //dly = delta(r);
  //dlyp = delta(rp);
  //dlym = delta(rm);
  dly = tex1D(texDelta, fabs(r));
  dlyp = tex1D(texDelta, fabs(rp));
  dlym = tex1D(texDelta, fabs(rm));

  r =  (rz - rzcellGPU[icelz] - dzGPU*0.5);
  rp = (rz - rzcellGPU[vecino5] - dzGPU*0.5);
  rm = (rz - rzcellGPU[vecino0] - dzGPU*0.5);
  r = auxdz * (r - int(r*invlzGPU + 0.5*((r>0)-(r<0)))*lzGPU);
  rp = auxdz * (rp - int(rp*invlzGPU + 0.5*((rp>0)-(rp<0)))*lzGPU);
  rm = auxdz * (rm - int(rm*invlzGPU + 0.5*((rm>0)-(rm<0)))*lzGPU);
  //dlz = delta(r);
  //dlzp = delta(rp);
  //dlzm = delta(rm);
  dlz = tex1D(texDelta, fabs(r));
  dlzp = tex1D(texDelta, fabs(rp));
  dlzm = tex1D(texDelta, fabs(rm));

  v = dlxm * dlym * dlzm * fetch_double(texVzGPU,vecinomxmymz) +
    dlxm * dlym * dlz  * fetch_double(texVzGPU,vecinomxmy) +
    dlxm * dlym * dlzp * fetch_double(texVzGPU,vecinomxmypz) +
    dlxm * dly  * dlzm * fetch_double(texVzGPU,vecinomxmz) + 
    dlxm * dly  * dlz  * fetch_double(texVzGPU,vecino2) +
    dlxm * dly  * dlzp * fetch_double(texVzGPU,vecinomxpz) +
    dlxm * dlyp * dlzm * fetch_double(texVzGPU,vecinomxpymz) +
    dlxm * dlyp * dlz  * fetch_double(texVzGPU,vecinomxpy) +
    dlxm * dlyp * dlzp * fetch_double(texVzGPU,vecinomxpypz) +
    dlx  * dlym * dlzm * fetch_double(texVzGPU,vecinomymz) +
    dlx  * dlym * dlz  * fetch_double(texVzGPU,vecino1) +
    dlx  * dlym * dlzp * fetch_double(texVzGPU,vecinomypz) + 
    dlx  * dly  * dlzm * fetch_double(texVzGPU,vecino0) + 
    dlx  * dly  * dlz  * fetch_double(texVzGPU,icelz) + 
    dlx  * dly  * dlzp * fetch_double(texVzGPU,vecino5) + 
    dlx  * dlyp * dlzm * fetch_double(texVzGPU,vecinopymz) + 
    dlx  * dlyp * dlz  * fetch_double(texVzGPU,vecino4) + 
    dlx  * dlyp * dlzp * fetch_double(texVzGPU,vecinopypz) + 
    dlxp * dlym * dlzm * fetch_double(texVzGPU,vecinopxmymz) + 
    dlxp * dlym * dlz  * fetch_double(texVzGPU,vecinopxmy) + 
    dlxp * dlym * dlzp * fetch_double(texVzGPU,vecinopxmypz) +
    dlxp * dly  * dlzm * fetch_double(texVzGPU,vecinopxmz) + 
    dlxp * dly  * dlz  * fetch_double(texVzGPU,vecino3) +
    dlxp * dly  * dlzp * fetch_double(texVzGPU,vecinopxpz) +
    dlxp * dlyp * dlzm * fetch_double(texVzGPU,vecinopxpymz) +
    dlxp * dlyp * dlz  * fetch_double(texVzGPU,vecinopxpy) +
    dlxp * dlyp * dlzp * fetch_double(texVzGPU,vecinopxpypz);

  vzCheckGPU[i] = v;

  return;

}

static ofstream fileOutputCheckVelocity;

void checkVelocity(int index, unsigned long long step, string fileCheckVelocity){
  
  if(index==0){
    ifstream fileinput(fileCheckVelocity.c_str());
    fileinput >> nCheck;
    double *rxCheck, *ryCheck, *rzCheck;
    rxCheck = new double [nCheck];
    ryCheck = new double [nCheck];
    rzCheck = new double [nCheck];
    for(int i=0;i<nCheck;i++)
      fileinput >> rxCheck[i] >> ryCheck[i] >> rzCheck[i];
    fileinput.close();

    cutilSafeCall(cudaMalloc((void**)&rxCheckGPU,nCheck*sizeof(double)));
    cutilSafeCall(cudaMalloc((void**)&ryCheckGPU,nCheck*sizeof(double)));
    cutilSafeCall(cudaMalloc((void**)&rzCheckGPU,nCheck*sizeof(double)));
    cutilSafeCall(cudaMemcpy(rxCheckGPU,rxCheck,nCheck*sizeof(double),cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(ryCheckGPU,ryCheck,nCheck*sizeof(double),cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(rzCheckGPU,rzCheck,nCheck*sizeof(double),cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMalloc((void**)&vxCheckGPU,nCheck*sizeof(double)));
    cutilSafeCall(cudaMalloc((void**)&vyCheckGPU,nCheck*sizeof(double)));
    cutilSafeCall(cudaMalloc((void**)&vzCheckGPU,nCheck*sizeof(double)));
    delete[] rxCheck;
    delete[] ryCheck;
    delete[] rzCheck;

    if((nCheck/threadsPerBlockCheckVelocity) < 60) threadsPerBlockCheckVelocity = 64;
    if((nCheck/threadsPerBlockCheckVelocity) < 60) threadsPerBlockCheckVelocity = 32;
    numBlocksCheckVelocity = (nCheck-1)/threadsPerBlockCheckVelocity + 1;

    string savefile;
    savefile =  outputname + ".checkVelocity";
    fileOutputCheckVelocity.open(savefile.c_str());
    fileOutputCheckVelocity << "#Number points    " << nCheck << endl;
  }
  else if(index==1){
    kernelCheckVelocity<<<numBlocksCheckVelocity,threadsPerBlockCheckVelocity>>>(vxGPU,
										 vyGPU,
										 vzGPU,
										 rxcellGPU,
										 rycellGPU,
										 rzcellGPU,
										 rxCheckGPU,
										 ryCheckGPU,
										 rzCheckGPU,
										 vxCheckGPU,
										 vyCheckGPU,
										 vzCheckGPU,
										 nCheck);
    double auxX[nCheck], auxY[nCheck], auxZ[nCheck];
    cutilSafeCall(cudaMemcpy(auxX,vxCheckGPU,nCheck*sizeof(double),cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(auxY,vyCheckGPU,nCheck*sizeof(double),cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpy(auxZ,vzCheckGPU,nCheck*sizeof(double),cudaMemcpyDeviceToHost));
    fileOutputCheckVelocity << step*dt << endl;
    for(int i=0;i<nCheck;i++)
      fileOutputCheckVelocity << auxX[i] << " " << auxY[i] << " " << auxZ[i] << endl;
  }
  else if(index==2){
    fileOutputCheckVelocity.close();
    cutilSafeCall(cudaFree(rxCheckGPU));
    cutilSafeCall(cudaFree(ryCheckGPU));
    cutilSafeCall(cudaFree(rzCheckGPU));
    cutilSafeCall(cudaFree(vxCheckGPU));
    cutilSafeCall(cudaFree(vyCheckGPU));
    cutilSafeCall(cudaFree(vzCheckGPU));
  }

  return;
}
