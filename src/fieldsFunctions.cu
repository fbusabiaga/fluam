/*
 * Compute divergence of vector field
 * vxIn = input vector field
 * div = output, divergence of the vector field
 */
__global__ void divergence(const double *vxIn,
			   const double *vyIn,
			   const double *vzIn,
			   double *div){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  int vecino0, vecino1, vecino2; 
  double vx, vy, vz, vx2, vy1, vz0;

  // Read neighbors
  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);

  // Read vector field
  vx = vxIn[i];
  vy = vyIn[i];
  vz = vzIn[i];
  vx2 = vxIn[vecino2];
  vy1 = vyIn[vecino1];
  vz0 = vzIn[vecino0];

  div[i] = invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0);
  
  return;
}


/*
 * Compute divergence of vector field
 * vxIn = input vector field
 * div = output, divergence of the vector field
 */
__global__ void divergence(const cufftDoubleComplex *vxIn,
			   const cufftDoubleComplex *vyIn,
			   const cufftDoubleComplex *vzIn,
			   cufftDoubleComplex *div){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return;   

  int vecino0, vecino1, vecino2; 
  double vx, vy, vz, vx2, vy1, vz0;

  // Read neighbors
  vecino0 = tex1Dfetch(texvecino0GPU,i);
  vecino1 = tex1Dfetch(texvecino1GPU,i);
  vecino2 = tex1Dfetch(texvecino2GPU,i);

  // Read vector field
  vx = vxIn[i].x;
  vy = vyIn[i].x;
  vz = vzIn[i].x;
  vx2 = vxIn[vecino2].x;
  vy1 = vyIn[vecino1].x;
  vz0 = vzIn[vecino0].x;

  // Compute divergence (real part)
  div[i].x = invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0);

  // Read vector field
  vx = vxIn[i].y;
  vy = vyIn[i].y;
  vz = vzIn[i].y;
  vx2 = vxIn[vecino2].y;
  vy1 = vyIn[vecino1].y;
  vz0 = vzIn[vecino0].y;

  // Compute divergence (imaginary part)
  div[i].y = invdxGPU * (vx - vx2) + invdyGPU * (vy - vy1) + invdzGPU * (vz - vz0);

  return;
}



/*
 * Add two vector fields
 * output = alpha * vIn_1 + beta * vIn_2
 */
__global__ void addVectorFields(const double *vxIn_1,
				const double *vyIn_1,
				const double *vzIn_1,
				const double *vxIn_2,
				const double *vyIn_2,
				const double *vzIn_2,
				double *vxOut,
				double *vyOut,
				double *vzOut,
				const double alpha,
				const double beta){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return; 

  vxOut[i] = alpha * vxIn_1[i] + beta * vxIn_2[i];
  vyOut[i] = alpha * vyIn_1[i] + beta * vyIn_2[i];
  vzOut[i] = alpha * vzIn_1[i] + beta * vzIn_2[i];

  return;
}







__global__ void kernelMaxElement(const double *vxGPU, const double *vyGPU, const double *vzGPU,
				 const double *bxGPU, const double *byGPU, const double *bzGPU){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=ncellsGPU) return; 

  double maxElement = 0;
  
  for(int j=0; j<ncellsGPU; j++){
    if(maxElement < abs(vxGPU[j])){
      maxElement = abs(vxGPU[j]);
    }
    if(maxElement < abs(vyGPU[j])){
      maxElement = abs(vyGPU[j]);
    }
    if(maxElement < abs(vzGPU[j])){
      maxElement = abs(vzGPU[j]);
    }
    if(maxElement < abs(bxGPU[j])){
      maxElement = abs(bxGPU[j]);
    }
    if(maxElement < abs(byGPU[j])){
      maxElement = abs(byGPU[j]);
    }
    if(maxElement < abs(bzGPU[j])){
      maxElement = abs(bzGPU[j]);
    }
  }

  max_dev = maxElement;

  return;
}
