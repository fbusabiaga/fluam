#include "MeshRectangularStaggered.h"

namespace fluam{

  MeshRectangularStaggered::MeshRectangularStaggered(){

  } 
  
  MeshRectangularStaggered::MeshRectangularStaggered(intd cells, reald L) : d_cells(cells), d_L(L){
    if(DIM == 3){
      d_ncells = L.x * L.y * L.z;
      d_h.x = d_L.x / d_cells.x;
      d_h.y = d_L.y / d_cells.y;
      d_h.z = d_L.z / d_cells.z;
    }
    else{
      d_ncells = L.x * L.y;
      d_h.x = d_L.x / d_cells.x;
      d_h.y = d_L.y / d_cells.y;
    }
  }

  MeshRectangularStaggered::MeshRectangularStaggered(intd cells, reald L, reald center) : d_cells(cells), d_L(L), d_center(center){
    if(DIM == 3){
      d_ncells = L.x * L.y * L.z;
      d_h.x = d_L.x / d_cells.x;
      d_h.y = d_L.y / d_cells.y;
      d_h.z = d_L.z / d_cells.z;
    }
    else{
      d_ncells = L.x * L.y;
      d_h.x = d_L.x / d_cells.x;
      d_h.y = d_L.y / d_cells.y;
    }
  }

  MeshRectangularStaggered::~MeshRectangularStaggered(){
    
  }

  __host__ __device__ intd MeshRectangularStaggered::get_cells(){
    return d_cells;
  }

  __host__ __device__ int MeshRectangularStaggered::get_ncells(){
    return d_ncells;
  }

  __host__ __device__ reald MeshRectangularStaggered::get_L(){
    return d_L;
  }
}
