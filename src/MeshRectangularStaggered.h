#ifndef MeshRectangularStaggered_h
#define MeshRectangularStaggered_h

// Add includes
#include "defines.h"
#include "Mesh.h"

namespace fluam{

  /*
    \brief Class MeshRectangularStaggered is a concrete implementation of the
    base class Mesh that defines functionalities for a rectangular staggered mesh.
    Scalar variables are defined at the center of cells,
    Vector variables are defined at the faces of the cells,
    Second order tensors are defined at the centers and edges of the cells.
  */
  class MeshRectangularStaggered : public Mesh{
  public:
    /*
      Construnctor.
    */
    MeshRectangularStaggered();
    MeshRectangularStaggered(intd cells, reald L);
    MeshRectangularStaggered(intd cells, reald L, reald center);

    /*
      Destructor.
    */
    ~MeshRectangularStaggered();

    /*
      Get cells along the axes.
     */ 
    __host__ __device__ intd get_cells();

    /*
      Get total number of cells.
    */
    __host__ __device__ int get_ncells();
    
    /*
      Get length along the axes.
    */
   __host__ __device__  reald get_L();

   /*
     Get cell centers along one axis.
    */
   __host__ __device__ reald get_cell_centers_axis(int axis);

   /*
     Get all cell centers.
   */
   __host__ __device__ reald get_cell_centers();

  protected:
  private:
    // Number of cells along the axes and total number of cells
    intd d_cells;
    int d_ncells;
    // Axes dimensions
    reald d_L;
    // Cell length
    reald d_h;
    // Center of the mesh
    reald d_center;
  };

} // namespace fluam

#endif
