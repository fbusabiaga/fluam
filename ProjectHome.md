# **fluam** #
**fluam** is a finite volume code for fluctuating hydrodynamics implemented in CUDA. It has several schemes for compressible and incompressible fluids with particles in suspension.

## Installation instructions ##
  * You will need a NVIDIA GPU with compute capability 1.3 or higher to use fluam. You don't need any GPU to compile and modify the code.

  * Third-party software: you will need the CUDA compiler nvcc and CUDA libraries.

  * Edit the file fluam/bin/MakefileHeader to include the right path to the NVIDIA SDK files and the HydroGrid code in case you have it. Set the right architecture for your GPU in "NVCCFLAGS".

  * Go to fluam/bin/ and type make

## Use ##
To run **fluam** type "fluam data.main"
data.main is a file with the option for the simulation, look fluam/bin/data.main for the options. Input files examples in fluam/bin/ directory.

## Benchmark Results ##
**fluam** runs up to 30 steps per second in a 1 million cells system.
