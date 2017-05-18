# Name executable
EFILE=spectra.exe

# Path to HydroGrid
SRC_HYDROLIB=../../HydroGrid/
vpath %.c $(SRC_HYDROLIB)
vpath %.h $(SRC_HYDROLIB)
vpath %.f90 $(SRC_HYDROLIB)

# Look for sources in the repository:
SRC_TOOLS=./
vpath %.cpp $(SRC_TOOLS)
vpath %.h $(SRC_TOOLS)

# Define compilers and flags and includes
FC = gfortran
CXX = g++
FCFLAGS = -O3 -Wall -ffree-line-length-none # Optimized 
CXXFLAGS = -O3 
CXXINCLS = -I./  
HydroAnalisysINCL = -I$(SRC_HYDROLIB)
LIBRARIES = -L../../HydroGrid/ -L/usr/lib64/ -L./ -lgfortran -lfftw3 -lfftw3f -lHydroAnalysis -lm


# Objects from fluam/tools
OBJS = callHydroGrid.o \
			 spectra.o 

# Objects from HydroGrid
# OBJS += \
#       Precision.o \
#       NURNGs.o \
#       RNG.o \
#       Random.o \
#       RNGs.o \
#       visit_writer.o \
#       FFTW.o \
#       VisitWriter.o \
#       HydroGridModule.o \
#       HydroGridCInterface.o 



all : $(OBJS) libCallHydroGrid.a ; $(CXX) $(CXXFLAGS) -o $(EFILE) $(OBJS) $(LIBRARIES) 

libCallHydroGrid.a: $(OBJS)
	ar rcs libCallHydroGrid.a $(OBJS)

spectra.o: spectra.cpp ; \
$(CXX) $(CXXFLAGS) $(CXXINCLS)  $(HydroAnalisysINCL) -c $<

callHydroGrid.o: callHydroGrid.cpp ; \
$(CXX) $(CXXFLAGS) $(CXXINCLS)  $(HydroAnalisysINCL) -c $<

#HydroAnalisis from Aleks
Precision.o: Precision.f90 ; \
	$(FC) $(FCFLAGS) -c $<

NURNGs.o: NURNGs.f90
	$(FC) $(FCFLAGS) -c $<

RNG.o: RNG.f90 Random.o ; \
	$(FC) $(FCFLAGS) -c $<

Random.o: Random.f90 Precision.o ; \
	$(FC) $(FCFLAGS) -c $<

RNGs.o: RNGs.c ; \
	$(CXX) $(CXXFLAGS) -c -DBL_FORT_USE_UNDERSCORE $<

visit_writer.o: visit_writer.c ; \
	$(CXX) $(CXXFLAGS) -c $<

FFTW.o: FFTW.f90 Precision.o ; \
	$(FC) $(FCFLAGS) -c $<

VisitWriter.o: VisitWriter.f90 ; \
	$(FC) $(FCFLAGS) -c $<

HydroGridModule.o: HydroGridModule.f90 FFTW.o VisitWriter.o ; \
	$(FC) $(FCFLAGS) -c $<

HydroGridCInterface.o: HydroGridCInterface.f90 HydroGridModule.o ; \
	$(FC) $(FCFLAGS) -c $<




clean:	; rm -f $(OBJS)  *.mod *.o *.exe *.a

