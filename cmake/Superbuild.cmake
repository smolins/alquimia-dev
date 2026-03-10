include(ExternalProject)

# Set the install directory for dependencies
set(INSTALL_DIR "${CMAKE_BINARY_DIR}/install")

# Pass down compiler/flags to external projects
set(COMMON_CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
)

# Detect system dependencies to avoid unnecessary downloads
find_package(HDF5 QUIET)
if(HDF5_FOUND)
  message(STATUS "Found system HDF5: ${HDF5_INCLUDE_DIRS}")
  set(PETSC_HDF5_ARGS "--with-hdf5=1")
else()
  message(STATUS "HDF5 not found, will be downloaded by PETSc")
  set(PETSC_HDF5_ARGS "--download-hdf5=1")
endif()

find_package(BLAS QUIET)
find_package(LAPACK QUIET)
if(BLAS_FOUND AND LAPACK_FOUND)
  message(STATUS "Found system BLAS/LAPACK")
  set(PETSC_BLASLAPACK_ARGS "")
else()
  message(STATUS "BLAS/LAPACK not found, will be downloaded by PETSc")
  set(PETSC_BLASLAPACK_ARGS "--download-fblaslapack=1")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(PETSC_DEBUG_ARG "--with-debugging=1")
else()
  set(PETSC_DEBUG_ARG "--with-debugging=0")
endif()

# PETSc
ExternalProject_Add(petsc
    GIT_REPOSITORY https://gitlab.com/petsc/petsc.git
    GIT_TAG v3.20.0
    PREFIX ${CMAKE_BINARY_DIR}/external/petsc
    BUILD_IN_SOURCE 1
    UPDATE_DISCONNECTED 1
    CONFIGURE_COMMAND /usr/bin/python3.10 ./configure --prefix=${INSTALL_DIR} --with-mpi=1 ${PETSC_DEBUG_ARG} --with-shared-libraries=1 ${PETSC_HDF5_ARGS} ${PETSC_BLASLAPACK_ARGS}
    BUILD_COMMAND make
    INSTALL_COMMAND make install
)

# Chemistry engines options
option(XSDK_WITH_PFLOTRAN "Enables support for the PFlotran chemistry engine [ON]." ON)
option(XSDK_WITH_CRUNCHFLOW "Enables support for the CrunchFlow chemistry engine [ON]." ON)

if (NOT XSDK_WITH_PFLOTRAN AND NOT XSDK_WITH_CRUNCHFLOW)
  message(FATAL_ERROR "At least one chemistry engine must be enabled (XSDK_WITH_PFLOTRAN or XSDK_WITH_CRUNCHFLOW).")
endif()

set(ALQUIMIA_DEPS petsc)
set(ALQUIMIA_EXTRA_ARGS)

# PFLOTRAN
if (XSDK_WITH_PFLOTRAN)
  ExternalProject_Add(pflotran
      DEPENDS petsc
      GIT_REPOSITORY https://bitbucket.org/pflotran/pflotran
      GIT_TAG v5.0.0
      PREFIX ${CMAKE_BINARY_DIR}/external/pflotran
      CONFIGURE_COMMAND ""
      UPDATE_DISCONNECTED 1
      BUILD_COMMAND make -C src/pflotran libpflotranchem.a PETSC_DIR=${INSTALL_DIR} PETSC_ARCH=""
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/src/pflotran/libpflotranchem.a ${INSTALL_DIR}/lib/libpflotranchem.a
              COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}/include/pflotran
              COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/src/pflotran ${INSTALL_DIR}/include/pflotran
  )
  list(APPEND ALQUIMIA_DEPS pflotran)
  list(APPEND ALQUIMIA_EXTRA_ARGS 
       -DXSDK_WITH_PFLOTRAN=ON
       -DTPL_PFLOTRAN_LIBRARIES=${INSTALL_DIR}/lib/libpflotranchem.a
       -DTPL_PFLOTRAN_INCLUDE_DIRS=${INSTALL_DIR}/include/pflotran)
else()
  list(APPEND ALQUIMIA_EXTRA_ARGS -DXSDK_WITH_PFLOTRAN=OFF)
endif()

# CrunchFlow
if (XSDK_WITH_CRUNCHFLOW)
  ExternalProject_Add(crunchflow
      DEPENDS petsc
      GIT_REPOSITORY https://bitbucket.org/crunchflow/crunchtope-dev
      GIT_TAG master
      PREFIX ${CMAKE_BINARY_DIR}/external/crunchflow
      CONFIGURE_COMMAND ""
      UPDATE_DISCONNECTED 1
      PATCH_COMMAND git apply --check source/MakefileForAlquimia.patch && git apply source/MakefileForAlquimia.patch || echo "Patch already applied or failed"
      BUILD_COMMAND make -C source libcrunchchem.a PETSC_DIR=${INSTALL_DIR} PETSC_ARCH=""
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/source/libcrunchchem.a ${INSTALL_DIR}/lib/libcrunchchem.a
              COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTALL_DIR}/include/crunchflow
              COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/source ${INSTALL_DIR}/include/crunchflow
  )
  list(APPEND ALQUIMIA_DEPS crunchflow)
  list(APPEND ALQUIMIA_EXTRA_ARGS 
       -DXSDK_WITH_CRUNCHFLOW=ON
       -DTPL_CRUNCHFLOW_LIBRARIES=${INSTALL_DIR}/lib/libcrunchchem.a
       -DTPL_CRUNCHFLOW_INCLUDE_DIRS=${INSTALL_DIR}/include/crunchflow)
else()
  list(APPEND ALQUIMIA_EXTRA_ARGS -DXSDK_WITH_CRUNCHFLOW=OFF)
endif()

# Alquimia itself
ExternalProject_Add(alquimia_core
    DEPENDS ${ALQUIMIA_DEPS}
    SOURCE_DIR ${CMAKE_SOURCE_DIR}
    BINARY_DIR ${CMAKE_BINARY_DIR}/alquimia-build
    INSTALL_DIR ${INSTALL_DIR}
    CMAKE_ARGS
        ${COMMON_CMAKE_ARGS}
        ${ALQUIMIA_EXTRA_ARGS}
        -DPETSC_DIR=${INSTALL_DIR}
        -DPETSC_ARCH=
        -DALQUIMIA_SUPERBUILD=OFF
)

# Forward the test target to the inner build
# We remove the dependency on alquimia_core so that 'make test' doesn't 
# trigger a re-check of all dependencies.
add_custom_target(test
    COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${INSTALL_DIR}/lib:$ENV{LD_LIBRARY_PATH} ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/alquimia-build --target test
)

# Rule to install the contents of the local install directory to the final destination
install(DIRECTORY ${INSTALL_DIR}/ DESTINATION .)
