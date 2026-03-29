# Building Standalone Chemistry Engines with Alquimia

Alquimia now supports building the standalone executables for its backend chemistry engines (CrunchFlow and PFLOTRAN) alongside the core Alquimia interface libraries.

## How to Enable Standalone Builds

When configuring Alquimia with CMake via the Superbuild, you can enable the standalone builds by passing the `-DALQUIMIA_BUILD_STANDALONE_ENGINES=ON` flag. 

### Example Configuration:
```bash
mkdir build && cd build
cmake .. \
  -DALQUIMIA_SUPERBUILD=ON \
  -DXSDK_WITH_PFLOTRAN=ON \
  -DXSDK_WITH_CRUNCHFLOW=ON \
  -DALQUIMIA_BUILD_STANDALONE_ENGINES=ON
make -j$(nproc)
```

## What Happens During the Build?

When `ALQUIMIA_BUILD_STANDALONE_ENGINES` is set to `ON`, the Alquimia Superbuild alters its behavior to accommodate both the shared/static libraries needed by Alquimia and the native standalone executables of the engines.

### 1. CrunchFlow
Normally, Alquimia applies a custom Git patch (`MakefileForAlquimia.patch`) to the CrunchFlow source to inject the `-DALQUIMIA` flag and compile `libcrunchchem.a`. 
- **Standalone Behavior:** The Superbuild will clone a *second, unpatched* instance of the CrunchFlow repository into `external/crunchflow_standalone`. It will build this clean repository using the standard Makefiles to produce the native `crunchflow` executable. This ensures the standalone binary behaves exactly as an unmodified, upstream build would.

### 2. PFLOTRAN
PFLOTRAN relies heavily on PETSc and HDF5. 
- **Standalone Behavior:** The Superbuild will compile the `pflotran` executable from the same source tree used to generate `libpflotranchem.a` in `external/pflotran_standalone`.
- **System HDF5 Handling:** If Alquimia detects an existing system installation of HDF5 (rather than having PETSc download and build it from scratch), the standalone PFLOTRAN Makefile will fail without explicit include and library paths. The updated CMake script automatically extracts your system's `HDF5_INCLUDE_DIRS` and `HDF5_LIBRARIES`, converts them into standard `-I` and `-L` flags, and explicitly passes them into the PFLOTRAN `make` command. 

## Where to Find the Executables

Once the compilation finishes, the standalone executables will be copied to the central install directory located in your CMake binary folder:

```text
build/
└── install/
    ├── bin/
    │   ├── crunchflow
    │   └── pflotran
    ├── lib/
    │   ├── libalquimia.a
    │   ├── libcrunchchem.a
    │   └── libpflotranchem.a
    └── include/
```

You can execute them directly from `build/install/bin/` or add this directory to your system's `PATH`.