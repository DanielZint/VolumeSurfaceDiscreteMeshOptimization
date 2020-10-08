# VSDMO

An example application for the surface discrete mesh optimization algorithm.

## Setup

### Windows

- skip for now: unpack VSDMO/vendor/installs.zip. The vendor folder should now contain four folders called "bin", "cmake", "include" and "lib".
- Open up CMake-Gui and choose VSDMO as source directory.
- Select a build folder, for example by creating a new folder `build` in VSDMO.
- Click "Add Entry" and enter `CMAKE_INSTALL_PREFIX`, select Type `Path`, the provide its value by browsing for "VSDMO/vendor/".
- Hit "Configure" and select The lastest x64 Compiler, then "Generate". Now open the Project either via "Open Project" or by double clicking the .sln file in your build directory.
- In Visual Studio, set "VSDMO" as Startup Project by right-clicking it.
- Build and run.

### Linux

none yet


## Controls

none