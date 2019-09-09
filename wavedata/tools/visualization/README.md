# VTK (Visualization Toolkit) Installation Guide

## Installation with Script (Recommended)
An install script is provided, which will build and install VTK.
Manual build and installation instructions are also provided at the end of this readme.
```
cd scripts/install
```
Before running the script, change the following lines to match your environment:
```
-DPYTHON_EXECUTABLE=/usr/bin/python3.5
-DPYTHON_INCLUDE_DIR=/usr/include/python3.5m
```
For virtualenv users, it should be something like:
```
-DPYTHON_EXECUTABLE=/home/username/.virtualenvs/envname/bin/python3.5
-DPYTHON_INCLUDE_DIR=/home/username/.virtualenvs/envname/include/python3.5m
```
Run the install script
```
./vtk_install.bash
```

For virtualenv users:
```
add2virtualenv /usr/local/vtk/lib/python3.5/site-packages
```

## Demo
```
python3.5 ./demos/kitti/vtk_vis_demo.py
```
- Press any key after the OpenCV image shows up to show the point cloud and voxel grid visualization

### Camera Controls
- Rotate:
  - Left click and drag - free rotate
  - Ctrl + left click - rotate around camera normal
- Translate:
  - Middle click and drag
  - Shift + left click and drag
- Zoom:
  - Scroll wheel
  - Right click and drag
  
### Keys
- 'R' - reset the camera to fit all points on screen
- 'S' - show solid faces (default)
- 'W' - show wireframes
- 'Q' or 'E' - quit/exit

### Additional Events
- See CustomInteractorStyle defined in vtk_vis_demo.py
- Middle click and release
- Keys '1', and '2'
  - '3' is used by VTK for 3D mode

## Known Issues
- The WindowInteractor currently blocks all execution. This can probably be fixed by using a Tk window instead.
- Pycharm may not build the skeleton for VTK when using a virtualenv for the Project Interpreter,
so autocomplete functionality will not work
  - File -> Invalidate Caches/Restart... does not seem to fix this problem
  - Using /usr/bin/python3.5 instead creates the skeleton correctly

## Manual Install Steps
1. Create a new folder to hold the source and build folders
```
cd <somepath>
mkdir vtk_temp
cd vtk_temp
```
2. Clone the git repository and checkout the latest stable build v7.1.1
```
git clone git@github.com:Kitware/VTK.git
cd VTK
git checkout tags/v7.1.1
cd ..
```
3. Configure Makefile for Python 3.5 using CMake, I used cmake-gui
```
mkdir vtk_build
cd vtk_build
sudo apt-get install tk tk-dev tcl tcl-dev
sudo apt-get install freeglut3-dev
cmake-gui &
```
- Turn on Advanced Settings
- Build with flags:
  - **BUILD_TESTING**=ON
  - **VTK_WRAP_PYTHON**=ON
  - **VTK_WRAP_TCL**=ON
    - If it asks, TK_INCLUDE_DIR and TCL_INCLUDE_DIR will probably be in /usr/include/tk and /usr/include/tcl 
  - **CMAKE_INSTALL_PREFIX**=/usr/local/VTK-7.1.1
    - You can also use /usr/ which will install the libraries directly into your Python folders,
    and you can skip some later steps. However, it will be hard to upgrade or remove later.
  - **VTK_PYTHON_VERSION**=3.5
  - **PYTHON_EXECUTABLE**=/usr/bin/python3.5
    - You can also use your virtualenv python, but it doesn't matter
  - **PYTHON_INCLUDE_DIR**=/usr/include/python3.5m
    - This may not show up until you press "Configure" once
- Press "Configure" and verify settings again
- Press "Generate" to create the Makefile
4. Build and install into the folder specified with **CMAKE_INSTALL_PREFIX**
```
make -j8
sudo make install
cd Wrapping/Python
make -j8
sudo make install
```
5. Create a symlink to make upgrading easier
```
cd /usr/local
sudo ln -s /VTK-7.1.1 vtk
```
6. Add the following to your .bashrc
```
# VTK
export PATH=/usr/local/vtk/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/vtk/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/vtk/lib/python3.5/site-packages:${PYTHONPATH}
```
7. Add a .conf file to /etc/ld.so.conf.d/ to let ldconfig know where the library files are
```
cd /etc/ld.so.conf.d
sudo touch VTK-7.1.1.conf
```
- Edit VTK-7.1.1.conf to contain the following line:
```
/usr/local/vtk/lib
```
8. Run ldconfig to configure your system with the new libraries
```
sudo ldconfig
```
- Check that the vtk libraries were found
```
ldconfig -p | grep vtkCommonCore
```
9. Verify the installation
```
python3.5
>>> import vtk
>>> vtk.vtkSphereSource()
(vtkFiltersSourcesPython.vtkSphereSource)0x7f3fd5ef7b28
```
