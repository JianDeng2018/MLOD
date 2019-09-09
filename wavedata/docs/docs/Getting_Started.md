# Wavedata
Wavedata is a library of dataset helper functions 
implemented in python. The aim of this library is to provide functions for 
data input/output of the different components of the dataset, as well as
functions for baseline evaluation on different autonomous driving tasks. The functions are currently compatible with
the Kitti dataset.
  
# Getting Started
Implemented and tested on Ubuntu 16.04.

1.Install Python 3.5 and Pip3: 

```
sudo apt-get install python3.5
sudo apt-get install python3-pip
```

2.Clone Repository:

```
git clone git@github.com:wavelab/wavedata.git
```
3.Install Dependencies:

```
cd ~/wavedata/
sudo pip3 install -r requirements.txt
```
   
4.Install opencv: 

For opencv2:

   ```
   sudo apt-get install python-opencv 
   ```
   
For opencv3:

   ```
   sudo apt-get update
   sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
   mkdir OpenCV
   cd OpenCV
   git clone http://github.com/opencv/opencv.git
   cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ../opencv-3
   make -j $(nproc --all)
   sudo make install
   ```
5.Add wavedata to python path: 
```
echo export PYTHONPATH="${PYTHONPATH}: path_to_wavedata". >> ~/.bashrc 
```

# Setting up Pylint for Pycharm
Pycharm notifies you about PEP 8 errors, however it still misses out on checking coding standards. The following section will show you how to setup pylint.

1. Install pylint
```
pip3 install pylint (python 3.5)
pip install pylint (python 2.7)
```

2. Open Pycharm and go to File/Settings or Ctrl+Shift+S.

3. Set up an external tools by clicking the plus sign.
![setup](/images/external_tools.png)

4. Configure the external tool as such so that you can run pylint for the current active file.
![config](/images/edit_tool.png)

5. To run pylint, simply go to Tools/External Tools and run the configured Pylint. If configured properly, you should be seeing something like this.
![running](/images/linting.png)
