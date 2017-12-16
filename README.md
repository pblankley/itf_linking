# itf_linking
Code base to take a version of the MPC's ITF and find the tracklets that relate to the same asteroid.

Install instructions:

Note: This project uses *Python 3*.
Clone this repo with the following commands

$ git clone https://github.com/pblankley/itf_linking.git

Once you have the repo cloned, there are a few libraries you will need to Install

Run the following commands from the command line if you do not have these packages installed

$ pip install novas

$ pip install novas_de405

$ conda install healpy
OR 
$ pip install --user healpy

Here is the reference for the novas library

Barron, E. G., Kaplan, G. H., Bangert, J., Bartlett, J. L., Puatua, W., Harris, W., & Barrett, P. (2011) “Naval Observatory Vector Astrometry Software (NOVAS) Version 3.1, Introducing a Python Edition,” Bull. AAS, 43, 2011.

Other than these packages, it is expected that the user has the usual python numerical libraries.

Once you have these installed, go to the linkitf directory and look at the two notebooks entitled orbit_fitting.ipynb and plots_solver.ipynb. These two notebooks give a rundown of some of the basic commands from our module, methods we used to arrive at answers, and results (results are in orbit_fitting.ipynb)

To run the entire codebase, you will need to get two files at minimum from Google Drive.  If you want to run the entire training dataset, go to the folder below and get this file:  data/train/UnnObs_Training_1_line_A_ec.mpc
If you want to run the ITF go to the folder below and get this file: data/train/itf_new_1_line_ec.mpc

Once you have the file you need (for either the ITF or training or both) put it in a folder of your choice, and go into driver.py, change the path to the training or ITF file (whichever applies in your case) to the path of where you put it, and follow the instructions in the driver file.

If you want to run the orbit_fitting notebook using the precomputed training files, just grab the entire data/ folder from the below Google Drive folder and put it in the linkitf/ folder of the repo. The notebook should run without a hitch. 

Google Drive Folder:
https://drive.google.com/drive/u/0/folders/1wYIXQePMZlOSC4czXHbUhU-__E1dGQC4

Once you follow the steps above and comment in the statements you need to in driver.py (according to the instructions), just run the driver.py file and your progress files will be saved in the same folder you put the main file. 
The driver.py file will also keep a running status on the jobs in a percent complete display.
