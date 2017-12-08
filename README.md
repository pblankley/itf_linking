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

Here is the reference for the novas library

Barron, E. G., Kaplan, G. H., Bangert, J., Bartlett, J. L., Puatua, W., Harris, W., & Barrett, P. (2011) “Naval Observatory Vector Astrometry Software (NOVAS) Version 3.1, Introducing a Python Edition,” Bull. AAS, 43, 2011.

Other than these packages, it is expected that the user has the usual python numerical libraries.

Once you have these installed, go to the linkitf directory of the repo and run the demo.py file.

To run the entire codebase, you will need to get two files from google drive and put them in a folder named 'data' in the linkitf directory

https://drive.google.com/drive/u/0/folders/1Dkzs4HMFHf-AaG5wrHtMsWi0zHhWsrlI

the files are itf_new_1_line.txt and itf_new_1_line_ec.mpc enter the paths for these files when you run driver.py 
