# Bird-Species-Classification
Classifies 200 species of birds based on the images. 

## Dataset used - 

Caltech-UCSD Birds-200-2011 
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

## Instructions

1)First download the dataset and extract the folder CUB_200_2011 in the directory

2)Then run the script **prepare_data.py** to create a .npy file

**Note: The script may take some time to run and will take a lot of computation power. So please close all the running applications and 
be patient.**

3) Then run train.py to train the model.

## Citations

@techreport{WahCUB_200_2011,
	Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
	Year = {2011}
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2011-001}
}
