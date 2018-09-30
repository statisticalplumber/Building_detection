# Building_detection
This project is to identify buildings in satellite using Unet and masking method

To execute this project, please follow the below guidlines
1: run_code.py is the main file
2: save images and polygon files in images and polygon folders under data
3: ProcessFile class is invoked in run_code that creates csv files of images and masks with user define pixel values like 128, 256 etc.
4: Processed image csv and mask csv files are used in main code to build model 
