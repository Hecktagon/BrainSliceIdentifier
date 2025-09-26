# **Brain Slice Identifier**  
## Description:  
Given measurements and an image of a slice of a mouse brain, SliceIdentifier.py will use AI image comparison as well as vector math to identify known brain slices from a reference database that most closely resemble the given slice.

# *Setup:*
### 1. Pulling from GitHub:  
  - Download/Open a shell like GitBash for Windows, or Terminal for Mac. (If your shell doesn't come with Git, you may need to download it from the Git website.)
  - Navigate in your file tree to wherever you would like to store the project.
  - Use this command to clone the project onto your computer from github:  
    `git clone https://github.com/Hecktagon/BrainSliceIdentifier`

### 2. Installing Dependancies:  
  - in a terminal ensure you are in the correct file location (should end in 'BrainSliceIdentifier')
  - Use the following command to install all needed dependancies:
    `pip install -r requirements.txt`

### 3. Running the Program:  
  - Run SliceIdentifier.py, you will be prompted for the following inputs:
  - `filepath`: paste the path to the image you want to check (crop image to just the hippocampus)
  - `retrain AI (y/n)`: y retrains the AI on the reference images, not needed unless changes were made to the reference folder
  - `hippocampus height`, `hippocampus width`, and `whole brain height`: provide appropriate measurements
  - `weight multiplier`: a scalar multiplier that changes how heavily measurements affect the matching process, as opposed to AI image matching
