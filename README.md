# COMP-6721---AI-ducation-analysis-project
## Overview
This repository contains the materials for the Part 1 submission of our AI-ducation analysis project. The key components included are:

- **Report_AI**: This text document provides a comprehensive overview of all three parts of our project, including all the necessary details.

- **Dataset**: This text document includes information about all categories of the dataset, their sources, and sample images.

- **Originality_form**: This form attests to the originality of our work and bears the signatures of all team members.

- **DataCleaning.py**: This Python file contains the complete code for Part 1 of the project. It encompasses both data cleaning and data visualization. To successfully execute the code, a Python interpreter is required, along with the following packages installed:
  - `numpy`
  - `torchvision`
  - `PIL`
  - `Matplotlib`

  The code assumes that the paths to the datasets are clearly specified. After running the code, a folder named "resize" is automatically generated, containing the cleaned images. Additionally, the code will display the following visualizations:
  - Histogram for pixel intensity
  - 5x5 image grid
  - Bar chart showing the distribution of images in each class

  You need to wait a moment for the visualization to be displayed. After closing the images the labels of the duplicates would be displayed.

-**BiasAnalysis.py**: This python file is to check the bias in the dataset based on 2 categories i.e., age and gender.

-**Training.py**: This python file is to develop a convolutional neural network architecture for main model and two variants.

-**FinalModel.py**: This python file is to perform KFold Cross validation on the dataset.

