# Coursework-2
Coursework 2 for ACT module

- Explain project idea
- Outline structure of project and how to run it
- Include what Python dependancies are needed

&nbsp;
## Chosen dataset
For this project, the dataset chosen was the *Kepler Exoplanet Search Results* dataset from [kaggle.com](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results). This choice was motivated largely by the intermediate size of the dataset, at around 10,000 rows, large enough to avoid overfitting whilst remaining small enough to be   
The Kaggle page lists the following as inspiration for projects with the dataset: "How often are exoplanets confirmed in the existing literature disconfirmed by measurements from Kepler? How about the other way round?"; "What general characteristics about exoplanets (that we can find) can you derive from this dataset?"; and "What exoplanets get assigned names in the literature? What is the distribution of confidence scores?".  
  
**-> What do I want to explore in the dataset?**
- effect of planetary radius on koi score?
- estimating values for objects missing data


The dataset's column headings are quite obscure, so the NASA Exoplanet Archive definitions for the headings are given [here](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html).  

&nbsp;
## Project Idea
How often are exoplanets confirmed in literature disconfirmed by Kepler's measurements? To find out, we would need to compare KOI disposition (candidacy disposition from Exoplanet Archive) to KOI pdisposition (disposition from Kepler data analysis).   
&nbsp;
## Structure of this project
This project is structed as follows: README (this file) and  license information, a csv file containing the dataset, and a folder containing a functions.py file for useful functions, and three folders for parts 1, 2, and 3 of the project.  
Part 1 (Q1) focusses on a traditional approach to the problem, using a non neural network method, and explains why that method was chosen and describes how well the traditional method works.  
Part 2 (Q2) repeats Part 1, except this time a neural network approach is used, and then compares this approach to the traditional approach.  
Part 3(Q3) explores a given 'research question' with the neural network and dataset. Choice of the question is expanded in the file for Q3.
```
│   dependencies.txt
│   KESR.csv
│   LICENSE
│   README.md
│   
└───py
    │   functions.py
    │   
    ├───Q1_folder
    │       q1.ipynb
    │
    ├───Q2_folder
    │       q2.ipynb
    │
    └───Q3_folder
            q3.ipynb
```

&nbsp;
## How to run this project
Firstly, the required Python modules must be installed. To do so, download [dependencies.txt](https://github.com/GHancock1/Coursework-2/blob/main/dependencies.txt) and copy the following into your terminal: ```pip install -r dependencies.txt```, or alternatively work through the file and esure all required modules are installed.  
Start with the Q1 file, found in ```/py/Q1_folder/q1.ipynb```. Read through each cell and run code if you wish. 

&nbsp;
## Useful information
Any reference to 'koi' or 'KOI' in the project referes to a Kepler Object of Interest

