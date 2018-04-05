
## Predictive Maintenance algorithms.

This contains the algorithms modeled for my linked in post [“Executive Introduction to a Predictive Maintenance program”](https://www.linkedin.com/pulse/executive-introduction-predictive-maintenance-program-coronado/), and ["Piloting a Predictive Maintenance Program"](https://www.linkedin.com/pulse/piloting-holistic-predictive-maintenance-program-rodrigo-coronado/) -  where I provided with an executive introduction to a holistic solution to predict maintenance of field equipment and how the process can be connected to the overall supply chain. 

## Algorithms

Based on the nature of the field equipment and business needs, different forecasting methods can be used. For this example, I used a data set including equipment with 100 sensors and 3 different readings each and modeled 3 algorithms with Python, which you can find in the scripts folder - and the datasets are accesible in the data folder. 


- Remaining Usefull Life. Regression algorithms were used to forecast the remaining usefull life.
- Forecast over Window 2. Classification algorithms were used to forecast failure in Window 1. 
- Forecast for Window 1 and Window2. A Multi-classification algorithm was used to forecast miltiple classes - Window 1 and window 2.
- Libraries.py: Withing the scripts folder, the libraries.py script contains all the funcitions used to Get data, wrangle data, feature engineering, etc. 

I have added some results from testing the regression algorithm into a Power BI notebook to test an interactive visualization. This notebook can be found inthe apps folder. 

-------------------------
Note: The data set is a public dataset published by Microsoft for their Predictive Maintenance template. 
The algorithms were modeled in Python taken into consideration the R code publicy availble in this template. 
