This is the continuation of the “Executive Introduction to a Predictive Maintenance program” post, where I provided with an executive introduction to a holistic solution to predict maintenance of field equipment and how the process can be connected to the overall supply chain. In this post I will go elaborate on the components of the overall solution, starting with the Machine Learning algorithms used to forecast failure, present the business insights, and show how to automate back office processes including the creation work orders, alerting maintenance decision makers and coordinating a maintenance crew.

Algorithms
Based on the nature of the field equipment and business needs, different forecasting methods can be used. For this example, I used a data set including equipment with 100 sensors and 3 different readings each and modeled 3 algorithms based on different business needs. 
For this blog I have used 3 algorithms which you can find in the scripts folder - and the datasets are accesible in the data folder. 


- Remaining Usufull Life. Regression algorithms were used to forecast the remaining usefull life.
- Forecast over Window 2. Classification algorithms were used to forecast failure in Window 1. 
- Forecast for Window 1 and Window2. A Multi-classification algorithm was used to forecast miltiple classes - Window 1 and window2.


-------------------------
Note: The data set is a public dataset publish by Microsoft for their Predictive Maintenance template. 
The algorithms where modeled in Python, taken into consideration the R code publicy availble in this template. 