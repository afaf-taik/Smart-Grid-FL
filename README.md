# Smart-Grid-FL

Source code used for papers "Electrical load forecasting using edge computing and federated learning" and " Empowering Prosumer Communities in Smart Grid with Wireless Communications and Federated Edge Learning".
 
Code works with data from PecanStreet Inc. Since we cannot provide the data, users can contact https://www.pecanstreet.org/ to get licence necessary to access data. 

In order to use the code run the makedatasets.py file first in the same folder as the data to transform the csv data to .h5 data

The federatedProcess.py file trains and saves the global model

The personalization process retrains the global model for personalization
