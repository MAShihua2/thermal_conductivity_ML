README:

Machine Learning for Thermal Conductivity Prediction based on US (SOAP) and UCSRO

This repository contains the machine learning code used in our study to predict the thermal conductivity of materials based on their structural and chemical properties. The code utilizes various machine learning techniques, including encoding local environments with unique local structures (US) and chemical short-range orders (UCSRO), and predicting properties using neural networks.

Description
The project applies a machine learning framework to Non-Equilibrium Molecular Dynamics (NEMD) data to predict thermal conductivity. It includes preprocessing of data, application of K-Means for clustering, and implementation of a neural network model for predictions.

Features
* Data Encoding: Encoding the local environment of atoms using SOAP descriptors and CSRO parameters.
* Clustering: Segmenting the data into clusters using the K-Means algorithm to manage extensive datasets.
* Neural Network: A four-layer neural network architecture designed for the prediction of thermal conductivity.
* Shapley Value Analysis: Using Python SHAP library to understand the contribution of US and UCSRO to the model's predictions.

Installation
see the project environment in env.txt

Citation
If you use this code or the associated data in your research, please cite it as follows:
Shihua Ma, Yaoxu Xiong, Jun Zhang, Shasha Huang, Shijun Zhaoa. Tunable Thermal Transport Properties in High-Entropy Alloys through Manipulating Chemical Short-Range Order Guided by Machine Learning (To be submited)