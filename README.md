# ☀️🌧️ WEATHER PREDICTION USING MACHINE LEARNING

## Live Demo
Click [here](https://weather-prediction-using-machine-learning.onrender.com) to check out the live app.

## Description
This project is a weather prediction app that uses machine learning algorithms to classify weather conditions based on historical data.

## Installation
1. Clone this repository
   ```bash
   git clone https://github.com/Snehakethireddi/Weather-Prediction-Using-Machine-Learning.git

## Overview

This project develops a machine learning model to classify weather conditions as "Rainy" or "Sunny" based on meteorological data. The goal is to provide accurate daily weather predictions to assist in agricultural planning, resource management, and various decision-making processes that depend on weather conditions.

### Key Features:
- **Accurate Weather Predictions**: The model predicts whether the weather will be rainy or sunny, based on historical data of weather patterns.
- **Agricultural Applications**: Helps farmers plan activities like irrigation, harvesting, and crop management.
- **Real-Time Data Integration**: Future implementations may include real-time weather API data for up-to-date predictions.

## Technologies Used

### Programming Language:
- **Python**: The core programming language used for implementing the machine learning model.

### Libraries:
- **Scikit-learn**: For machine learning algorithms and tools, such as classification models (Logistic Regression, KNN, SVM, Decision Trees).
- **Pandas**: For data manipulation, cleaning, and analysis.
- **NumPy**: For numerical operations and matrix handling.
- **Matplotlib**: For data visualization, especially for plotting graphs and charts.
- **Seaborn**: For enhanced data visualization (correlation heatmaps, classification plots, etc.).

### Development Environment:
- **Jupyter Notebook**: Used for exploratory data analysis, model building, and visualization.

## Dataset

The dataset used in this project consists of historical weather data with the following features:

- **Rainfall**: Amount of rain in mm.
- **Temperature**: Temperature in Celsius.
- **Humidity**: Percentage of humidity.
- **Wind Speed**: Wind speed in km/h.

### Dataset Format:
The dataset is a CSV file containing records of weather conditions, including the target label, "Rainy" or "Sunny". 

## Methodology

### Data Preprocessing

1. **Data Cleaning**: Missing values were handled by either filling with the mean or removing rows with critical missing values.
2. **Feature Scaling**: Data was standardized to ensure that the scale of features like temperature, humidity, and wind speed did not disproportionately affect model performance.
3. **Encoding**: Categorical variables (e.g., weather labels) were encoded to numeric values for the machine learning model.

### Model Training & Evaluation

Multiple machine learning algorithms were tested to identify the best-performing model:

- **Logistic Regression**: A baseline model to classify weather conditions.
- **K-Nearest Neighbors (KNN)**: A non-parametric method that can capture complex patterns.
- **Support Vector Machine (SVM)**: A powerful classifier with high accuracy.
- **Decision Tree Classifier**: A model that provides clear decision boundaries and is interpretable.

### Performance Metrics

The models were evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The proportion of actual positive cases correctly identified by the model.
- **F1-Score**: The harmonic mean of precision and recall, used to evaluate model performance.
- **Confusion Matrix**: Provides a summary of prediction results to visualize true positives, false positives, true negatives, and false negatives.

### Hyperparameter Tuning

Model performance was optimized using:

- **GridSearchCV**: To find the best set of hyperparameters for the models.
- **Cross-Validation**: Ensures robust evaluation by splitting the data into multiple training and testing folds.

## Results

The best-performing model was identified based on the evaluation metrics. A series of visualizations were generated to highlight:

- **Model Comparison**: Comparing the accuracy, precision, recall, and F1-scores of the different models.
- **Feature Importance**: Identifying which features (e.g., rainfall, temperature) have the most significant impact on weather predictions.
- **Confusion Matrix**: Visualizing classification errors and correct predictions.

## Outcome

- Demonstrated how supervised learning can improve weather forecasting accuracy.
- The model can be applied in various domains, including:
  - **Agriculture**: Assisting farmers in planning irrigation and crop management based on predicted weather.
  - **Disaster Management**: Helping authorities plan for rainy or stormy conditions.
  - **Urban Planning**: Informing city infrastructure projects based on seasonal weather predictions.

## Future Improvements

1. **Real-Time Data**: Integrating real-time weather API data to predict weather conditions more accurately and in real time.
2. **Deep Learning**: Experimenting with more complex models such as neural networks or ensemble methods.
3. **Web Application**: Developing a user-friendly web application to allow farmers and other stakeholders to input their region and get weather predictions.

