# HDB-resale-price
 - [Background](#Background)
 - [Problem Statement](#Problem-Statement)
 - [Data Sources](#Data-Sources)
 - [Executive Summary](#Executive-Summary)
 - [Data Dictionary](#Data-Dictionary)
 - [Recommendations](#Recommendations)
 - [Conclusion](#Conclusion)
 

## Background
![HDB Resale]([https://github.com/ethan-eplee/HorseRacePrediction/blob/main/images/horseracing.jpg](https://unsplash.com/photos/blue-and-white-concrete-building-during-daytime-DZ5yS11N9CA))

From 2012 to 2021, Housing and Development Board (HDB) resale prices in Singapore experienced significant fluctuations due to various policy changes, market demand, and economic factors. In 2013, the Singapore government implemented several cooling measures, including the Total Debt Servicing Ratio (TDSR) framework and a reduction in the Mortgage Servicing Ratio (MSR) for HDB loans, which led to a decline in resale prices from 2014 to 2017. However, from 2018 onwards, the HDB resale market began to recover, driven by factors such as a stable economy and increased demand for public housing.

By 2020, the COVID-19 pandemic had created unique pressures on the market, including delays in the construction of new Build-to-Order (BTO) flats, which drove more buyers to the resale market. Consequently, 2020 and 2021 saw a surge in HDB resale prices, with prices reaching record highs in some areas.

## Problem Statement
Estate agents face several challenges in this dynamic market. These include accurately pricing properties in a fluctuating market, managing client expectations amid rising prices, and navigating the complexities of government regulations. Additionally, there is a growing trend for new homeowners to prioritise purchasing homes near top schools, further intensifying competition in certain areas, which adds another layer of complexity for agents trying to meet client demands.

![BetLowest](https://github.com/ethan-eplee/HorseRacePrediction/blob/main/images/bet_lowest_odds.jpg)

And so the problem we want to address here is: 

**Can we use machine learning to make predictions for resale prices?**

We will follow the data science process to answer this problem.
1. Define the problem
2. Gather & clean the data
3. Explore the data
4. Model the data
5. Evaluate the model
6. Answer the problem

--- 
## Data Sources
The dataset contains HDB resale data from March 2012 to April 2021. The dataset contains 150,634 rows and 78 columns. The data can be downloaded from Google Drive at [this link](https://drive.google.com/file/d/1KnT9XkEfdUG2GSIzglD-0GyegZJ0WNSA/view?usp=sharing).

The data dictionary will be provided at the bottom of this file.

---
## Executive Summary
**INTRODUCTION**

This project seeks to make predictions on the outcome of HDB resale prices through regression models. For regression models, we aim to predict the finish time of the horses, hereby predicting the winner of the race.

With the prediction results, agents will be able to use different strategies to meet their client's needs. Backtesting results of each model will also show the number of bets and profit made from each strategy.

**METHODOLOGY**

The work was done in 7 seperate notebooks.
1. Preprocessing - Cleaning and tidying of data, Feature Engineering
2. EDA - Data visualisation and analysis of key patterns
3. Regression Modelling - Testing dataset fitted on 3 models to get regression predictions
5. Evaluation - Evaluation of results, Feature Importance
6. Backtesting - Using test data to answer the problem statement of whether we can predict resale prices
7. Deployment - To build an application using Streamlit, where agents can key in simplified inputs to get a prediction on the resale price.

The application was deployed on Streamlit and can be accessed through this [link](TO BE INCLUDED). A screenshot showing the app is shown below. Please note that this app was only intended as an educational and demo tool, and not meant to be used for real life house planning.

![App](https://github.com/ethan-eplee/HorseRacePrediction/blob/main/images/app.jpg)


**SIGNIFICANT FINDINGS**

After evaluating the 13 variables using PyCaret to identify the top models based on their train and test RMSE, we narrowed down our selection to three models: Light Gradient Boosting Machine, Extreme Gradient Boosting Machine and CatBoost.

RMSE (Root Mean Square Error) is a valuable metric for evaluating the quality of a regression model. It gives you insight into how well the model's predictions match the actual values. 

We also consider another factor which is the time spent to run each model. Below is the table showing all RMSE and Run time




![Feature Importance](https://github.com/ethan-eplee/HorseRacePrediction/blob/main/images/shap_features.jpg)

The SHAP summary plot showed that lower values of a horse's recent ranks contributed to a higher probability of the horse finishing top. The quality of the jockey, as shown by his recent ranks, also play a big role in determinining if a horse will win.

In the backtesting phase, we ran our model predictions through different strategies, and found that 7 out of 8 models actually returned a positive value. In the notebook, we ran a few strategies, but the simplest ones were:
1. Bet $1 when model that predicted a horse will win the top position
2. Bet $1 on the horse that model predicted with the fastest time during a race

A summary of the backtesting results when ran on these two strategies are shown below.

|                **Model** 	| **Money** 	| **Bets Made** 	|
|-------------------------:	|----------:	|--------------:	|
|               SMOTE + RF 	|     375.2 	|           743 	|
| Random Forest Classifier 	|     268.1 	|            68 	|
|      Logistic Regression 	|      23.0 	|            32 	|
|     Gaussian Naive Bayes 	|      10.7 	|           177 	|
|         Ridge Regression 	|     360.4 	|           480 	|
|                     LGBM 	|     307.3 	|           617 	|
|           KNN Regression 	|     237.6 	|           480 	|
|  Random Forest Regressor 	|     -48.1 	|           542 	|

---




## Data Dictionary

There are two datasets obtained from Kaggle, courtesy of the Hong Kong Jocket Club. The first is the related to the horse and the the second is related to the race. Both of these tables can be joined on the race_id column.

| Columns               	| Description                                                                                                                                                                            	|
|-----------------------	|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| finishing_position    	| The rank of the horse. (E.g. the horse with finishing_position 1 is the first to finish)                                                                                               	|
| horse_number          	| The number for the horse in the specific race. (Note that the same horse may have different numbers in different races)                                                                	|
| horse_name            	| English name of the horse                                                                                                                                                              	|
| horse_id              	| ID of the horse. (The ID for a horse is unique in all the races)                                                                                                                       	|
| jockey                	| The one who rides the horse in the race. (A jockey can ride different horses in the races)                                                                                             	|
| trainer               	| The one who trains the horse. (Multiple horses from a trainer can appear in the same race)                                                                                             	|
| actual_weight         	| The extra weight that a horse carries in the race. (The horses with better performances in the previous races will carry extra weights to make the race more competitive)              	|
| declared_horse_weight 	| The weight of the horse on date of the race.                                                                                                                                           	|
| draw                  	| The position of the horse at the starting point. The inner positions are usually advantageous and correspond to smaller draw numbers.                                                  	|
| length_behind_winner  	| The length behind the winner at the finish line. The unit is “horse length”.                                                                                                           	|
| running_position_i    	| The rank of the horse at the i-th timing point. (The running position will be “NA” if the total distance of the race is short and the horses do not cross the particular timing point) 	|
| finish_time           	| The total time from the starting point to the finish line. The unit is in seconds.                                                                                                     	|
| win_odds              	| The multiplier of the amount you bet to be received if you win. THe odds are usually determined automatically by the total money bet on each horse.                                    	|
| race_id               	| The ID of the race for this entry. The race_id is consistent in the two data files.                                                                                                    	|
| race_distance         	| The race distance in metres for each race                                                                                                                                              	|

---
## Recommendations 
- Backtesting results were good, but I cannot be sure if they are reflective of real life horse racing.
- One of the drawbacks of the backtesting is that the races were all treated as if on the same starting ground. In reality, results from a race would have to be updated into the model, for retraining, to predict the results of the next race. Due to time constraints, we have simplified the problem and saved ourselves time and effort to retrain the model multiple times.
- I treated one row of data as one sample. I perhaps should have treated all rows of a unique race as one sample as we want to see which horse can win relative to its race opponents.
- I am unable to make a prediction if the horse is new and has not raced before.
- Try out more complicated models to see how they fare?

---
## Conclusion
Overall, we were able to get a good result from the models and backtesting. Most of the models and strategies, though simplistic, allowed us to "profit" over the course of 2,000 races. I am convinced that using one of these statistical models would give us an edge over the average punter, but of course I would have to test this out in real life to prove it!
