# HDB-resale-price
 - [Background](#Background)
 - [Problem Statement](#Problem-Statement)
 - [Data Sources](#Data-Sources)
 - [Executive Summary](#Executive-Summary)
 - [Data Dictionary](#Data-Dictionary)
 - [Feature Selection & Justification](#Feature-Selection-&-Justification)
 - [Recommendations](#Recommendations)
 - [Conclusion](#Conclusion)
 

## Background
![HDB Resale](images/anatoliy-shostak-DZ5yS11N9CA-unsplash.jpg)

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

This project seeks to make predictions on the outcome of HDB resale prices through regression models. With the prediction price, agents will be able to use different strategies to meet their client's needs. 

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

Based on a context of structured tabular dataset, we managed to filter out the three strongest options for modeling which are Light Gradient Boosting Machine (LightGBM), CatBoost and Extreme Gradient Boosting Machine (XGBoost).

We further analysed performance of three machine learning models using metrics such as Train RMSE, Test RMSE, Train R², Test R² and Run Time (s). 
Here’s the analysis:

1. Light Gradient Boosting Machine
Train RMSE: 23,594
Test RMSE: 23,699
Train R²: 0.97
Test R²: 0.97
Run Time: 50 seconds
Analysis:The LightGBM model shows strong performance with a low RMSE (Root Mean Square Error) on both the training and test datasets, indicating good predictive accuracy.The R² (coefficient of determination) of 0.97 is high, signifying that the model explains 97% of the variance in the data.
The run time of 50 seconds suggests that the model is relatively fast.

2. CatBoost
Train RMSE: 25,359
Test RMSE: 25,130
Train R²: 0.97
Test R²: 0.97
Run Time: 72 seconds
Analysis:CatBoost also performs well, with slightly higher RMSE values compared to LightGBM. However, the difference is minimal, and the model still maintains high accuracy.The R² is consistent at 0.97, indicating that it explains 97% of the data’s variance, similar to LightGBM.
The run time is 72 seconds, making it slower than LightGBM but still within a reasonable range.

3. XGBoost
Train RMSE: 22,924
Test RMSE: 22,906
Train R²: 0.97
Test R²: 0.97
Run Time: 38 seconds
Analysis: XGBoost has the lowest RMSE among the three models, indicating the best predictive performance on both the training and test datasets.
The R² is consistent at 0.97, similar to the other models, explaining 97% of the variance in the data.
The run time is the fastest at 38 seconds, making XGBoost not only the most accurate but also the most efficient in terms of computational speed.

Overall Summary
LightGBM is a strong contender with very close performance metrics to XGBoost and a slightly longer run time.
CatBoost is still a solid model, particularly when we have a lot of categorical data, but it lags slightly behind in terms of RMSE and run time.
XGBoost outperforms the other two models in terms of both accuracy (lowest RMSE) and efficiency (shortest run time). It is our final model of choice since we prioritize predictive performance and computation time.


## Data Dictionary

1. **resale_price**: the property's sale price in Singapore dollars. This is the target variable that you're trying to predict for this challenge.
2. **Tranc_YearMonth**: year and month of the resale transaction, e.g. 2015-02.
3. **town**: HDB township where the flat is located, e.g. BUKIT MERAH.
4. **flat_type**: type of the resale flat unit, e.g. 3 ROOM.
5. **block**: block number of the resale flat, e.g. 454.
6. **street_name**: street name where the resale flat resides, e.g. TAMPINES ST 42.
7. **storey_range**: floor level (range) of the resale flat unit, e.g. 07 TO 09.
8. **floor_area_sqm**: floor area of the resale flat unit in square metres.
9. **flat_model**: HDB model of the resale flat, e.g. Multi Generation.
10. **lease_commence_date**: commencement year of the flat unit's 99-year lease.
11. **Tranc_Year**: year of resale transaction.
12. **Tranc_Month**: month of resale transaction.
13. **mid_storey**: median value of storey_range.
14. **lower**: lower value of storey_range.
15. **upper**: upper value of storey_range.
16. **mid**: middle value of storey_range.
17. **full_flat_type**: combination of flat_type and flat_model.
18. **address**: combination of block and street_name.
19. **floor_area_sqft**: floor area of the resale flat unit in square feet.
20. **hdb_age**: number of years from lease_commence_date to present year.
21. **max_floor_lvl**: highest floor of the resale flat.
22. **year_completed**: year which construction was completed for resale flat.
23. **residential**: boolean value if resale flat has residential units in the same block.
24. **commercial**: boolean value if resale flat has commercial units in the same block.
25. **market_hawker**: boolean value if resale flat has a market or hawker centre in the same block.
26. **multistorey_carpark**: boolean value if resale flat has a multistorey carpark in the same block.
27. **precinct_pavilion**: boolean value if resale flat has a pavilion in the same block.
28. **total_dwelling_units**: total number of residential dwelling units in the resale flat.
29. **1room_sold**: number of 1-room residential units in the resale flat.
30. **2room_sold**: number of 2-room residential units in the resale flat.
31. **3room_sold**: number of 3-room residential units in the resale flat.
32. **4room_sold**: number of 4-room residential units in the resale flat.
33. **5room_sold**: number of 5-room residential units in the resale flat.
34. **exec_sold**: number of executive type residential units in the resale flat block.
35. **multigen_sold**: number of multi-generational type residential units in the resale flat block.
36. **studio_apartment_sold**: number of studio apartment type residential units in the resale flat block.
37. **1room_rental**: number of 1-room rental residential units in the resale flat block.
38. **2room_rental**: number of 2-room rental residential units in the resale flat block.
39. **3room_rental**: number of 3-room rental residential units in the resale flat block.
40. **other_room_rental**: number of "other" type rental residential units in the resale flat block.
41. **postal**: postal code of the resale flat block.
42. **Latitude**: Latitude based on postal code.
43. **Longitude**: Longitude based on postal code.
44. **planning_area**: Government planning area that the flat is located.
45. **Mall_Nearest_Distance**: distance (in metres) to the nearest mall.
46. **Mall_Within_500m**: number of malls within 500 metres.
47. **Mall_Within_1km**: number of malls within 1 kilometre.
48. **Mall_Within_2km**: number of malls within 2 kilometres.
49. **Hawker_Nearest_Distance**: distance (in metres) to the nearest hawker centre.
50. **Hawker_Within_500m**: number of hawker centres within 500 metres.
51. **Hawker_Within_1km**: number of hawker centres within 1 kilometre.
52. **Hawker_Within_2km**: number of hawker centres within 2 kilometres.
53. **hawker_food_stalls**: number of hawker food stalls in the nearest hawker centre.
54. **hawker_market_stalls**: number of hawker and market stalls in the nearest hawker centre.
55. **mrt_nearest_distance**: distance (in metres) to the nearest MRT station.
56. **mrt_name**: name of the nearest MRT station.
57. **bus_interchange**: boolean value if the nearest MRT station is also a bus interchange.
58. **mrt_interchange**: boolean value if the nearest MRT station is a train interchange station.
59. **mrt_latitude**: latitude (in decimal degrees) of the nearest MRT station.
60. **mrt_longitude**: longitude (in decimal degrees) of the nearest MRT station.
61. **bus_stop_nearest_distance**: distance (in metres) to the nearest bus stop.
62. **bus_stop_name**: name of the nearest bus stop.
63. **bus_stop_latitude**: latitude (in decimal degrees) of the nearest bus stop.
64. **bus_stop_longitude**: longitude (in decimal degrees) of the nearest bus stop.
65. **pri_sch_nearest_distance**: distance (in metres) to the nearest primary school.
66. **pri_sch_name**: name of the nearest primary school.
67. **vacancy**: number of vacancies in the nearest primary school.
68. **pri_sch_affiliation**: boolean value if the nearest primary school has a secondary school affiliation.
69. **pri_sch_latitude**: latitude (in decimal degrees) of the nearest primary school.
70. **pri_sch_longitude**: longitude (in decimal degrees) of the nearest primary school.
71. **sec_sch_nearest_dist**: distance (in metres) to the nearest secondary school.
72. **sec_sch_name**: name of the nearest secondary school.
73. **cutoff_point**: PSLE cutoff point of the nearest secondary school.
74. **affiliation**: boolean value if the nearest secondary school has a primary school affiliation.
75. **sec_sch_latitude**: latitude (in decimal degrees) of the nearest secondary school.
76. **sec_sch_longitude**: longitude (in decimal degrees) of the nearest secondary school.

---
## Feature Selection & Justification

---
## Reflections
We want to reflect on key stages in our team work process:
1. Data Handling
- **Cleaning & Feature Engineering: The team effectively cleaned the data and engineered key features, like flat_type and hdb_age, which were crucial for accurate predictions.
- **Variable Selection: Relevant variables were carefully chosen, including handling categorical data properly and exploring interactions, improving model accuracy.
2. Model Fine-Tuning
- **Hyperparameter Tuning: The team used grid and random search techniques to optimize hyperparameters, ensuring models were finely tuned for performance.
- **Cross-Validation: Cross-validation helped prevent overfitting and ensured the models generalized well.
- **Technical Challenges: Challenges like balancing model complexity with run time and handling large datasets were tackled effectively, ensuring efficient model training.
Summary
The team’s strong approach to data handling, variable selection, and model fine-tuning, combined with overcoming technical challenges, led to the development of accurate and reliable models for predicting HDB resale prices.
---
## Conclusion

In this project, we aimed to predict HDB resale prices in Singapore by leveraging a comprehensive dataset that included various factors such as the property's location, type, size, age, and proximity to amenities. Through careful data handling, feature engineering, and model selection, we successfully identified key variables that significantly impact resale prices. By comparing advanced machine learning models—LightGBM, XGBoost, and CatBoost—we were able to fine-tune and validate our models to achieve high accuracy and reliability. The results demonstrate the effectiveness of these models in capturing the complexities of the HDB resale market, providing valuable insights for stakeholders and setting the stage for future enhancements and potential real-world applications.
