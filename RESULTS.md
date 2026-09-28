**House price prediction**
The house price prediction model ran with 240 samples in the test data. Mean absolute error was approximately $27,481 and R2 score around 0.917. The predicted value for this Downtown 2,000 sqft house is approximately $401,135 but this is not a guaranteed sale price. 


**Code output**
_Records: 240
MAE: $27,481.25
R2: 0.917
Predicted price for a 2,000 sq ft Downtown house: $401,135.39_

**Customer churn prediction**
For the customer churn prediction model, there were 300 records used for testing, resulting in test accuracy of 0.7 and ROC AUC of 0.731. This customer is at risk of churn with a predicted churn probability of 96.6% (above 0.5). A business could use this to decide to send out a retention offer or pick up a phone for a customer like this for example.


**Code output**
_Records: 300
Accuracy: 0.700
ROC AUC: 0.731
New customer churn probability: 96.6%
At-risk classification: 1_

**Customer segmentation**
This model used 240 records in the data set and produced three clusters after spending, frequency of purchases, age and location of customers had been scaled. A summary of the resulting groups and an image of the elbow plot for determination of number of clusters can be found in the files above.
**Code output**
_Records: 240
Clusters used: 3

         annual_spending  purchase_frequency    age           suggested_strategy
cluster
0                 905.62                6.28  33.27  Use introductory promotions
1                2608.16               18.17  48.07        Offer premium rewards
2                 747.80                5.25  59.29  Use introductory promotions_

**Optional housing-demand forecast**
The model was trained with 120 monthly records. The resulting model is a simple trend forecast. It does not attempt to model seasonality, interest rates, housing inventory, etc.


**Code output**
_Historical records: 120
Training MAE: 7.99
Six-month forecast:
     month  predicted_demand
2033-01-01             460.4
2033-02-01             463.4
2033-03-01             466.4
2033-04-01             469.5
2033-05-01             472.5
2033-06-01             475.5_
