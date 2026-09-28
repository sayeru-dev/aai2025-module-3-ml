#House price prediction
The house price prediction model ran with 240 samples in the test data. Mean absolute error was approximately $27,481 and R2 score around 0.917. The predicted value for this Downtown 2,000 sqft house is approximately $401,135 but this is not a guaranteed sale price. 

#Customer churn prediction
For the customer churn prediction model, there were 300 records used for testing, resulting in test accuracy of 0.7 and ROC AUC of 0.731. This customer is at risk of churn with a predicted churn probability of 96.6% (above 0.5). A business could use this to decide to send out a retention offer or pick up a phone for a customer like this for example.

#Customer segmentation
This model used 240 records in the data set and produced three clusters after spending, frequency of purchases, age and location of customers had been scaled. A summary of the resulting groups and an image of the elbow plot for determination of number of clusters can be found in the files above.

#Optional housing-demand forecast
The model was trained with 120 monthly records. The resulting model is a simple trend forecast. It does not attempt to model seasonality, interest rates, housing inventory, etc.
