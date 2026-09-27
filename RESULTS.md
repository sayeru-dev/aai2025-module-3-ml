# Successful Output and Interpretation

## House price prediction

The model used 240 records. The test mean absolute error was about $27,481 and the R2 score was 0.917. The predicted price for a 2,000 square foot Downtown house was about $401,135. The result means the model used square footage and location to estimate price, but the estimate is not a guaranteed sale price.

## Customer churn prediction

The model used 300 records. Test accuracy was 0.700 and ROC AUC was 0.731. The sample customer had a 96.6% predicted churn probability and was classified as at risk because the probability was above 0.5. A business could use this result to prioritize a retention offer or customer-service follow-up.

## Customer segmentation

The model used 240 records and created three clusters after scaling spending, purchase frequency, age, and region. The saved summary and elbow plot help compare the groups. The highest-spending group received a premium-rewards strategy, while lower-spending groups received an introductory-promotion strategy.

## Optional housing-demand forecast

The model used 120 monthly records and forecast the next six months. This is a basic trend forecast; it does not model seasonality, interest rates, inventory, or other market variables.
