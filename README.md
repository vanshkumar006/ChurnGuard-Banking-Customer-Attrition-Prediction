ChurnGuard: Banking Customer Attrition Prediction
ChurnGuard is an end-to-end Machine Learning pipeline that identifies customers likely to churn from credit card services. By leveraging advanced gradient boosting and robust preprocessing, the system provides high-precision warnings to help financial institutions improve retention rates.

📊 Performance Summary
Model	ROC-AUC	PR-AUC	F1-Score
XGBoost	0.9932	0.9691	0.9044
Logistic Regression	0.9198	0.7328	0.6424

🛠️ Key Features
Automated Data Cleaning: Removes ID features and handles data leakage by stripping pre-calculated Naive Bayes variables.
Class Imbalance Handling: Implements SMOTE for linear models and Weighted Loss Functions for tree-based models to ensure minority class (Churn) detection.
Predictive Insight Extraction: Generates feature importance visualizations to show why customers are leaving.
Stratified Validation: Uses stratified splitting to maintain class proportions across training and testing sets.

📈 Key Insights
According to the XGBoost model, the top 3 drivers of customer attrition are:
Total_Trans_Ct: Decreasing transaction frequency is the strongest signal of impending churn.
Total_Revolving_Bal: Customers with lower revolving balances are statistically more likely to close their accounts.
Total_Relationship_Count: Customers with fewer products/links to the bank churn at a higher rate.
