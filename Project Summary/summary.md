# INX Future Inc. Employee Performance Prediction

## 1. Project Overview

The aim of this project was to analyse employee data from INX Future Inc., understand the patterns related to employee performance, and develop a machine learning model to predict performance ratings.

I used Python to clean and explore the data, prepare features, train classification models, and evaluate their performance.

## 2. Work Completed

The project covered the following activities:

- Checked the dataset for missing values, duplicate records, and potential outliers.
- Performed exploratory data analysis to understand employee performance patterns.
- Created the `ExperienceBeforeCurrentCompany` feature.
- Prepared numerical and categorical features for machine learning.
- Trained and compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models.
- Used GridSearchCV to tune the Gradient Boosting model.
- Analysed feature importance and saved the final model for future predictions.

## 3. Key Findings

The analysis identified several patterns related to employee performance ratings.

Employee environment satisfaction and salary hike percentage had the strongest positive correlations with performance rating among the numerical features examined. Years since the last promotion had a negative correlation with performance rating.

The feature importance analysis also identified employee environment satisfaction, salary hike percentage, and years since the last promotion as the three most important features used by the final Gradient Boosting model.

These findings describe relationships in the available data. They do not prove that these factors directly cause changes in employee performance.

## 4. Final Model Results

I compared four classification models using all available predictor features.

The tuned Gradient Boosting model achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 92.92% |
| Macro Precision | 0.9252 |
| Macro Recall | 0.8556 |
| Macro F1-score | 0.8866 |

Gradient Boosting achieved the highest accuracy and macro F1-score among the four models evaluated using all predictor features.

Hyperparameter tuning produced a small improvement in macro F1-score compared with the original Gradient Boosting model.

## 5. Recommendations

Based on the analysis, I recommend the following:

- Review employee environment satisfaction and workplace conditions to identify areas that may need improvement.
- Examine salary hike and promotion patterns alongside performance ratings to understand how employee evaluations and rewards are related.
- Investigate differences across departments and job roles before making organization-wide decisions.
- Evaluate the model on additional employee data before using it in a real HR setting.
- Use model predictions as supporting information rather than the sole basis for decisions affecting employees.

## 6. Conclusion

This project demonstrated how data analysis and machine learning can be used to explore employee performance patterns and predict performance ratings.

The tuned Gradient Boosting model achieved 92.92% accuracy and a macro F1-score of 0.8866 on the test dataset.

However, the models performed less effectively when using only eight background and job-related features. Some of the most important predictors require information collected after an employee joins the company.

Therefore, the current model is more suitable for analysing existing employees than predicting the future performance of new candidates. Further validation would be needed before using it to support real HR decisions.