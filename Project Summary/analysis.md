# INX Future Inc. Employee Performance Analysis

## 1. Introduction

In this project, I analysed employee data from INX Future Inc. to understand the patterns associated with employee performance ratings.

I used Python for data processing and exploratory data analysis, then trained and compared different machine learning models to predict employee performance ratings.

The main goal was to understand the data, identify useful patterns, evaluate different models, and select a suitable model for making predictions.

## 2. Data Processing and Cleaning

I started by loading the original employee dataset and checking its structure, column data types, missing values, duplicate records, and statistical summaries.

The original dataset contained 1,200 records and 28 columns.

During data quality checks, I found that:
- There were no missing values.
- No duplicate rows were found.
- The numerical values were generally within reasonable ranges.
- Some experience-related columns contained potential outliers that needed further investigation.

I examined the potential outliers using the Interquartile Range (IQR) method. The investigation identified 51 potential outliers in TotalWorkExperienceInYears and 56 in ExperienceYearsAtThisCompany.

After checking the values, I decided to retain these records because they appeared reasonable for long-serving employees. I did not find a sufficient reason to remove them.

Since no data quality issues required correction, I retained the original records and saved the processed dataset for further analysis. The raw dataset remained unchanged.

## 3. Feature Engineering

I created a new feature called `ExperienceBeforeCurrentCompany`.

This feature represents the difference between an employee's total work experience and their experience at the current company.

It helps describe how much work experience an employee had before joining the current organization.

I checked the new feature for negative and missing values before using it in further analysis.

## 4. Exploratory Data Analysis

I used exploratory data analysis to understand employee performance ratings and examine patterns across different employee characteristics.

The analysis covered performance rating distribution, department, job role, employee satisfaction, salary hikes, training, overtime, business travel, work experience, and correlations among numerical features.

### 4.1 Performance Rating Distribution

The target variable was `PerformanceRating`, which contained three categories.

| Performance Rating | Number of Employees |
|---|---:|
| 2 | 194 |
| 3 | 874 |
| 4 | 132 |
| Total | 1,200 |

Rating 3 was the most common category, accounting for most employees in the dataset.

This imbalance was important when evaluating the classification models because accuracy alone might not fully represent their ability to predict all three categories.

### 4.2 Department-Wise Performance

The department analysis showed that Development had the highest average performance rating at 3.09, followed by Data Science at 3.05.

Finance had the lowest average rating at 2.78, followed by Sales at 2.86.

Although these differences provide useful information about the dataset, they do not prove that an employee's department directly determines their performance.

### 4.3 Employee Satisfaction and Work Factors

Employee environment satisfaction showed a positive correlation of approximately 0.396 with performance rating.

Employees with higher environment satisfaction generally had higher average performance ratings in the dataset.

Work-life balance showed a weaker positive correlation of 0.124. Job satisfaction, job involvement, and relationship satisfaction had correlations close to zero.

These results suggest that the relationship between performance ratings and workplace factors differs across features.

### 4.4 Salary Hikes and Training

Salary hike percentage showed a positive correlation of approximately 0.334 with performance rating.

Employees with a rating of 4 received an average salary hike of 20.70%, compared with 15.07% for rating 2 and 14.43% for rating 3.

Training frequency showed only small differences in average performance ratings across groups. The analysis did not reveal a consistent pattern in which more training sessions were associated with higher ratings.

Salary hikes may also reflect previous performance assessments, so this relationship should not be interpreted as proof that a higher salary hike causes better performance.

### 4.5 Overtime and Business Travel

Employees who worked overtime had an average performance rating of 2.99, compared with 2.93 for employees who did not.

Average ratings across business travel groups were also similar, ranging from 2.94 to 2.98.

These differences were relatively small and do not establish a causal relationship between overtime, business travel, and performance.

### 4.6 Work Experience and Job Roles

The analysis showed differences in average work experience across performance rating categories. Employees with a rating of 2 had higher average total work experience, company experience, and current-role experience than employees with ratings of 3 or 4.

Business Analysts had the highest average job-role performance rating at 3.19, followed by Developers at 3.11. Finance Managers had the lowest average at 2.78.

The number of employees varied across job roles, so the averages for smaller groups should be interpreted carefully.

### 4.7 Correlation Analysis

I used correlation analysis to examine linear relationships between numerical features and performance ratings.

The strongest relationships with performance rating were:

| Feature | Correlation with Performance Rating |
|---|---:|
| EmpEnvironmentSatisfaction | 0.396 |
| EmpLastSalaryHikePercent | 0.334 |
| YearsSinceLastPromotion | -0.168 |
| ExperienceYearsInCurrentRole | -0.148 |

The correlation heatmap also revealed relationships between experience-related features. For example, job level and total work experience had a correlation of approximately 0.78.

These relationships suggest that some features contain overlapping information.

However, correlation measures linear association rather than causation. Features with weak correlations may still help machine learning models identify more complex patterns.

## 5. Model Development

After completing data processing and exploratory data analysis, I prepared the dataset for machine learning.

I used `PerformanceRating` as the target variable.

I excluded `EmpNumber` because it is an employee identifier. I also excluded `Attrition` from the prediction features because it indicates whether an employee has left the company.

I tested two feature sets:

1. Eight background and job-related features.
2. All available predictor features, including employee satisfaction, salary hike percentage, promotion history, and work experience.

The data was divided into training and testing sets using an 80:20 split, with `random_state=42` and stratification.

I used StandardScaler for numerical features and OneHotEncoder for categorical features within the model preprocessing pipelines.

I trained four classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## 6. Model Evaluation

I compared the models using accuracy, macro precision, macro recall, and macro F1-score.

The first experiment used eight features. The results showed that the models struggled to identify the minority performance rating categories reliably.

I then trained the models using all available predictor features.

The results were:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.08% | 0.7643 | 0.6761 | 0.7121 |
| Decision Tree | 89.17% | 0.8451 | 0.8361 | 0.8405 |
| Random Forest | 90.00% | 0.9304 | 0.7763 | 0.8360 |
| Gradient Boosting | 92.92% | 0.9134 | 0.8623 | 0.8856 |

Gradient Boosting achieved the highest accuracy and macro F1-score among these four models.

The results also showed that using more employee information improved the model performance in this experiment. However, some of those features are only available after an employee joins the company.

## 7. Hyperparameter Tuning

I used GridSearchCV to tune the Gradient Boosting model and identify a suitable combination of hyperparameters.

After tuning, the model achieved:

- Accuracy: 92.92%
- Macro precision: 0.9252
- Macro recall: 0.8556
- Macro F1-score: 0.8866

The accuracy remained unchanged, while the macro F1-score improved slightly from 0.8856 to 0.8866.

This indicates that hyperparameter tuning provided only a small improvement over the original Gradient Boosting model.

## 8. Feature Importance

I examined feature importance using the tuned Gradient Boosting model to understand which features it relied on most when making predictions.

The three most important features were:

1. Employee environment satisfaction
2. Employee last salary hike percentage
3. Years since the last promotion

Together, these features accounted for approximately 69% of the model's impurity-based feature importance.

The analysis also showed contributions from department, current-role experience, and work-life balance.

These results describe the model's learned patterns. They do not prove that the features directly cause changes in employee performance.

## 9. Key Findings

The main findings from the project are:

- Performance rating 3 was the most common category in the dataset.
- Employee environment satisfaction and salary hike percentage had the strongest positive correlations with performance rating among the numerical features examined.
- Years since the last promotion had a negative correlation with performance rating.
- The models performed better when trained using all available predictor features than when using only eight background and job-related features.
- Gradient Boosting achieved the best overall results among the four models tested using all predictor features.
- Hyperparameter tuning produced a small improvement in macro F1-score.
- Feature importance highlighted environment satisfaction, salary hike percentage, and promotion history as the features the final model relied on most.

## 10. Limitations

The model was evaluated using a held-out test set from the same dataset used for training. Its performance may differ when applied to other organizations or future employee data.

The dataset also contains more employees with a performance rating of 3 than ratings of 2 or 4. Although the model achieved good overall results, its ability to identify each category should be considered when interpreting the results.

Some important features, such as environment satisfaction, salary hike percentage, and promotion history, may not be available when hiring a new candidate.

The experiment using eight background and job-related features performed less effectively. Therefore, the current model is better suited to analysing existing employees than predicting the performance of new candidates.

Finally, feature importance and correlation do not establish cause and effect. Model predictions should support HR analysis rather than serve as the sole basis for decisions affecting employees.