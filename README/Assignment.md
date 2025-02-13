# Case

We are currently looking to hire a data scientist for our Data Science team here at LeoVegas. The ideal candidate should possess strong problem-solving skills, along with strong expertise in machine learning and coding skills. This test has been sent to you to assess your skills and knowledge in these areas.

Best of luck!

## Introduction

Customer churn is a significant concern for all customer-oriented businesses, including gambling operators. To improve retention, we need to identify players likely to churn so the CRM team can proactively engage them with relevant offers. This approach can significantly reduce churn rates and improve customer loyalty. It can also increase revenue and improve customer base stability.

Specifically, this model aims to provide the CRM team with a list of players likely to churn, enabling targeted campaigns (or other retention actions). This will help LeoVegas reduce churn rates and increase player lifetime value.

## Task

You have a dataset containing player and daily aggregations for approximately 12,500 customers who made their first deposit within a 2.5-year period. This dataset includes various features to help understand customer behavior and predict churn.

1. **Churn Implementation:** Implement churn as 30-days of inactivity, define what inactivity means, and justify your choice.

2. **Exploratory Data Analysis (EDA):** Perform a brief EDA focusing on key features relevant to churn prediction. Summarize your findings concisely using a few plots and summary statistics.  Address any significant data quality issues.

3. **Model Building:** Build a predictive model to classify players as likely to churn or not. Justify your choice of model and evaluation metrics.  A simple model (e.g., logistic regression, random forest) is sufficient. Focus on demonstrating your ability to build and evaluate a model effectively, not necessarily maximizing performance.

4. **Model Evaluation:**  Evaluate your model's performance using appropriate metrics. Briefly discuss any limitations of your model.

You are *not* required to build everything from scratch. Feel free to use any appropriate open-source packages.

## Data

The dataset contains 12,500 customers described by the following features:

* `player_key`: A unique customer identifier
* `birth_year`: The customer's birth year
* `date`: The date of the aggregated metrics below
* `turnover_sum`: The sum of bets placed
* `turnover_num`: The number of bets placed
* `NGR_sum`: The net gaming revenue (NGR) (i.e., how much the individual lost)
* `deposit_sum`: The sum of deposits from the customer's bank account to their gambling account
* `deposit_num`: The number of deposits from the customer's bank account to their gambling account
* `withdrawal_sum`: The sum of withdrawals from the customer's gambling account to their bank account
* `withdrawal_num`: The number of withdrawals from the customer's gambling account to their bank account
* `login_num`: The number of logins made

**Example Data Values**

| player_key            | birth_year | date       | turnover_sum | turnover_num | NGR        | deposit_sum  | deposit_num | withdrawal_sum | withdrawal_num | login_num |
| ----------------------| ---------- | ---------- | ------------ | ------------ | ---------- | ------------ | ----------- | -------------- | -------------- | --------- |
| -2930472881471393003  | 2000       | 2021-05-03 | 245.99       | 11           | 58.04      | 65.0         | 2           | 0.0            | 0              | 4         |
| 3019988275617763302   | 1990       | 2020-10-31 | 284.21       | 304          | 100.41     | 83.0         | 2           | 0.0            | 0              | 4         |
| 3626642277166270101   | 1959       | 2022-09-30 | 62.52        | 294          | -18.93     | 29.0         | 2           | 45.0           | 2              | 6         |

The data is derived from real customers but has been transformed to obfuscate the real customer data.

**Deliverables:**

A Jupyter Notebook or Python script(s) including your code, EDA results, model training and evaluation, and a concise discussion of your findings.

## Grading

This assignment assesses your understanding of key data science principles, not
model performance.  Submissions will be evaluated based on problem-solving,
machine learning principles, design decisions, and coding style.  Limit
additional work to written text.

Specifically, we will assess:

* **A correct implementation of your churn target** and its justification.
* **Exploratory Data Analysis (EDA):** Your exploration of data distributions,
  missing values, and relationships.
* **Code quality** will be assessed based on readability and efficiency.
* **Machine learning principles:** No fundamental rules of ML model building, such as avoiding target leakage, are violated.
* **An understanding of model limitations** and potential improvements in written text.

This assignment is designed to be completed within 2-4 hours.  We encourage you to prioritize functionality and demonstrate your problem-solving abilities.  Given the time constraint, a perfectly polished solution isn't necessary.  If you're unable to complete all aspects, please submit your work up to the time limit; we're mainly interested in seeing your progress and approach.

**Good Luck!**
