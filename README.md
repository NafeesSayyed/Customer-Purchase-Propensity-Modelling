# Customer Purchase Propensity Modelling

Predicting whether a customer will convert after seeing a marketing campaign — built end-to-end on a synthetic e-commerce dataset designed to behave like the messy, imperfect data you'd actually find in the real world.

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Imbalanced-learn · XGBoost · Jupyter Notebook

## Why this project

Every business with a marketing budget runs into the same problem: blasting offers to your entire customer base is expensive, and most of it is wasted on people who were never going to buy anyway (or were going to buy regardless of the campaign). The interesting question isn't "did they purchase?" — it's "who is actually worth targeting, and why?"

I built this project to answer that question properly, not just fit a model and report an accuracy score. That meant spending real time on:

- Which customers are most likely to purchase?
- What behaviors actually precede a purchase, versus what's just noise?
- Do some campaign channels or timings clearly outperform others?
- Can a model meaningfully improve targeting over random selection?

To do this properly, I needed a dataset that behaved like real data — with the quirks, inconsistencies, and imperfections you'd actually encounter — rather than a clean textbook CSV. So I generated one synthetically, with intentional data quality issues baked in, and treated cleaning it as part of the actual work rather than something to skip past.

---

## The Dataset

More than 50,000 rows, 39 raw features across five categories, and one binary target: `purchased`.

| Attribute | Value |
|-----------|------:|
| Rows | 50,000 |
| Features | 39 raw + engineered |
| Target | `purchased` |
| Problem Type | Binary Classification |

Feature groups:
- Customer demographics
- Browsing behavior
- Transaction history
- Campaign information
- Customer engagement

---

## How the project came together

```
Data Collection
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Class Imbalance Handling
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Business Insights
```

---

## Exploratory Data Analysis

Before touching a model, I spent time actually understanding the data — where it was messy, what was missing, and what patterns were hiding in customer behavior. This covered:

- Dataset overview and structure
- Missing value analysis
- Duplicate detection
- Outlier detection (IQR method)
- Class distribution
- Campaign channel performance
- Customer segment behavior
- Purchase rate patterns
- Feature correlation

This step ended up shaping a lot of the feature engineering decisions later — several "obvious" features turned out to be much weaker predictors than expected, and some non-obvious ones (like recency) turned out to matter a lot.

---

## Data Preprocessing

Cleaning a dataset with intentional quality issues meant handling:

- **Missing values** — imputed using mean, median, or constant values depending on the feature
- **Duplicates** — identified and removed
- **Inconsistent categories** — fixed mismatched labels, trailing whitespace, and inconsistent naming across categorical fields
- **Outliers** — treated using the IQR method
- **Encoding** — one-hot encoding for nominal features, label encoding where order mattered
- **Scaling** — StandardScaler for numeric features
- **Train-test split** — to keep evaluation honest

---

## Feature Engineering

Rather than throwing raw columns at a model, I built features grounded in actual customer behavior and marketing logic:

- Purchase frequency
- Average monthly spend
- Cart abandonment rate
- Purchase conversion rate
- Browsing-to-purchase ratio
- Campaign time bucket
- RFM score
- Customer segment
- Age group

This was iterative — I'd add a feature, check its impact on model performance and importance rankings, and either keep it, refine it, or drop it.

---

## Feature Selection

Not everything survived. After checking importance scores, correlation, and validation performance, I dropped several features that added noise more than signal:

- Family size
- Pages viewed
- Cart additions
- Refund requests
- Age (superseded by age group)
- Individual RFM component scores (once the combined RFM score was created)

---

## Handling Class Imbalance

Purchases are the minority class here, so I tested a few different strategies rather than assuming one would work best:

- Baseline (no resampling)
- Random oversampling
- Random undersampling
- SMOTE
- SMOTEENN

Each was compared on precision, recall, F1-score, and confusion matrices — accuracy alone would've been misleading given the imbalance.

---

## Models Evaluated

- Logistic Regression
- Ridge Classifier
- Support Vector Machine (RBF kernel)
- K-Nearest Neighbors
- Naive Bayes
- Random Forest
- XGBoost

Hyperparameters were tuned manually rather than left at defaults.

---

## Results

**Best performer: Ridge Classifier + Random Oversampling**

| Metric | Score |
|---------|-------|
| Accuracy | **84.36%** |
| Precision | **0.67** |
| Recall | **0.85** |
| F1 Score | **0.75** |

The Ridge Classifier gave the best balance between catching actual buyers (recall) and not over-flagging non-buyers (precision), while also holding the highest accuracy of everything tested.

XGBoost, SVM (RBF), and Random Forest all performed respectably close behind — worth keeping in mind if the business priorities shift (e.g., if precision matters more than recall for a given campaign).

---

## What actually mattered — Feature Importance

Based on Random Forest's feature importance rankings, in order:

1. Days since last purchase
2. Total orders
3. RFM score
4. Purchase frequency
5. Product views
6. Purchase conversion rate
7. Total amount spent
8. Website visits
9. Previous campaign response rate
10. Annual income

The takeaway that stood out most: **recency and order history dominate.** Demographics like age and income mattered far less than how recently and how often someone had already engaged.

---

## Business Insights

Pulling it together, a few findings stood out enough to be actionable:

- **Recency and order history** are the strongest purchase signals by a clear margin.
- Higher **RFM scores** correspond to meaningfully higher purchase rates — this alone could justify a simpler rule-based targeting strategy for teams not ready to deploy ML.
- **Campaign timing** has a measurable effect on conversion, not just channel choice.
- **Customer segmentation** noticeably improves how well campaigns target the right people.
- **Past campaign response** is one of the strongest predictors of future response — loyalty (or fatigue) compounds.
- Across the board, **behavioral features beat demographic ones** for prediction — who someone *is* matters less than what they've actually *done*.

---
## What's Next

A few directions I want to take this further:

- SHAP-based explainability, to move beyond feature importance and explain individual predictions
- A Streamlit dashboard so the model is usable by non-technical stakeholders
- Docker packaging for reproducibility
- Hyperparameter optimization with Optuna instead of manual tuning
- A FastAPI endpoint for real model deployment
- Eventually, testing this pipeline against a real-world customer dataset instead of synthetic data

---
## Author

**Nafees Sayyed**
Bachelor of Engineering — Computer Science (Data Science)
GitHub: [github.com/Nafees2006](https://github.com/Nafees2006)

---

## License

MIT License.
