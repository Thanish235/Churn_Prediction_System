# Customer Segmentation and Churn Prediction System

An end-to-end Machine Learning and Business Intelligence pipeline built using the IBM Telecom Dataset. This project implements a robust predictive system to forecast customer churn alongside an unsupervised clustering workflow to identify key customer segments for targeted retention strategies.

---

## Project Workflow & Methodology

The project was executed through a structured data science pipeline:
### 1. Data Cleaning
* **Handling Missing Values:** Handled blank string structures (`" "`) found within the `Total Charges` column by safely casting them into numerical formats (`pd.to_numeric`) and filling missing records with `0` (representing brand new customers with 0 tenure months).
* **Dimensionality Reduction:** Dropped high-cardinality geographic, operational, and redundant metadata labels (`CustomerID`, `Count`, `Country`, `State`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `City`, `Churn Score`, `CLTV`, `Churn Reason`) to prepare a clean matrix.

### 2. Encoding
* **Categorical Transformation:** Converted multi-class nominal variables and binary categorical attributes into clean numerical configurations using **One-Hot Encoding** (`pd.get_dummies(drop_first=True)`) to eliminate mathematical bias in distance calculations.

### 3. Exploratory Data Analysis (EDA)
* **Statistical Visualizations:** Plotted data distributions using Seaborn and Matplotlib (`histplot`, `boxplot`, `countplot`) to observe variances in billing traits (`Monthly Charges`), loyalty behaviors (`Tenure Months`), and contract operational frameworks against customer churn status.

### 4. Feature Selection & Target Extraction
* **Feature Matrices:** Separated the historical features matrix ($X$) from the downstream target variable ($Y = \text{`Churn Value`}$). 
* Re-filtered final analytical subsets to prevent data leakage during subsequent cluster assignments.

### 5. Train-Test Split
* **Data Slicing:** Partitioned the feature space into independent subsets using an 80/20 stratification ratio via `train_test_split` with a fixed `random_state=42` to guarantee experiment reproducibility.

### 6. Random Forest (Hyperparameter Tuning & Feature Importance)
* **Class Imbalance Mitigation:** Configured balanced class weights (`class_weight='balanced'`) inside an ensemble architecture to combat sample skewness (majority active users vs. minority churn cases).
* **Hyperparameter Optimization:** Iterated over parameters (`n_estimators=320`, `max_depth=8`) to prevent overfitting while maximizing sensitivity metrics.
* **Feature Importance Evaluation:** Calculated tree node impurity decreases to assess which billing indicators and service setups dominantly influence a customer's decision to drop services.

### 7. Cross-Validation
* Built internal validation iterations to confirm model stability across alternate dataset slices, avoiding optimistic bias on the fixed test partition.

### 8. ROC-AUC Score Evaluation
* Generated predictive class probabilities (`predict_proba`) and mapped the False Positive Rate (FPR) vs. True Positive Rate (TPR).
* Computed the **ROC-AUC Score** to validate the ensemble classifier's distinct capability to separate churning profiles from loyal profiles.

### 9. Customer Segmentation Using K-Means
* Combined basic user attributes with the engineered classifier output (`Churn Probability`) to drive a multi-dimensional customer behavioral segmentation.
* Scaled all metrics using `StandardScaler` to uniform $z$-scores before running spatial computations.

### 10. K-Optimum Value (Elbow Curve)
* Iterated through a range of clusters ($k \in [1, 15]$) tracking Within-Cluster Sum of Squares (**WCSS**).
* Utilized the **Elbow Curve Method** to systematically identify the optimal curvature point, confirming $K=3$ as the ideal segment count.

### 11. Multi-Dimensional Visualization
* Generated comprehensive Seaborn scatter plots visualizing intersections between billing attributes, loyalty durations, and churn probability layers mapped explicitly across the target segments.

### 12. Business Insights & Segments Generated
The system clustered the telecom consumer base into 3 high-impact operational archetypes:
1. **Budget Loyal Customers:** Characterized by low monthly charges and high tenure. These are stable users requiring minimal marketing overhead.
2. **High Risk New Customers:** Display low overall tenure matched with steep monthly charges. This group exhibits a volatile churn probability and requires immediate customer success interventions and onboarding incentives.
3. **Premium Loyal Customers:** Maintain high monthly commitments paired with deep loyalty durations. They present prime candidates for expansion, cross-selling, and loyalty rewards programs.

---

## Tech Stack & Libraries Used
* **Languages:** Python (Google Colab Environment)
* **Core Frameworks:** Pandas, NumPy
* **Machine Learning & Stats:** Scikit-Learn (RandomForestClassifier, KMeans, StandardScaler)
* **Data Visualization:** Seaborn, Matplotlib
