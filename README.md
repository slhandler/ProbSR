This project focuses on leveraging machine learning models to predict the probability that road temperatures are subfreezing. Three different algorithms are examined: Logistic regression with elastic net penalty (LR), random forest (RF), and gradient boosted trees (GBT). XGBoost could be used to supplement GBT, but I wanted to be able to extract the individual trees for implementation using C++. 

The predictors are various model fields from the High-Resolution Rapid Refresh (HRRR) model downloaded from the University of Utah HRRR archive. These predictors include surface based fields such as temperature, relative humidity, and radiation products. Verification (targets) are provided by Road Weather Information System (RWIS) observations.

The directory structure for this project is as such:
 - notebooks: Jupyter notebooks consisting of some sample exploratory data analysis from the dataset along with model verification using standard evaluation metrics.
 - scripts: Python scripts that run through all amjor steps (i.e.,hyperparameter tuning, training, final evaluation) for each of the three models.
 - data: Directory where data is stored/written to.
 - models: The final trained models