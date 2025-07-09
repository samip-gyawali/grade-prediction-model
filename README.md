# Grades Prediction Model

This project implements a simple **polynomial regression model** (degree 2) using the `scikit-learn` library to predict student final grades. The model is trained on the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance), which includes a variety of features.

We used the following features in our model:

- **Academic**: `G1` (1st period grade), `G2` (2nd period grade), `studytime`, `failures`, `absences`, `paid`
- **Demographic & lifestyle**: `age`, `sex`, `Pstatus`, `traveltime`, `freetime`, `Dalc` (weekday alcohol consumption)

All features were scaled using **z-score normalization**, and categorical variables were properly encoded. The model was trained with **polynomial feature expansion (degree=2)**.

The final model achieved a **mean prediction error (RMSE) of 1.31** on the test set. This indicates high accuracy, with predictions typically within ±1.3 grade points (on a 0–20 scale).