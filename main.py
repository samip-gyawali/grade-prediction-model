from ucimlrepo import fetch_ucirepo 
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np
from matplotlib import pyplot as plt


model = linear_model.Lasso(alpha=0.08)

poly = PolynomialFeatures(degree=2, include_bias=False)

# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  

# preprocess
features = ['age', 'traveltime', 'studytime', 'failures', 'freetime', 'Dalc', 'absences', 'sex', 'Pstatus', 'paid']

X = student_performance.data.features[features]

sex_process = (X['sex'].values == 'F').astype(int)  # 0 as Male, 1 as Female
Pstatus_process = (X['Pstatus'].values == 'T').astype(int)  # 0 is Apart, 1 is Together
extraclass_process = (X['paid'].values == 'yes').astype(int)  # 0 is no, 1 is yes

X['sex'], X['Pstatus'], X['paid'] = sex_process, Pstatus_process, extraclass_process

X['G1'] = student_performance.data.targets['G1']
X['G2'] = student_performance.data.targets['G2']

y = np.array(student_performance.data.targets ['G3'])

scaler = StandardScaler()

X_poly = scaler.fit_transform(poly.fit_transform(X))

scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores
print(np.sqrt(mse_scores.mean()))