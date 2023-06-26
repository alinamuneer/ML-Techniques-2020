from libsvm.commonutil import svm_read_problem
from libsvm.svmutil import *
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV

labels, features = svm_read_problem('libsvm.data', return_scipy=True)
svc = svm.SVC()
param_grid = [
{'gamma': np.power(2., np.arange(3, -15, -2)),
'C': np.power(2., np.arange(-5, 15, 2)),
'kernel': ["poly"]}
]
clf = GridSearchCV(svc, param_grid)
clf.fit(features[:15], labels[:15])
print(clf.best_params_)

model = svm_train(labels[:15], features[:15], '-c 0.3125 -g 8.0 -t 1 -d 3')
p_label, p_acc, p_val = svm_predict(labels[15:], features[15:], model)

#p_labs, p_acc, p_vals = svm_predict(y[:59], x[:59], model [,'predicting_options'])

