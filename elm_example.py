from sklearn.datasets import load_digits, load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from extreme_learning_machines import ELMClassifier
from extreme_learning_machines import ELMRegressor

boston = load_boston()

X = StandardScaler().fit_transform(boston.data)[:200]
y = boston.target[:200]

elm = ELMRegressor(n_hidden=50, weight_scale=10)

elm.fit(X, y)
print "Regression score %f" % elm.score(X, y)


digits_dataset_binary = load_digits(n_class=2)

X = MinMaxScaler().fit_transform(digits_dataset_binary.data[:200])
y = digits_dataset_binary.target[:200]


elm = ELMClassifier(n_hidden=50, weight_scale=10)

elm.fit(X, y)
print "Classification score %f" % elm.score(X, y)