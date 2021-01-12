from sklearn import svm
from sklearn import datasets
from IrisService import IrisClassifier

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

iris_classifier_service = IrisClassifier()
iris_classifier_service.pack('model', clf)
saved_path = iris_classifier_service.save()
print(f'trained and saved to {saved_path}')
