!pip install wekapy
import pandas as pd
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.classifiers import Evaluation
from weka.core.classes import Random
# Load dataset
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("weather.nominal.arff")
data.class_is_last()
# Preprocess the data (not necessary for this dataset)
# Implement J48 algorithm
cls = Classifier(classname="weka.classifiers.trees.J48")
cls.build_classifier(data)
# Verify the result
evaluation = Evaluation(data)
evaluation.crossvalidate_model(cls, data, 10, Random(1))
# Print classification results
print("Classification Results:")
print(evaluation.summary())
# Print entropy values and Kappa statistic
print("Entropy:", evaluation.mean_entropy())
print("Kappa:", evaluation.kappa())
# Extract if-then rules
print("If-then rules:")
print(cls)