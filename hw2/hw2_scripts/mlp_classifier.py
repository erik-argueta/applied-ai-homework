# =======================
# 1. Load "iris flower" 
# =======================
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
iris = datasets.load_iris()


# ===============================================================
# 2. Split 80% of original into training; 20% into test dataset
# ===============================================================
X, y = iris['data'], iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================================================================================================
# 3. Train the decision tree algo to classify the dataset using a simple MLPClassifier in scikit-learn
#       with only one hidden layer with 3 neurons, but any other hyperparameters (e.g. activation functions,
#       learning_rate, batch_size, momentum, etc.) Compute the % accuracy for both training and testing error.
# ==============================================================================================================
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

subject_model = MLPClassifier(
    hidden_layer_sizes=(3,),        # ...sizes=(3,) -> one layer with three neurons;  sizes=(5, 2) -> two hidden layers with five and two neurons
    random_state=42,
)

subject_model.fit(X_train, y_train)
print("\n3. Initial Hyperparameters:")
print(f"Training Accuracy: {subject_model.score(X_train, y_train):.2%}")
print(f"Test Accuracy: {subject_model.score(X_test, y_test):.2%}")
# Training Accuracy: 94.17%
# Test Accuracy : 96.67%



# =====================================================================================================
# 4. Adjust or change any MLP hyperparameters (except for adding more layers or neurons) to enhance 
#       classification performance.  
# =====================================================================================================

subject_model = MLPClassifier(
    hidden_layer_sizes=(3,),
    random_state=42,
    max_iter=700,      # default is 200
)

subject_model.fit(X_train, y_train)
print("\n4. New Hyperparameters:")
print(f"Training Accuracy: {subject_model.score(X_train, y_train):.2%}")
print(f"Test Accuracy: {subject_model.score(X_test, y_test):.2%}")
# Training Accuracy: 98.33%
# Test Accuracy: 100.00%


# ===================================================================================================================================
# 5. Increase the network complexity by adding more layers and/or neurons until the new architecture shows the improved accuracy. 
#       Briefly explain if you were able to improve the classification performance.
# ===================================================================================================================================
subject_model = MLPClassifier(
    hidden_layer_sizes=(5,2),
    random_state=42,
    max_iter=1500,
    solver='lbfgs'      # The default is 'adam'; is an optimizer in the family of quasi-Newton methods.
)

subject_model.fit(X_train, y_train)
print("\n5. New Hyperparameters:")
print(f"Training Accuracy: {subject_model.score(X_train, y_train):.2%}")
print(f"Test Accuracy: {subject_model.score(X_test, y_test):.2%}")
# Training Accuracy: 
# Test Accuracy: 100.00%