
# Project 2: Supervised Learning
### Building a Student Intervention System

## 1. Classification vs Regression

Your goal is to identify students who might need early intervention - which type of supervised machine learning problem is this, classification or regression? Why?

> This is a classifcation problem, where the goal is to predict student graduation outcomes with binary output -- whether the student passed ("yes"), or didn't pass ("no"). Classification is for solving problems with discrete value outputs.

## 2. Exploring the Data



- Total number of students = **395**
- Number of students who passed = **265**
- Number of students who failed = **130**
- Graduation rate of the class (%) = **67.09**
- Number of features = **31**


## 3. Preparing the Data

- Identify feature and target columns
- Preprocess feature columns
- Split data into training and test sets

## 4. Training and Evaluating Models
Choose 3 supervised learning models that are available in scikit-learn, and appropriate for this problem.

Produce a table showing training time, prediction time, F<sub>1</sub> score on training set and F<sub>1</sub> score on test set, for each training set size.


### Model #1: [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

- What are the general applications of this model? What are its strengths and weaknesses?
> Classification via optimization.

    > Underfitting.

- Given what you know about the data so far, why did you choose this model to apply?
> Many features, limited data.


| Logistic regression       | Training set size 100|Training set size 200|Training set size 300|
| --------------------------|:-----:|:-----:|:-----:|
| Training time (secs)      |0.002  |0.002  |0.005  |
| Prediction time (secs)    |0.000  |0.000  |0.000  |
| F1 score for training set |0.8571 |0.8380 |0.8381 |
| F1 score for test set     |0.7612 |0.7794 |0.7910 |

---

### Model #2: [SVM (Support Vector Machine)](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

- What are the general applications of this model? What are its strengths and weaknesses?
> Classification via margin maximization.

- Given what you know about the data so far, why did you choose this model to apply?
> Many features, limited data.


| SVM                       | Training set size 100|Training set size 200|Training set size 300|
| --------------------------|:-----:|:-----:|:-----:|
| Training time (secs)      |0.004  |0.005  |0.011  |
| Prediction time (secs)    |0.002  |0.003  |0.006  |
| F1 score for training set |0.8591 |0.8693 |0.8692 |
| F1 score for test set     |0.7838 |0.7755 |0.7586 |

---

### Model #3: [Nearest Neighbor](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

- What are the general applications of this model? What are its strengths and weaknesses?
> Classification via lazy learner.

    > All features considered equally.

- Given what you know about the data so far, why did you choose this model to apply?
> Many features, preserves data from training examples.



| KNN                       | Training set size 100|Training set size 200|Training set size 300|
| --------------------------|:-----:|:-----:|:-----:|
| Training time (secs)      |0.001  |0.001  |0.001  |
| Prediction time (secs)    |0.023  |0.003  |0.008  |
| F1 score for training set |0.7972 |0.8571 |0.8722 |
| F1 score for test set     |0.7068 |0.7121 |0.7482 |

## 5. Choosing the Best Model

- Based on the experiments you performed earlier, in 1-2 paragraphs explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?
> Logistic Regression.

    > Simple, works with few data, explainable.

- In 1-2 paragraphs explain to the board of supervisors in layman's terms how the final model chosen is supposed to work.
> Calculates probability using coefficients for features.

- Fine-tune the model. Use Gridsearch with at least one important parameter tuned and with at least 3 settings. Use the entire training set for this.
> Used Gridsearch to tune the parameters C, class_weight, and max_iter.

- What is the model's final F<sub>1</sub> score?
> Final F1 score is 0.8.

---
