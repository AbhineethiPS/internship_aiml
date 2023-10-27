# internship_aiml
## diabetics analysis
##### ABSTRACT 
Diabetes is a prevalent chronic health condition that affects millions of people worldwide. Early 
diagnosis and effective management are crucial to mitigate its adverse effects on individuals' health. 
Machine Learning (ML) techniques have shown promise in predicting the risk of diabetes based on 
various patient attributes. This project presents a user-friendly web application built using Streamlit 
to provide a practical and accessible solution for diabetes prediction.
The proposed system leverages a dataset containing patient information such as age, gender, BMI, 
family history, and other relevant features. A machine learning model is trained using this dataset 
to predict the likelihood of an individual developing diabetes. Various ML algorithms, such as 
logistic regression, random forests, and support vector machines, are explored to identify the most 
accurate predictive model.
The Streamlit framework is used to create an interactive and user-friendly web application that 
allows users to input their health data and receive an instant prediction of their diabetes risk. The 
application provides not only a binary prediction (diabetic or non-diabetic) but also a probability 
score, which helps users understand the level of risk associated with their health status.
This project aims to empower individuals with a convenient tool to assess their diabetes risk and 
make informed decisions about their health. By combining the power of machine learning and the 
accessibility of Streamlit, this application offers a practical and efficient means of early diabetes 
detection, ultimately contributing to improved healthcare and the well-being of individuals.
##### INTRODUCTION 
Diabetes is a chronic medical condition of growing concern worldwide, characterized by elevated 
blood sugar levels that can have severe health implications. Early detection and management of 
diabetes are essential to mitigate its complications and improve the quality of life for those 
affected. Machine Learning (ML) techniques have demonstrated their potential in predicting 
diabetes risk, offering a powerful tool for early diagnosis and prevention. To make this predictive 
capability accessible to a broader audience, we propose the development of a user-friendly web 
application utilizing Streamlit, a modern web application framework for Python.
This project seeks to bridge the gap between the sophisticated world of machine learning and 
the practical needs of individuals concerned about their health. By combining ML with the 
simplicity and interactivity of Streamlit, our application aims to provide an intuitive and 
efficient means for users to assess their risk of developing diabetes. The system leverages a 
dataset containing various health-related attributes, including age, gender, body mass index 
(BMI), family history, and other factors, to train a predictive model. Multiple ML algorithms 
are explored to identify the most accurate model for diabetes risk prediction.
Through this web application, users can input their personal health data, and within moments, 
receive a prediction of their likelihood of developing diabetes. The application not only offers a 
binary classification (diabetic or non-diabetic) but also provides a probability score, enabling 
users to gauge the level of risk associated with their health status.
This project carries the potential to empower individuals with a valuable tool for informed 
decision-making regarding their health. By democratizing the use of machine learning for 
diabetes prediction, we aim to contribute to the early detection of this prevalent condition, 
ultimately fostering better healthcare outcomes and enhancing the well-being of individuals. In 
the following sections, we will delve into the technical aspects of the project, including data 
collection and preprocessing, machine learning model development, and the user interface 
design using Streamlit.

##### PROBLEM STATEMENT
Developing a diabetes prediction model using machine learning poses a set of challenges. 
Firstly, obtaining a diverse and high-quality dataset for model training is a critical concern, as 
data quality significantly impacts prediction accuracy. Second, the selection and evaluation 
of the most suitable machine learning algorithm for diabetes prediction require careful 
consideration, as various algorithms offer different trade-offs between accuracy and 
interpretability. Additionally, handling imbalanced data, a common issue in medical datasets, 
is essential to prevent bias in the predictive model. Ensuring user privacy and data security, 
especially in a healthcare context, demands robust measures to comply with regulatory 
standards such as HIPAA. Designing an intuitive, user-friendly interface for the application 
to cater to users with varying levels of technical expertise is another crucial challenge. Finally, 
promoting ethical use of the application by providing responsible and empowering 
information to users while avoiding unnecessary anxiety adds another layer of complexity to 
the problem. Addressing these challenges is essential to create a reliable and user-centric 
diabetes prediction system.

##### METHODOLOGIES 
The methodology for developing a diabetes prediction application using machine learning and 
Streamlit involves a series of steps, from data collection to application deployment. Here is a 
concise outline of the methodology:
1. Data Collection and Preprocessing:
 - Collect a diverse and reliable dataset containing health-related features, such as age, 
gender, BMI, family history, and glucose levels.
 - Clean the data by handling missing values, outliers, and inconsistencies.
 - Normalize or standardize the data to ensure consistency in feature scales.
2. Exploratory Data Analysis (EDA):
 - Perform EDA to gain insights into the dataset, identify correlations, and understand the 
distribution of features.
 - Visualize key statistics and relationships between variables using plots and charts.
3. Feature Engineering:
 - Select the most relevant features for diabetes prediction through feature selection 
techniques.
 - Create new features or transformations if necessary to improve model performance.
4. Machine Learning Model Development:
 - Split the dataset into training and testing sets for model development and evaluation.
 - Train multiple machine learning algorithms (e.g., logistic regression, random forests, 
support vector machines) on the training data.
 - Evaluate model performance using appropriate metrics, such as accuracy, precision, recall, 
and F1 score.
5. Imbalanced Data Handling:
 - Address class imbalance by implementing techniques like oversampling (SMOTE), 
undersampling, or utilizing specialized algorithms designed for imbalanced datasets.
6. Model Explainability:
 - Enhance model transparency and user trust by implementing explainability techniques 
such as SHAP values, feature importance scores, or LIME (Local Interpretable Model-
Agnostic Explanations
##### IMPLEMENTATION
1. Data Collection and Preprocessing:
 - Collect a dataset containing health-related attributes, ensuring it includes a target variable 
indicating diabetes status.
 - Preprocess the data by addressing missing values, outliers, and normalizing feature scales, taking 
care not to leak any information from the test set into the training set.
2. Exploratory Data Analysis (EDA):
 - Explore the dataset to understand its characteristics, including feature distributions and potential 
correlations between attributes.
3. Feature Engineering:
 - Select relevant features for diabetes prediction based on your findings from EDA.
 - Engineer new features, if necessary, to capture meaningful patterns in the data.
4. Data Splitting:
 - Split the dataset into a training set and a separate testing set to evaluate the model's performance 
accurately.
5. SVM Model Development:
 - Train an SVM classifier using the training data with an appropriate kernel (e.g., linear, 
polynomial, or radial basis function).
 - Fine-tune hyperparameters like the regularization parameter (C) and kernel parameters for 
optimal model performance.
6. Model Evaluation:
 - Assess the SVM model's performance using various evaluation metrics, including accuracy, 
precision, recall, F1 score, and the receiver operating characteristic (ROC) curve.
7. Imbalanced Data Handling:
 - Implement strategies to address class imbalance, such as adjusting class weights or oversampling 
the minority class to ensure a balanced prediction
##### RESULT
The expected results and outcomes of the "DiabetesPredict Pro" application are as follows:
1. Accurate Diabetes Risk Prediction:
 - The primary outcome is to provide users with accurate predictions of their risk of developing 
diabetes based on their health data. This will help individuals gain insights into their health 
status.
2. User Empowerment:
 - By offering clear explanations and educational content, the application aims to empower 
users with knowledge about diabetes risk factors and prevention strategies. The outcome is 
informed decision-making for a healthier lifestyle.
3. User-Friendly Experience:
 - The application's user-friendly interface ensures that users, regardless of their technical 
background, can easily input their data and receive predictions. The outcome is a seamless and 
intuitive user experience.
4. Privacy and Security:
 - Implementing robust data privacy and security measures ensures that user health information 
is protected. The outcome is user confidence in the application's privacy standards.
5. Ethical Use of Healthcare Information:
 - The application's adherence to ethical guidelines ensures that it promotes responsible and 
informed healthcare decisions. The outcome is a positive impact on users' well-being without 
causing undue anxiety.
6. Scalability and Performance:
 - By deploying the application on a scalable platform and optimizing its performance, the 
outcome is the ability to accommodate a growing user base and provide a responsive user 
experience.
##### CODE 
### import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and Columns in this dataset
diabetes_dataset.shape

# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


print('Accuracy score of the test data : ', test_data_accuracy)

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

import pickle

filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')*
