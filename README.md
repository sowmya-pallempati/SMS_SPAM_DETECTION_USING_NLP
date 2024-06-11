# SMS Spam Detection Using Natural Language Processing(NLP)

## Overview
This project focuses on building a machine-learning model to classify SMS messages as spam or ham (non-spam). 
Using Natural Language Processing (NLP) techniques, we preprocess the text data, convert it into numerical vectors, and train a Naive Bayes classifier to detect spam messages.

## Dataset
The dataset consists of SMS messages labeled as either spam (1) or ham (0). The data is read from a text file and processed into a format suitable for machine learning tasks.

## Steps Involved
**1. Data Loading:**
We load the data using read_csv with a specified separator and column names.  
**2. Exploratory Data Analysis (EDA)**
We perform EDA to understand the dataset, including:
* Length of messages
* Unique message counts
* Top messages
  
**3. Feature Engineering**  
  
**a) Text Preprocessing**  
We preprocess the text data to convert it into a numerical format suitable for machine learning:

* Removing Punctuation: Using Python's built-in string library.
* Removing Stopwords: Utilizing NLTK's list of English stopwords.
* Tokenization: Splitting the text into individual tokens (words).

**b) Vectorization**  
We convert the preprocessed text into numerical vectors using the Bag-of-Words model and TF-IDF transformation:

* Term Frequency: Counting word occurrences.
* Inverse Document Frequency: Weighing counts to reduce the impact of frequent words.
* Normalization: Normalizing vectors to unit length

**4. Model Training and Evaluation**  
  
We train a Naive Bayes classifier on the vectorized data:

**Initial Model Evaluation:** Evaluating the model's performance on the training set to determine its accuracy.

**5. Pipeline Implementation and Model Evaluation**  

To streamline and automate the entire workflow, we use Scikit-Learn's pipeline capabilities:

* **Train-Test Split:** Splitting the data into 80% training and 20% testing sets before any preprocessing
* **Data Pipeline:** Creating a pipeline that incorporates all preprocessing steps and model training.
* **Pipeline Model Training:** Retraining the model using the pipeline on the training data.
* **Pipeline Model Evaluation:** Evaluating the pipeline model's performance on the test set to ensure true predictive performance.
For spam detection, the cost of misclassifying "ham" as "spam" is generally higher than misclassifying "spam" as "ham", so metrics like **precision** and **recall** are crucial.

![image](https://github.com/sowmya-pallempati/SMS_SPAM_DETECTION_USING_NLP/assets/112984551/5b40ef6e-de85-4c33-b673-ed90339d7a98)

## Conclusion
This project demonstrates the use of NLP and machine learning techniques to detect spam messages in SMS data. 
The implementation of a data pipeline ensures a streamlined and automated workflow for future use and model retraining. 
Proper model evaluation using a train-test split provides a realistic measure of the model's predictive power.



  

