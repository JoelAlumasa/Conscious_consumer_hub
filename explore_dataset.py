#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

# Define the paths to the dataset files
base_dataset_path = '/home/joel/Downloads/archive (2)'
train_dataset_filename = 'train.ft.txt'
test_dataset_filename = 'test.ft.txt'


# In[3]:


train_dataset_path = os.path.join(base_dataset_path, train_dataset_filename)
test_dataset_path = os.path.join(base_dataset_path, test_dataset_filename)


# In[4]:


# Load a subset of the dataset into a Pandas DataFrame for initial exploration
def load_dataset_subset(file_path, num_lines=10000):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [next(file) for _ in range(num_lines)]
    return pd.DataFrame(lines, columns=['review'])

print("Loading a subset of the training dataset...")
train_df = load_dataset_subset(train_dataset_path)

print("Loading a subset of the test dataset...")
test_df = load_dataset_subset(test_dataset_path)


# In[5]:


# Display the first few entries of the train dataset to understand its structure
train_df.head()


# In[11]:


# Define a function to extract the label and text from a review line
def preprocess_review(line):
    # Extract the label
    label = line.split(' ')[0]
    # Map '__label__2' to 1 (positive) and '__label__1' to 0 (negative)
    label = 1 if label == '__label__2' else 0
    # Extract the text
    text = ' '.join(line.split(' ')[1:])
    # Clean the text
    text = clean_text(text)
    return label, text

# Apply the preprocessing function to each review
train_df['label'], train_df['cleaned_text'] = zip(*train_df['review'].apply(preprocess_review))

test_df['label'], test_df['cleaned_text'] = zip(*test_df['review'].apply(preprocess_review))


# In[12]:


import re

# Define a function for cleaning and normalizing text
def clean_text(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # replace multiple whitespaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

# Apply the cleaning function to the text
train_df['cleaned_text'] = train_df['text'].apply(clean_text)
test_df['cleaned_text'] = test_df['text'].apply(clean_text)


# In[14]:


train_df.head()


# In[9]:


# Save the subset for easier access in the future (optional)
train_df.to_csv('train_subset.csv', index=False)
test_df.to_csv('test_subset.csv', index=False)


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limit the number of features to the top 5000

# Fit and transform the cleaned text
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
y_train = train_df['label']

X_test = vectorizer.transform(test_df['cleaned_text'])
y_test = test_df['label']


# In[18]:


from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)


# In[19]:


from sklearn.metrics import classification_report, accuracy_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print out the classification report and accuracy
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[20]:


from joblib import dump

# Serialize and save the trained model
dump(model, 'sentiment_model.joblib')

# Serialize and save the vectorizer
dump(vectorizer, 'tfidf_vectorizer.joblib')

