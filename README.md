# Clustering Transaction Narrations to Extract Types of Transactions
Sections:
1. Loading The Dataset
2. Cleaning
   1. Lower casing
   2. Eliminating non alphabetic characters
   3. Dropping duplicates
   4. Tokenization
   5. Lemmatization
   6. Removing stop words
   7. Removing some irrelevant words
   8. Removing Non English words
   9. Removing empty rows
3. Exploring After Cleaning
   1. Word Frequency Bar Plot
4. Embedding
   1. Pre-trained Word2Vec
   2. Trained Word2Vec
   3. Sentence Transformers
5. Clustering
   1. K means clustering
   2. DBSCAN
6. Testing

To run the notebook, the train and test dataset path need to be specified in the relevant portion of the notebook. Then the cells can be executed serially.


The DBSCAN algorithm may take a long time to run or sometimes may crash due to running out of memory.