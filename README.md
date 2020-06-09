# question-answering-system
Repo for simple Question Answering (QA) system inspired by Project 6 of "CS50's Introduction to AI with Python" (https://cs50.harvard.edu/ai/2020/projects/6/questions/)

## Usage

```
$ python questions.py corpus
Query: What are the types of supervised learning?
Types of supervised learning algorithms include Active learning , classification and regression.

$ python questions.py corpus
Query: When was Python 3.0 released?
Python 3.0 was released on 3 December 2008.

$ python questions.py corpus
Query: How do neurons connect in a neural network?
Neurons of one layer connect only to neurons of the immediately preceding and immediately following layers.
```

## Brief description
The question answering (QA) system will perform two tasks: document retrieval and passage retrieval. The system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined. 

To find the most relevant documents, tf-idf is used to rank documents based both on term frequency for words in the query as well as inverse document frequency for words in the query. Once the most relevant documents are found, there many possible metrics for scoring passages, but a combination of inverse document frequency and a query term density measure are used. 

## Potential next steps
1. Analyzing the type of question word used
2. Looking for synonyms of query words
2. Lemmatization to handle different forms of the same word
