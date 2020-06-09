import nltk
import sys, os
import string
from collections import Counter
import numpy as np

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files_content = {}
    files = os.listdir(directory)
    
    for filename in files:
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                files_content[filename] = f.read()
                
    return files_content


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    
    punctuation = list(string.punctuation)
    stopwords = nltk.corpus.stopwords.words('english')

    tokens = nltk.word_tokenize(document)
    tokens = [token.lower() for token in tokens if token not in stopwords + punctuation]
    tokens = Counter(tokens)
     
    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    
    file_idfs = {}
    
    n = len(documents)
    
    all_words = set()
    for filename in documents:
        all_words = all_words.union(documents[filename])
    
    for word in all_words:
        count = sum([1 for filename in documents if word in documents[filename]])
        file_idfs[word] = np.log(n/count)
    
    return file_idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    top_files = []
    for filename in files:
        tf_idf_score = 0
        for word in query:
            tf = files[filename][word]
            idf = idfs.get(word,0)
            tf_idf_score += tf*idf
        top_files.append((filename, tf_idf_score))
        
    top_files = sorted(top_files, key=lambda tup: -tup[1])[:n]
    
    return [filename for filename, tf_idf_score in top_files]
    


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top_sentences = []
    for sentence in sentences:
        matching_word_measure = 0
        query_term_density = 0
        for word in query:
            if word in sentences[sentence]:
                idf = idfs.get(word, 0)
                matching_word_measure += idf
                query_term_density += 1
        query_term_density /= sum(sentences[sentence].values())
        top_sentences.append((sentence, matching_word_measure, query_term_density))
   
    top_sentences = sorted(top_sentences, key=lambda tup: (-tup[1], -tup[2]))[:n]
    
    return [sentence for sentence, matching_word_measure, query_term_density in top_sentences]


if __name__ == "__main__":
    main()
