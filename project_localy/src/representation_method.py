import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler


def bow_representation(news_df):
    """
    Create Bag-of-Words representation from news dataframe.

    Args:
        news_df (pd.DataFrame): DataFrame with columns 'FullText' and 'Classe'

    Returns:
        tuple: (sparse matrix of BOW features, list of labels)
    """
    texts = news_df["FullText"]
    labels = news_df["Classe"]

    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)

    return bow_matrix, labels


def tfidf_representation(news_df):
    """
    Create TF-IDF representation from news dataframe.

    Args:
        news_df (pd.DataFrame): DataFrame with columns 'FullText' and 'Classe'

    Returns:
        tuple: (TF-IDF feature matrix, list of labels)
    """
    texts = news_df["FullText"].astype(str).tolist()
    labels = news_df["Classe"].tolist()

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(texts)

    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count_matrix)

    return tfidf_matrix, labels


def word2vec_representation(news_df):
    """
    Create Word2Vec document embeddings for news dataframe.

    Args:
        news_df (pd.DataFrame): DataFrame with columns 'FullText' and 'Classe'

    Returns:
        tuple: (normalized document vectors, list of labels)
    """
    vector_size = 300
    window = 10
    min_count = 5

    # Preprocess texts: lowercase and tokenize
    news_df["FullText"] = news_df["FullText"].astype(str).str.lower()
    tokenized_texts = news_df["FullText"].apply(word_tokenize).tolist()

    # Train Word2Vec model
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
    )

    def document_vector(doc):
        vectors = [w2v_model.wv[word] for word in doc if word in w2v_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vector_size)

    doc_vectors = np.array([document_vector(doc) for doc in tokenized_texts])

    # Normalize vectors between 0 and 1
    scaler = MinMaxScaler()
    doc_vectors_normalized = scaler.fit_transform(doc_vectors)

    labels = news_df["Classe"].tolist()

    return doc_vectors_normalized, labels
