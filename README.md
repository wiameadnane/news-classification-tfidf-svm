# News Classification using TF-IDF and Support Vector Machine

## Project Overview

This project implements the research paper "A Novel Text Mining Approach Based on TF-IDF and Support Vector Machine for News Classification" as part of a Natural Language Processing (NLP) mini-project. The implementation demonstrates automatic news article classification using traditional machine learning techniques.

### Academic Information
- **Student**: Wiame Adnane
- **Supervisor**: Arsalane Zarghili
- **Institution**: EuroMed University of Fez (UEMF) - School of Digital Engineering and Artificial Intelligence (EIDIA)
- **Academic Year**: 2024-2025
- **Course**: Natural Language Processing (NLP)
- **Semester**: 8

## Problem Statement

With the explosion of online textual content, particularly through weblogs and social networks, users face a massive flow of information in the form of news headlines. This abundance makes it difficult to quickly and efficiently identify relevant information. Since textual data is generally unstructured, automatic processing becomes complex. Therefore, it is essential to develop efficient methods for automatic article classification to organize this information and enable targeted access according to user needs.

## Objectives

1. **Propose an automatic news classification method** based on:
   - Text preprocessing
   - Feature extraction using TF-IDF method
   - Classification using Support Vector Machine (SVM)

2. **Evaluate the effectiveness** of this method on standard datasets (BBC and 20Newsgroup) to demonstrate performance in terms of accuracy

3. **Provide an efficient tool** for automatically sorting news into different categories to facilitate the search for relevant information and identify dominant trends

## Methodology

The approach follows a three-step pipeline:

### 1. Text Preprocessing
- **Data cleaning**: Removal of unnecessary characters (punctuation, dates, citations, off-topic sentences)
- **Case transformation**: Converting to lowercase to standardize words and avoid case-related duplicates
- **Tokenization**: Breaking text into isolated words
- **Stopword filtering**: Removal of frequent but uninformative words

### 2. Feature Extraction with TF-IDF
- Calculation of word weighting based on term frequency in a document and inverse document frequency in the corpus
- This weighting gives more importance to rare but discriminating words for classification
- Formula: `w_ij = tf_ij × log(N/df_i)`
  - `w_ij`: weight of word i in document j
  - `tf_ij`: frequency of word i in document j
  - `N`: total number of documents
  - `df_i`: number of documents containing word i

### 3. Classification with Support Vector Machine (SVM)
- Use of Nu-SVC with RBF (non-linear) kernel
- Nu parameter set to 0.5 (maximum value)
- The SVM is applied to extracted TF-IDF features, improving performance by considering only relevant words

## Dataset

### BBC Dataset
- **Source**: BBC news articles collected between 2004-2005
- **Total articles**: 2,225
- **Categories**: 5 categories
  - Business: 510 articles
  - Entertainment: 386 articles
  - Politics: 417 articles
  - Sport: 511 articles
  - Tech: 401 articles

## Implementation Details

### Libraries Used
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning algorithms (TF-IDF, SVM)
- `nltk`: Natural language processing (tokenization, stopwords)
- `re`: Regular expressions for text cleaning
- `os`: File system operations

### Key Parameters
- **TF-IDF Vectorizer**:
  - Max features: 5,000
  - N-gram range: (1,1) - unigrams only
- **SVM (Nu-SVC)**:
  - Kernel: RBF (Radial Basis Function)
  - Nu: 0.5
- **Train/Test Split**: 80/20 with stratification

## Results

### Performance Metrics

| Category      | Precision | Recall | F1-Score | Support |
|---------------|-----------|---------|-----------|---------|
| Business      | 1.00      | 0.98    | 0.99     | 102     |
| Entertainment | 1.00      | 0.99    | 0.99     | 77      |
| Politics      | 0.99      | 0.99    | 0.99     | 84      |
| Sport         | 0.98      | 1.00    | 0.99     | 102     |
| Tech          | 0.99      | 1.00    | 0.99     | 80      |

**Overall Accuracy: 99.10%**

### Comparison with Original Paper
- **Our Implementation**: 99.10% accuracy
- **Original Paper**: 97.84% accuracy
- **Improvement**: +1.26% accuracy

All categories achieved F1-scores of 0.99, indicating balanced and reliable classification across all news categories.

## Project Structure

```
Projet/
├── BBC dataset/
│   ├── business/          # Business news articles (510 files)
│   ├── entertainment/     # Entertainment news articles (386 files)
│   ├── politics/          # Politics news articles (417 files)
│   ├── sport/            # Sports news articles (511 files)
│   └── tech/             # Technology news articles (401 files)
├── Wiame Adnane Code Mini-Projet.ipynb    # Main implementation notebook
├── Wiame Adnane Rapport Mini-Projet.pdf   # Project report (French)
├── papier_MP.pdf                          # Original research paper
└── README.md                              # This file
```

## Usage

### Prerequisites
```bash
pip install pandas scikit-learn nltk
```

### Running the Code
1. Open the Jupyter notebook: `Wiame Adnane Code Mini-Projet.ipynb`
2. Ensure the BBC dataset is in the correct directory path
3. Run all cells sequentially to:
   - Load and preprocess the data
   - Extract TF-IDF features
   - Train the SVM classifier
   - Evaluate performance

### Key Functions
- `load_bbc_dataset()`: Loads BBC dataset from directory structure
- `clean_text()`: Performs text preprocessing (cleaning, normalization)
- `remove_stopwords_from_tokens()`: Removes English stopwords

## Technical Highlights

### Strengths
- **Clear methodology**: Well-defined pipeline following classical but effective approach
- **Excellent results**: 99.10% accuracy with balanced F1-scores across all categories
- **Reproducible**: Clear implementation with standard datasets
- **Improved performance**: Better results than the original paper

### Limitations
- **Traditional approach**: Uses classical ML techniques rather than modern deep learning
- **Single language**: Only tested on English news articles
- **Limited dataset diversity**: Clean, well-structured data only
- **No comparison**: No experimental comparison with other classification methods

## Future Improvements

1. **Replace TF-IDF** with richer vector representations (Word2Vec, BERT embeddings)
2. **Use deep neural networks** to better capture semantic and syntactic dependencies
3. **Test on multilingual** or noisier datasets
4. **Implement systematic comparison** with other classification algorithms
5. **Add real-time classification** capabilities for live news feeds

## Metrics Explanation

- **Precision**: Proportion of correctly classified examples among those assigned to a class
- **Recall**: Ability of the model to find all relevant examples of a class
- **F1-Score**: Harmonic mean of precision and recall (ranges 0-1, higher is better)
- **Accuracy**: Overall percentage of correctly classified articles

## References

Original Paper: "A Novel Text Mining Approach Based on TF-IDF and Support Vector Machine for News Classification" (2016)

---

*This implementation demonstrates the effectiveness of combining TF-IDF feature extraction with SVM classification for news categorization, achieving state-of-the-art results on the BBC news dataset.*
