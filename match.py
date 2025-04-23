import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text by removing special characters, 
    converting to lowercase, and lemmatizing.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Convert to lowercase and tokenize
    words = text.lower().split()
    
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    
    # Add domain-specific stopwords
    domain_stops = {'resume', 'experience', 'skill', 'year', 'work', 'position', 'job'}
    stops.update(domain_stops)
    
    return " ".join([lemmatizer.lemmatize(word) for word in words if word not in stops])

def extract_top_keywords(docs, top_n=20, ngram=(1, 3)):
    """
    Enhanced keyword extraction using TF-IDF with preprocessing
    and n-gram range for better feature representation.
    """
    # Preprocess documents
    processed_docs = [preprocess_text(doc) for doc in docs]
    
    # Use TF-IDF with custom parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=ngram,
        min_df=2,           # Ignore terms that appear in less than 2 docs
        max_df=0.9,         # Ignore terms that appear in more than 90% of docs
        max_features=10000  # Limit features to top 10,000
    )
    
    # Handle empty corpus case
    if not any(processed_docs):
        return [""] * len(docs)
    
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    keywords = []
    for row in tfidf_matrix:
        # Extract top keywords for each document
        scores = zip(row.indices, row.data)
        top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        top_terms = [feature_names[idx] for idx, _ in top_indices]
        keywords.append(" ".join(top_terms))
    
    return keywords

def match_by_keywords(res_df, jd_df, cross_category=True, threshold=0.0):
    """
    Enhanced matching function with options for cross-category matching
    and filtering by similarity threshold.
    """
    recs = []
    tfidf_kw = TfidfVectorizer(stop_words='english')
    
    # Get unique categories
    res_categories = set(res_df['cat'].unique())
    jd_categories = set(jd_df['cat'].unique())
    
    # Function to process matches for a subset of resumes and JDs
    def process_matches(rsub, jsub):
        if rsub.empty or jsub.empty:
            return
            
        # Build corpus: first all JD keywords, then resume keywords
        corpus = jsub['keywords'].tolist() + rsub['keywords'].tolist()
        mat = tfidf_kw.fit_transform(corpus)
        jd_mat = mat[:len(jsub)]
        res_mat = mat[len(jsub):]
        
        # Calculate similarity matrix
        sim = cosine_similarity(res_mat, jd_mat)
        
        # Create match records
        for i, rid in enumerate(rsub['ID']):
            for j, idx in enumerate(jsub.index):
                score = sim[i, j]
                
                # Skip low similarity scores if threshold is set
                if score < threshold:
                    continue
                    
                recs.append({
                    'Resume_ID': rid,
                    'JD_idx': idx,
                    'JD_title': jsub.at[idx, 'title'],
                    'Score': round(float(score), 3),
                    'Resume_Category': rsub.iloc[i]['cat'] if 'cat' in rsub.columns else 'unknown',
                    'JD_Category': jsub.at[idx, 'category'] if 'category' in jsub.columns else 'unknown'
                })
    
    # Process matches within same category
    for category in res_categories.intersection(jd_categories):
        rsub = res_df[res_df['cat'] == category]
        jsub = jd_df[jd_df['cat'] == category]
        process_matches(rsub, jsub)
    
    # Process cross-category matches if enabled
    if cross_category:
        # For each resume category, match with all JD categories
        for res_cat in res_categories:
            rsub = res_df[res_df['cat'] == res_cat]
            
            # Match with JDs from other categories
            for jd_cat in jd_categories:
                if jd_cat == res_cat:
                    continue  # Already processed
                    
                jsub = jd_df[jd_df['cat'] == jd_cat]
                process_matches(rsub, jsub)
    
    return pd.DataFrame(recs)

# Main execution
if __name__ == "__main__":
    print("Loading data...")
    res = pd.read_csv('Resume.csv')
    jd = pd.read_csv('jd.csv')
    
    # Normalize categories to lowercase
    res['cat'] = res['Category'].str.lower()
    jd['cat'] = jd['category'].str.lower()
    
    print("Extracting keywords from job descriptions...")
    jd_texts = jd['description'].fillna("")
    jd['keywords'] = extract_top_keywords(jd_texts, top_n=20, ngram=(1, 3))
    
    print("Extracting keywords from resumes...")
    res_texts = res['Resume_str'].fillna("")
    res['keywords'] = extract_top_keywords(res_texts, top_n=20, ngram=(1, 3))
    
    print("Matching resumes to job descriptions...")
    # Run matching with cross-category support and low threshold
    matches_kw = match_by_keywords(res, jd, cross_category=True, threshold=0.0)
    
    # Display statistics
    print(f"Generated {len(matches_kw)} matches for {matches_kw['Resume_ID'].nunique()} resumes")
    
    print("\nTop 20 matches:")
    top_matches = matches_kw.sort_values('Score', ascending=False).head(20)
    print(top_matches[['Resume_ID', 'JD_title', 'Score', 'Resume_Category', 'JD_Category']])
    
    # Show score distribution
    print("\nScore distribution:")
    print(f"Mean: {matches_kw['Score'].mean():.3f}")
    print(f"Median: {matches_kw['Score'].median():.3f}")
    print(f"Max: {matches_kw['Score'].max():.3f}")
    print(f"Min: {matches_kw['Score'].min():.3f}")
    
    # Save results
    matches_kw.to_csv('resume_jd_keyword_matches.csv', index=False)
    print("Saved matches to resume_jd_keyword_matches.csv")
