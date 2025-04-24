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
    Enhanced text preprocessing with domain-specific term handling.
    Preserves important technical and professional terms while removing noise.
    """
    if not isinstance(text, str):
        return ""
    
    # Handle HTML and special characters
    text = re.sub(r'<[^>]+>', ' ', text)   # Remove HTML tags
    
    # Define domain-specific terms to preserve
    # These are important terms that shouldn't be removed during preprocessing
    domain_terms = {
        # Technical skills
        'python', 'java', 'javascript', 'html', 'css', 'sql', 'nosql', 'aws', 'azure', 
        'docker', 'kubernetes', 'react', 'angular', 'nodejs', 'django', 'flask',
        # Business terms
        'roi', 'kpi', 'sla', 'crm', 'erp', 'seo', 'cpa', 'cpc', 'agile', 'scrum',
        # Legal terms
        'attorney', 'counsel', 'patent', 'litigation', 'paralegal', 'compliance',
        # Financial terms
        'accounting', 'audit', 'cpa', 'gaap', 'revenue', 'budget', 'forecast',
        # Healthcare terms
        'clinical', 'diagnosis', 'medical', 'nursing', 'patient', 'pharmaceutical'
    }
    
    # Custom preprocessing to preserve domain terms
    words = []
    for word in re.findall(r'\b[a-zA-Z]+\b', text.lower()):
        if word in domain_terms:
            words.append(word)  # Preserve domain terms as is
        else:
            # Replace numbers and punctuation
            cleaned_word = re.sub(r'[^a-zA-Z]', '', word)
            if cleaned_word:
                words.append(cleaned_word)
    
    # Convert to lowercase for non-domain terms
    words = [w.lower() for w in words]
    
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    
    # Add domain-specific stopwords (common words that don't add value)
    domain_stops = {'resume', 'experience', 'skill', 'year', 'work', 'position', 'job', 
                   'career', 'knowledge', 'proficient', 'proficiency', 'responsible'}
    stops.update(domain_stops)
    
    # Process all words
    processed_words = []
    for word in words:
        if word in domain_terms:
            processed_words.append(word)  # Keep domain terms unchanged
        elif word not in stops and len(word) > 2:
            processed_words.append(lemmatizer.lemmatize(word))
    
    return " ".join(processed_words)

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
    Enhanced matching function with category-based weight adjustments,
    cross-category matching and filtering by similarity threshold.
    """
    recs = []
    tfidf_kw = TfidfVectorizer(stop_words='english')
    
    # Get unique categories
    res_categories = set(res_df['cat'].unique())
    jd_categories = set(jd_df['cat'].unique())
    
    # Category relevance matrix - weighted similarity between different categories
    # Higher value means more relevance between categories
    category_weights = {
        ('information-technology', 'digital-media'): 0.8,
        ('information-technology', 'engineering'): 0.7,
        ('business-development', 'sales'): 0.9,
        ('business-development', 'marketing'): 0.8,
        ('finance', 'accounting'): 0.9,
        ('finance', 'banking'): 0.8,
        ('advocate', 'legal'): 1.0,
        ('healthcare', 'fitness'): 0.6,
        # Add more category relationships as needed
    }
    
    # Function to get category similarity weight (symmetric relationship)
    def get_category_weight(cat1, cat2):
        if cat1 == cat2:
            return 1.0  # Same category = full weight
        
        # Check both directions (order doesn't matter)
        if (cat1, cat2) in category_weights:
            return category_weights[(cat1, cat2)]
        if (cat2, cat1) in category_weights:
            return category_weights[(cat2, cat1)]
        
        # Default weight for unrelated categories
        return 0.5  # Reduced weight for cross-category matches
    
    # Function to process matches for a subset of resumes and JDs
    def process_matches(rsub, jsub, category_weight=1.0):
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
                # Apply category weight to adjust similarity score
                raw_score = sim[i, j]
                weighted_score = raw_score * category_weight
                
                # Skill match boosting - additional boost for skill keyword matches
                resume_text = rsub.iloc[i]['keywords'].lower()
                jd_text = jsub.at[idx, 'keywords'].lower()
                
                # Count shared skill keywords (simplistic approach)
                skill_terms = ['python', 'java', 'c++', 'javascript', 'sql', 'aws', 'machine learning']
                skill_bonus = 0
                for skill in skill_terms:
                    if skill in resume_text and skill in jd_text:
                        skill_bonus += 0.01  # Small bonus per matched skill
                
                # Apply skill bonus
                final_score = min(weighted_score + skill_bonus, 1.0)  # Cap at 1.0
                
                # Skip low similarity scores if threshold is set
                if final_score < threshold:
                    continue
                    
                recs.append({
                    'Resume_ID': rid,
                    'JD_idx': idx,
                    'JD_title': jsub.at[idx, 'title'],
                    'Score': round(float(final_score), 3),
                    'Resume_Category': rsub.iloc[i]['cat'] if 'cat' in rsub.columns else 'unknown',
                    'JD_Category': jsub.at[idx, 'category'] if 'category' in jsub.columns else 'unknown',
                    'Category_Weight': category_weight
                })
    
    # Process matches within same category
    for category in res_categories.intersection(jd_categories):
        rsub = res_df[res_df['cat'] == category]
        jsub = jd_df[jd_df['cat'] == category]
        process_matches(rsub, jsub, category_weight=1.0)  # Full weight for same category
    
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
                # Apply category weight for cross-category matches
                cat_weight = get_category_weight(res_cat, jd_cat)
                process_matches(rsub, jsub, category_weight=cat_weight)
    
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
