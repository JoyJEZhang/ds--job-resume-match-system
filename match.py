import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_top_keywords(docs, top_n=10, ngram=(1, 2)):
    """
    Given a list of raw documents, compute TF‑IDF over the corpus
    and return a list of keyword strings, one per document,
    where each string is the top_n terms joined by spaces.
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram)
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    keywords = []
    for row in tfidf_matrix:
        # row is a sparse vector; convert to (idx, score) and pick top_n
        scores = zip(row.indices, row.data)
        top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        top_terms = [feature_names[idx] for idx, _ in top_indices]
        keywords.append(" ".join(top_terms))
    return keywords


# 1. Load data
res = pd.read_csv('Resume.csv')
jd = pd.read_csv('jd.csv')

# 2. Normalize categories to lowercase
res['cat'] = res['Category'].str.lower()
jd['cat'] = jd['category'].str.lower()

# 3. Extract keywords for each JD and each resume
jd_texts = jd['description'].fillna("")
res_texts = res['Resume_str'].fillna("")

jd['keywords'] = extract_top_keywords(jd_texts, top_n=10, ngram=(1, 2))
res['keywords'] = extract_top_keywords(res_texts, top_n=10, ngram=(1, 2))


def match_by_keywords(res_df, jd_df):
    """
    Match resumes to JDs by computing cosine similarity
    on the extracted keyword strings.
    """
    recs = []
    tfidf_kw = TfidfVectorizer(stop_words='english')

    # process per category
    for c in set(res_df['cat']).intersection(jd_df['cat']):
        rsub = res_df[res_df['cat'] == c]
        jsub = jd_df[jd_df['cat'] == c]
        if rsub.empty or jsub.empty:
            continue

        # build corpus: first all JD keywords, then resume keywords
        corpus = jsub['keywords'].tolist() + rsub['keywords'].tolist()
        mat = tfidf_kw.fit_transform(corpus)
        jd_mat = mat[:len(jsub)]
        res_mat = mat[len(jsub):]

        sim = cosine_similarity(res_mat, jd_mat)
        for i, rid in enumerate(rsub['ID']):
            for j, idx in enumerate(jsub.index):
                score = sim[i, j]
                recs.append({
                    'Resume_ID': rid,
                    'JD_idx': idx,
                    'JD_title': jsub.at[idx, 'title'],
                    'Score': round(float(score), 3)
                })
    return pd.DataFrame(recs)


# 4. Run matching on keywords (tune thresh as needed)
matches_kw = match_by_keywords(res, jd)

# 5. Inspect and save
print(matches_kw.sort_values('Score', ascending=False).head(20))
matches_kw.to_csv('resume_jd_keyword_matches.csv', index=False)
