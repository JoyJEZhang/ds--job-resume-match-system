import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ResumeRecommendationSystem:
    def __init__(self, resume_file='Resume.csv', jd_file='jd.csv', matches_file='resume_jd_keyword_matches.csv'):
        """Initialize the recommendation system with the necessary data files."""
        self.resume_df = pd.read_csv(resume_file)
        self.jd_df = pd.read_csv(jd_file)
        
        # Load matches if exists, otherwise generate them
        if os.path.exists(matches_file):
            self.matches_df = pd.read_csv(matches_file)
            print(f"Loaded {len(self.matches_df)} existing matches.")
        else:
            print("Matches file not found. Run match.py first to generate matches.")
            return
        
        # Normalize categories
        self.resume_df['cat'] = self.resume_df['Category'].str.lower()
        self.jd_df['cat'] = self.jd_df['category'].str.lower()
        
        # Create a dictionary of user feedback
        self.user_feedback = {}
        
        # Load user feedback if exists
        if os.path.exists('user_feedback.csv'):
            feedback_df = pd.read_csv('user_feedback.csv')
            for _, row in feedback_df.iterrows():
                key = (row['Resume_ID'], row['JD_idx'])
                self.user_feedback[key] = row['Feedback']
                
        # Calculate score ranges for classification
        scores = self.matches_df['Score'].values
        self.good_threshold = np.percentile(scores, 80)  # Top 20% are "good"
        self.poor_threshold = np.percentile(scores, 40)  # Bottom 40% are "poor"
        
        print(f"Score thresholds - Good: {self.good_threshold:.3f}, Poor: {self.poor_threshold:.3f}")
        print("Recommendation system initialized successfully.")
    
    def get_top_recommendations(self, resume_id, top_n=5):
        """Get top job recommendations for a specific resume."""
        # Filter matches for the given resume
        resume_matches = self.matches_df[self.matches_df['Resume_ID'] == resume_id]
        
        if resume_matches.empty:
            print(f"No matches found for resume ID {resume_id}")
            return None
        
        # Sort by score and get top N
        top_matches = resume_matches.sort_values('Score', ascending=False).head(top_n)
        
        # Enhance with JD details
        results = []
        for _, match in top_matches.iterrows():
            jd_idx = match['JD_idx']
            jd_details = self.jd_df.iloc[jd_idx]
            
            results.append({
                'Resume_ID': resume_id,
                'JD_idx': jd_idx,
                'Job_Title': match['JD_title'],
                'Company': jd_details['company'] if 'company' in jd_details else 'N/A',
                'Category': jd_details['category'] if 'category' in jd_details else 'N/A',
                'Match_Score': match['Score'],
                'Quality': self.get_match_quality(match['Score']),
                'User_Feedback': self.user_feedback.get((resume_id, jd_idx), None)
            })
        
        return pd.DataFrame(results)
    
    def get_match_quality(self, score):
        """Determine the quality of a match based on the score."""
        if score >= self.good_threshold:
            return "good"
        elif score <= self.poor_threshold:
            return "poor"
        else:
            return "average"
    
    def get_recommendations_for_category(self, category, top_n=20):
        """Get top resume-job matches for a specific job category."""
        # Filter resumes by category
        category_resumes = self.resume_df[self.resume_df['cat'] == category.lower()]
        
        if category_resumes.empty:
            print(f"No resumes found for category {category}")
            return None
        
        # Get sample of resume IDs in this category
        sample_ids = category_resumes['ID'].sample(min(5, len(category_resumes))).tolist()
        
        all_recommendations = []
        for resume_id in sample_ids:
            recs = self.get_top_recommendations(resume_id, top_n=top_n//5)
            if recs is not None:
                all_recommendations.append(recs)
        
        if not all_recommendations:
            return None
            
        return pd.concat(all_recommendations).sort_values('Match_Score', ascending=False)
    
    def collect_user_feedback(self, resume_id, jd_idx, feedback):
        """Collect user feedback on recommended matches."""
        if feedback not in ['good', 'average', 'poor']:
            print("Feedback must be one of: 'good', 'average', 'poor'")
            return False
        
        # Store feedback
        self.user_feedback[(resume_id, jd_idx)] = feedback
        
        # Save to file
        feedback_df = pd.DataFrame([{
            'Resume_ID': resume_id,
            'JD_idx': jd_idx,
            'Feedback': feedback,
            'Timestamp': pd.Timestamp.now()
        }])
        
        # Append to existing file or create new
        if os.path.exists('user_feedback.csv'):
            feedback_df.to_csv('user_feedback.csv', mode='a', header=False, index=False)
        else:
            feedback_df.to_csv('user_feedback.csv', index=False)
            
        print(f"Feedback recorded: {feedback} for Resume {resume_id} and JD {jd_idx}")
        return True
    
    def analyze_feedback(self):
        """Analyze user feedback to improve recommendations."""
        if not self.user_feedback:
            print("No user feedback available for analysis.")
            return None
        
        # Convert feedback to DataFrame
        feedback_df = pd.DataFrame([
            {'Resume_ID': k[0], 'JD_idx': k[1], 'Feedback': v} 
            for k, v in self.user_feedback.items()
        ])
        
        # Merge with matches to get scores
        merged_df = pd.merge(
            feedback_df,
            self.matches_df,
            on=['Resume_ID', 'JD_idx']
        )
        
        # Group by feedback category and analyze scores
        feedback_analysis = merged_df.groupby('Feedback')['Score'].agg(['mean', 'std', 'count']).reset_index()
        
        # Plot distribution of scores by feedback category
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Feedback', y='Score', data=merged_df, order=['good', 'average', 'poor'])
        plt.title('Distribution of Match Scores by User Feedback')
        plt.savefig('feedback_analysis.png')
        
        # Calculate optimal thresholds based on feedback
        if all(category in merged_df['Feedback'].values for category in ['good', 'average', 'poor']):
            good_threshold = merged_df[merged_df['Feedback'] == 'good']['Score'].quantile(0.25)
            poor_threshold = merged_df[merged_df['Feedback'] == 'poor']['Score'].quantile(0.75)
            
            print(f"Suggested thresholds based on user feedback:")
            print(f"Good match: Score > {good_threshold:.3f}")
            print(f"Poor match: Score < {poor_threshold:.3f}")
            print(f"Average match: Score between {poor_threshold:.3f} and {good_threshold:.3f}")
            
            # Update thresholds based on feedback
            self.good_threshold = good_threshold
            self.poor_threshold = poor_threshold
        
        return feedback_analysis
    
    def simulate_user_validation(self, num_samples=30):
        """
        Simulate user validation with a more balanced distribution of feedback.
        This improved version ensures a roughly even split between good, average, and poor ratings.
        """
        # Get a stratified sample of matches
        high_scores = self.matches_df[self.matches_df['Score'] >= self.good_threshold].sample(
            min(num_samples // 3, len(self.matches_df[self.matches_df['Score'] >= self.good_threshold]))
        )
        
        mid_scores = self.matches_df[
            (self.matches_df['Score'] < self.good_threshold) & 
            (self.matches_df['Score'] > self.poor_threshold)
        ].sample(
            min(num_samples // 3, len(self.matches_df[
                (self.matches_df['Score'] < self.good_threshold) & 
                (self.matches_df['Score'] > self.poor_threshold)
            ]))
        )
        
        low_scores = self.matches_df[self.matches_df['Score'] <= self.poor_threshold].sample(
            min(num_samples // 3, len(self.matches_df[self.matches_df['Score'] <= self.poor_threshold]))
        )
        
        # Combine samples
        sample_matches = pd.concat([high_scores, mid_scores, low_scores])
        
        # Add some randomness to feedback (80% aligned with score, 20% random)
        for _, match in sample_matches.iterrows():
            resume_id = match['Resume_ID']
            jd_idx = match['JD_idx']
            score = match['Score']
            
            # Determine feedback based on score
            if np.random.random() < 0.8:  # 80% of the time, align feedback with score
                if score >= self.good_threshold:
                    feedback = 'good'
                elif score <= self.poor_threshold:
                    feedback = 'poor'
                else:
                    feedback = 'average'
            else:  # 20% of the time, assign random feedback
                feedback = np.random.choice(['good', 'average', 'poor'])
                
            self.collect_user_feedback(resume_id, jd_idx, feedback)
            
        print(f"Simulated {len(sample_matches)} user feedback entries.")
        return self.analyze_feedback()

# Demo function
def main():
    # Initialize recommendation system
    recommender = ResumeRecommendationSystem()
    
    # Get top recommendations for a random resume
    sample_resume_id = recommender.resume_df['ID'].sample(1).iloc[0]
    print(f"\nTop recommendations for Resume ID {sample_resume_id}:")
    recommendations = recommender.get_top_recommendations(sample_resume_id)
    if recommendations is not None:
        print(recommendations[['Job_Title', 'Company', 'Match_Score', 'Quality']].to_string(index=False))
    
    # Get top matches for a category
    category = 'information-technology'
    print(f"\nTop matches for category: {category}")
    category_matches = recommender.get_recommendations_for_category(category, top_n=10)
    if category_matches is not None:
        print(category_matches[['Resume_ID', 'Job_Title', 'Match_Score', 'Quality']].to_string(index=False))
    
    # Simulate user validation
    print("\nSimulating user validation...")
    recommender.simulate_user_validation(30)
    
    # Display sample user feedback interface
    if recommendations is not None and len(recommendations) > 0:
        match = recommendations.iloc[0]
        print(f"\nSample user feedback interface:")
        print(f"Resume ID: {match['Resume_ID']}")
        print(f"Job: {match['Job_Title']} at {match['Company']}")
        print(f"Match Score: {match['Match_Score']:.3f} (Quality: {match['Quality']})")
        print("Please rate this match: [good] [average] [poor]")
        
        # Simulate user input
        print("Collecting sample feedback...")
        recommender.collect_user_feedback(match['Resume_ID'], match['JD_idx'], 'good')

if __name__ == "__main__":
    main() 