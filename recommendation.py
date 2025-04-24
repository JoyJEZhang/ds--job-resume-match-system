import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ResumeRecommendationSystem:
    def __init__(self, resume_file='Resume.csv', jd_file='jd.csv', matches_file='resume_jd_keyword_matches.csv'):
        """
        Initialize the recommendation system with data files and set up initial thresholds.
        """
        # Load data
        print("Loading resume-JD recommendation system...")
        try:
            self.resume_df = pd.read_csv(resume_file)
            self.jd_df = pd.read_csv(jd_file)
            self.matches_df = pd.read_csv(matches_file)
            
            # Add Quality column if missing
            if 'Quality' not in self.matches_df.columns:
                # Set initial thresholds based on score distribution
                scores = self.matches_df['Score']
                if len(scores) > 0:
                    # Use percentile-based thresholds for more meaningful categorization
                    self.good_threshold = scores.quantile(0.75)  # Top 25% are "good"
                    self.poor_threshold = scores.quantile(0.25)  # Bottom 25% are "poor"
                else:
                    self.good_threshold = 0.03
                    self.poor_threshold = 0.0
                
                # Add quality labels
                self.matches_df['Quality'] = self.matches_df['Score'].apply(self.get_match_quality)
            else:
                # Extract thresholds from existing data
                good_matches = self.matches_df[self.matches_df['Quality'] == 'good']
                poor_matches = self.matches_df[self.matches_df['Quality'] == 'poor']
                
                if not good_matches.empty and not poor_matches.empty:
                    self.good_threshold = good_matches['Score'].min()
                    self.poor_threshold = poor_matches['Score'].max()
                else:
                    self.good_threshold = 0.03
                    self.poor_threshold = 0.0
            
            print(f"Data loaded successfully. Found {len(self.resume_df)} resumes, {len(self.jd_df)} job descriptions, and {len(self.matches_df)} matches.")
            print(f"Current thresholds - Good: {self.good_threshold:.3f}, Poor: {self.poor_threshold:.3f}")
            
            # Load feedback if available
            self.user_feedback = {}
            if os.path.exists('user_feedback.csv'):
                feedback_df = pd.read_csv('user_feedback.csv')
                for _, row in feedback_df.iterrows():
                    self.user_feedback[(row['Resume_ID'], row['JD_idx'])] = row['Feedback']
                print(f"Loaded {len(self.user_feedback)} user feedback entries.")
            
            # Initialize machine learning model for adaptive scoring
            self.ml_model = None
            self.ml_model_trained = False
        
        except Exception as e:
            print(f"Error initializing recommendation system: {e}")
            raise
    
    def get_top_recommendations(self, resume_id, top_n=5):
        """
        Get top job recommendations for a specific resume.
        """
        try:
            if resume_id not in self.resume_df['ID'].values:
                print(f"Resume ID {resume_id} not found in dataset.")
                return None
            
            # Get matches for this resume
            resume_matches = self.matches_df[self.matches_df['Resume_ID'] == resume_id]
            
            if resume_matches.empty:
                print(f"No matches found for Resume ID {resume_id}.")
                return None
            
            # Sort by score and take top N
            top_matches = resume_matches.sort_values('Score', ascending=False).head(top_n)
            
            # Merge with JD data for more information
            result = pd.merge(
                top_matches,
                self.jd_df,
                left_on='JD_idx',
                right_index=True,
                suffixes=('', '_full')
            )
            
            # Select relevant columns
            result = result.rename(columns={'title': 'Job_Title', 'company': 'Company'})
            result = result[['Resume_ID', 'JD_idx', 'Job_Title', 'Company', 'Score', 'Quality']]
            
            return result
        
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None
    
    def get_match_quality(self, score):
        """
        Determine match quality based on score thresholds.
        """
        if score >= self.good_threshold:
            return 'good'
        elif score <= self.poor_threshold:
            return 'poor'
        else:
            return 'average'
    
    def get_recommendations_for_category(self, category, top_n=20):
        """
        Get top resume-JD matches for a specific job category.
        """
        try:
            # Filter JDs by category
            category_jds = self.jd_df[self.jd_df['category'].str.lower() == category.lower()]
            
            if category_jds.empty:
                print(f"No job descriptions found for category: {category}")
                return None
            
            # Get all matches for these JDs
            jd_indices = category_jds.index.tolist()
            category_matches = self.matches_df[self.matches_df['JD_idx'].isin(jd_indices)]
            
            if category_matches.empty:
                print(f"No matches found for category: {category}")
                return None
            
            # Sort by score and take top N
            top_matches = category_matches.sort_values('Score', ascending=False).head(top_n)
            
            # Merge with JD data for more information
            result = pd.merge(
                top_matches,
                self.jd_df,
                left_on='JD_idx',
                right_index=True,
                suffixes=('', '_full')
            )
            
            # Select relevant columns
            result = result.rename(columns={'title': 'Job_Title', 'company': 'Company'})
            result = result[['Resume_ID', 'JD_idx', 'Job_Title', 'Company', 'Score', 'Quality']]
            
            return result
        
        except Exception as e:
            print(f"Error getting category recommendations: {e}")
            return None
    
    def collect_user_feedback(self, resume_id, jd_idx, feedback):
        """
        Collect and store user feedback on match quality.
        """
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
        
        # Periodically update thresholds based on feedback
        if len(self.user_feedback) % 5 == 0:  # After every 5 feedback entries
            self.adjust_thresholds_from_feedback()
            
        return True
    
    def adjust_thresholds_from_feedback(self):
        """
        Adaptively adjust match quality thresholds based on user feedback.
        Uses quantile analysis with outlier handling for robust threshold setting.
        """
        if not self.user_feedback or len(self.user_feedback) < 10:
            print("Not enough feedback to adjust thresholds reliably.")
            return
        
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
        
        if merged_df.empty:
            print("No matching feedback data found in matches.")
            return
        
        # Check if we have all feedback categories
        if not all(category in merged_df['Feedback'].values for category in ['good', 'average', 'poor']):
            print("Need feedback for all categories (good, average, poor) to calculate thresholds")
            return
        
        # Calculate optimal thresholds with outlier handling
        # For 'good' threshold, use 25th percentile as conservative estimate
        good_scores = merged_df[merged_df['Feedback'] == 'good']['Score']
        poor_scores = merged_df[merged_df['Feedback'] == 'poor']['Score']
        
        # Use robust statistics (IQR-based) to filter outliers
        def filter_outliers(scores):
            Q1 = scores.quantile(0.25)
            Q3 = scores.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return scores[(scores >= lower_bound) & (scores <= upper_bound)]
        
        # Apply outlier filtering
        good_scores_filtered = filter_outliers(good_scores)
        poor_scores_filtered = filter_outliers(poor_scores)
        
        # Use confidence-weighted thresholds based on sample size
        def confidence_weighted_threshold(scores, percentile, min_samples=5):
            if len(scores) < min_samples:
                weight = len(scores) / min_samples  # Reduced confidence with fewer samples
                # Blend with original threshold
                return scores.quantile(percentile) * weight + self.good_threshold * (1-weight)
            return scores.quantile(percentile)
        
        # Calculate new thresholds
        new_good_threshold = confidence_weighted_threshold(good_scores_filtered, 0.25)
        new_poor_threshold = confidence_weighted_threshold(poor_scores_filtered, 0.75)
        
        # Apply temporal smoothing to avoid dramatic threshold changes
        # Blend new thresholds with old ones (80% new, 20% old)
        smoothed_good = 0.8 * new_good_threshold + 0.2 * self.good_threshold
        smoothed_poor = 0.8 * new_poor_threshold + 0.2 * self.poor_threshold
        
        # Ensure thresholds don't cross
        if smoothed_good <= smoothed_poor:
            # Find midpoint between average good and poor scores
            midpoint = (good_scores.mean() + poor_scores.mean()) / 2
            # Set thresholds equidistant from midpoint
            margin = max(0.01, (good_scores.mean() - poor_scores.mean()) / 4)
            smoothed_good = midpoint + margin
            smoothed_poor = midpoint - margin
        
        print(f"Adjusting thresholds based on {len(self.user_feedback)} feedback entries:")
        print(f"Previous - Good: {self.good_threshold:.3f}, Poor: {self.poor_threshold:.3f}")
        print(f"New      - Good: {smoothed_good:.3f}, Poor: {smoothed_poor:.3f}")
        
        # Update thresholds
        self.good_threshold = smoothed_good
        self.poor_threshold = smoothed_poor
        
        # Update quality labels in matches dataframe
        self.matches_df['Quality'] = self.matches_df['Score'].apply(self.get_match_quality)
        
        # Train ML model if enough data
        if len(merged_df) >= 30:
            self.train_ml_match_model(merged_df)
    
    def train_ml_match_model(self, feedback_data):
        """
        Train a machine learning model to predict match quality based on user feedback.
        """
        try:
            print("Training machine learning model for match quality prediction...")
            
            # Prepare training data
            X = feedback_data[['Score']]  # Currently using just score, could add more features
            
            # Convert feedback to binary labels (1 for good, 0 for not good)
            y = (feedback_data['Feedback'] == 'good').astype(int)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            test_accuracy = model.score(X_test, y_test)
            print(f"Model accuracy on test set: {test_accuracy:.2f}")
            
            # Calculate precision-recall curve
            y_scores = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            pr_auc = auc(recall, precision)
            print(f"Precision-Recall AUC: {pr_auc:.2f}")
            
            # Save model
            self.ml_model = model
            self.ml_model_trained = True
            print("Machine learning model trained successfully.")
            
            # Generate PR curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve for Match Quality Prediction')
            plt.legend(loc='best')
            plt.savefig('match_quality_pr_curve.png')
            
            return True
            
        except Exception as e:
            print(f"Error training machine learning model: {e}")
            return False
    
    def analyze_feedback(self):
        """
        Analyze user feedback to improve recommendations.
        """
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
        
        # Calculate optimal thresholds based on feedback (call existing method)
        self.adjust_thresholds_from_feedback()
        
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
        print(f"Match Score: {match['Score']:.3f} (Quality: {match['Quality']})")
        print("Please rate this match: [good] [average] [poor]")
        
        # Simulate user input
        print("Collecting sample feedback...")
        recommender.collect_user_feedback(match['Resume_ID'], match['JD_idx'], 'good')

if __name__ == "__main__":
    main() 