import pandas as pd
import numpy as np
from recommendation import ResumeRecommendationSystem
import argparse
import sys

def display_menu():
    """Display the main menu options."""
    print("\n===== Resume-JD Recommendation System =====")
    print("1. Get recommendations for a specific resume")
    print("2. Get top matches for a job category")
    print("3. Provide feedback on a match")
    print("4. Analyze user feedback")
    print("5. Run simulation (for testing)")
    print("6. View match score statistics")
    print("7. Exit")
    return input("Enter your choice (1-7): ")

def get_resume_id(recommender):
    """Get a valid resume ID from the user."""
    while True:
        try:
            # Show sample of available resume IDs
            sample_ids = recommender.resume_df['ID'].sample(5).tolist()
            print(f"\nSample resume IDs: {sample_ids}")
            
            resume_id = input("Enter resume ID (or 'r' for random, 'b' to go back): ")
            
            if resume_id.lower() == 'b':
                return None
            elif resume_id.lower() == 'r':
                return recommender.resume_df['ID'].sample(1).iloc[0]
            else:
                resume_id = int(resume_id)
                if resume_id in recommender.resume_df['ID'].values:
                    return resume_id
                else:
                    print(f"Resume ID {resume_id} not found in the dataset.")
        except ValueError:
            print("Please enter a valid integer ID, 'r' for random, or 'b' to go back.")

def get_job_category(recommender):
    """Get a valid job category from the user."""
    # Get unique categories
    categories = recommender.jd_df['category'].unique().tolist()
    
    print("\nAvailable job categories:")
    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")
    
    while True:
        try:
            choice = input("Enter category number (or 'b' to go back): ")
            
            if choice.lower() == 'b':
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(categories):
                return categories[choice-1]
            else:
                print(f"Please enter a number between 1 and {len(categories)}.")
        except ValueError:
            print("Please enter a valid number or 'b' to go back.")

def provide_feedback(recommender):
    """Allow user to provide feedback on a match."""
    # Get resume ID
    resume_id = get_resume_id(recommender)
    if resume_id is None:
        return
    
    # Get recommendations for this resume
    recommendations = recommender.get_top_recommendations(resume_id, top_n=5)
    
    if recommendations is None or recommendations.empty:
        print(f"No recommendations found for resume ID {resume_id}")
        return
    
    # Display recommendations
    print("\nTop recommendations:")
    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {rec['Job_Title']} at {rec['Company']} (Score: {rec['Match_Score']:.3f}, Quality: {rec['Quality']})")
    
    # Get match selection
    while True:
        try:
            choice = input("\nSelect match to provide feedback (1-5, or 'b' to go back): ")
            
            if choice.lower() == 'b':
                return
            
            choice = int(choice)
            if 1 <= choice <= len(recommendations):
                selected_match = recommendations.iloc[choice-1]
                break
            else:
                print(f"Please enter a number between 1 and {len(recommendations)}.")
        except ValueError:
            print("Please enter a valid number or 'b' to go back.")
    
    # Get feedback
    while True:
        feedback = input("Rate this match (good/average/poor): ").lower()
        if feedback in ['good', 'average', 'poor']:
            recommender.collect_user_feedback(
                selected_match['Resume_ID'], 
                selected_match['JD_idx'], 
                feedback
            )
            break
        else:
            print("Please enter one of: good, average, poor")

def view_statistics(recommender):
    """Display statistics about match scores."""
    if recommender.matches_df.empty:
        print("No matches available for analysis.")
        return
    
    # Basic statistics
    print("\n=== Match Score Statistics ===")
    print(f"Total matches: {len(recommender.matches_df)}")
    print(f"Total resumes with matches: {recommender.matches_df['Resume_ID'].nunique()}")
    print(f"Total job descriptions: {recommender.matches_df['JD_idx'].nunique()}")
    
    # Score distribution
    scores = recommender.matches_df['Score']
    print("\nScore Distribution:")
    print(f"Mean: {scores.mean():.3f}")
    print(f"Median: {scores.median():.3f}")
    print(f"Min: {scores.min():.3f}")
    print(f"Max: {scores.max():.3f}")
    
    # Current thresholds
    print(f"\nCurrent quality thresholds:")
    print(f"Good match: Score > {recommender.good_threshold:.3f}")
    print(f"Poor match: Score < {recommender.poor_threshold:.3f}")
    print(f"Average match: Score between {recommender.poor_threshold:.3f} and {recommender.good_threshold:.3f}")
    
    # Quality distribution
    quality_counts = {
        'good': len(recommender.matches_df[recommender.matches_df['Score'] >= recommender.good_threshold]),
        'average': len(recommender.matches_df[
            (recommender.matches_df['Score'] < recommender.good_threshold) & 
            (recommender.matches_df['Score'] > recommender.poor_threshold)
        ]),
        'poor': len(recommender.matches_df[recommender.matches_df['Score'] <= recommender.poor_threshold])
    }
    
    total = sum(quality_counts.values())
    print("\nQuality Distribution:")
    for quality, count in quality_counts.items():
        percentage = (count / total) * 100
        print(f"{quality.capitalize()}: {count} ({percentage:.1f}%)")

def main():
    """Main function to run the recommendation app."""
    print("Initializing Resume-JD Recommendation System...")
    recommender = ResumeRecommendationSystem()
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            # Get recommendations for a resume
            resume_id = get_resume_id(recommender)
            if resume_id is not None:
                recommendations = recommender.get_top_recommendations(resume_id)
                if recommendations is not None and not recommendations.empty:
                    print(f"\nTop recommendations for Resume ID {resume_id}:")
                    print(recommendations[['Job_Title', 'Company', 'Match_Score', 'Quality']].to_string(index=False))
        
        elif choice == '2':
            # Get top matches for a category
            category = get_job_category(recommender)
            if category is not None:
                print(f"\nFinding top matches for category: {category}...")
                matches = recommender.get_recommendations_for_category(category, top_n=10)
                if matches is not None and not matches.empty:
                    print(matches[['Resume_ID', 'Job_Title', 'Match_Score', 'Quality']].to_string(index=False))
        
        elif choice == '3':
            # Provide feedback
            provide_feedback(recommender)
        
        elif choice == '4':
            # Analyze feedback
            print("\nAnalyzing user feedback...")
            feedback_analysis = recommender.analyze_feedback()
            if feedback_analysis is not None:
                print("\nFeedback Analysis:")
                print(feedback_analysis.to_string(index=False))
                print("\nFeedback analysis chart saved as 'feedback_analysis.png'")
        
        elif choice == '5':
            # Run simulation
            num_samples = input("Enter number of samples to simulate (default: 30): ")
            try:
                num_samples = int(num_samples) if num_samples else 30
            except ValueError:
                num_samples = 30
            
            print(f"\nSimulating {num_samples} user feedback entries...")
            feedback_analysis = recommender.simulate_user_validation(num_samples)
            if feedback_analysis is not None:
                print("\nSimulated Feedback Analysis:")
                print(feedback_analysis.to_string(index=False))
        
        elif choice == '6':
            # View statistics
            view_statistics(recommender)
        
        elif choice == '7':
            # Exit
            print("Exiting. Thank you for using the Resume-JD Recommendation System!")
            sys.exit(0)
        
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 