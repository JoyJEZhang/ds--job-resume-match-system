import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from recommendation import ResumeRecommendationSystem
import base64
from io import BytesIO

# Initialize the recommendation system
@st.cache_resource
def load_recommender():
    st.info("Loading recommendation system... this might take a moment.")
    return ResumeRecommendationSystem()

def get_table_download_link(df, filename, text):
    """Generate a link to download the dataframe as a CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {text}</a>'
    return href

def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generate a link to download a matplotlib plot as a PNG image"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def display_dataframe_with_styling(df, columns, gradient_column=None):
    """Safely display a dataframe with styling applied"""
    # Reset index to ensure unique indices
    display_df = df[columns].copy().reset_index(drop=True)
    
    if gradient_column and gradient_column in display_df.columns:
        styled_df = display_df.style.format({col: '{:.3f}' for col in display_df.select_dtypes(include=['float']).columns})
        
        if len(display_df) > 0:  # Only apply background gradient if there's data
            styled_df = styled_df.background_gradient(subset=[gradient_column], cmap='viridis')
            
        st.dataframe(styled_df)
    else:
        st.dataframe(display_df)

def main():
    st.set_page_config(page_title="Resume-JD Recommendation System", page_icon="📄", layout="wide")
    
    st.title("Resume-JD Recommendation System")
    
    # Initialize the recommender
    recommender = load_recommender()
    
    # Sidebar menu
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        [
            "Home",
            "Get Resume Recommendations",
            "Category Recommendations", 
            "Provide Feedback",
            "Analyze Feedback",
            "Simulation",
            "Statistics"
        ]
    )
    
    # Home page
    if page == "Home":
        st.header("Resume-Job Description Matching System")
        st.markdown("""
        ### Welcome to the Resume-JD Recommendation System!
        
        This application helps match resumes with job descriptions using natural language processing techniques.
        
        #### Features:
        - Get job recommendations for a specific resume
        - Find top matches for a job category
        - Provide feedback on match quality
        - Analyze user feedback to improve recommendations
        - Run simulations for testing
        - View match statistics
        
        #### Dataset Information:
        - Total Resumes: {0}
        - Total Job Descriptions: {1}
        - Job Categories: {2}
        
        #### How to use:
        Use the sidebar on the left to navigate between different sections of the application.
        """.format(
            len(recommender.resume_df),
            len(recommender.jd_df),
            ", ".join(recommender.jd_df['category'].unique())
        ))
        
        # Display sample data
        st.subheader("Sample Resumes")
        st.dataframe(recommender.resume_df[['ID', 'Category']].sample(5))
        
        st.subheader("Sample Job Descriptions")
        st.dataframe(recommender.jd_df[['category', 'title', 'company']].sample(5))
        
    # Get recommendations for a specific resume
    elif page == "Get Resume Recommendations":
        st.header("Resume Recommendations")
        
        # Initialize session state for resume_id if not present
        if 'selected_resume_id' not in st.session_state:
            st.session_state.selected_resume_id = None
            
        # Simple selection method
        selection_method = st.radio(
            "Select a resume",
            ["Sample IDs", "Enter ID manually", "Random ID"],
            horizontal=True
        )
        
        # Sample IDs option
        if selection_method == "Sample IDs":
            sample_ids = recommender.resume_df['ID'].sample(10).tolist()
            selected_id = st.selectbox("Choose a resume ID", sample_ids)
            if st.button("Use this ID"):
                st.session_state.selected_resume_id = int(selected_id)
                st.success(f"Selected Resume ID: {selected_id}")
        
        # Manual entry option
        elif selection_method == "Enter ID manually":
            manual_id = st.number_input("Enter resume ID", min_value=1, value=10000000, step=1)
            if st.button("Use this ID"):
                if manual_id in recommender.resume_df['ID'].values:
                    st.session_state.selected_resume_id = int(manual_id)
                    st.success(f"Selected Resume ID: {manual_id}")
                else:
                    st.error(f"Resume ID {manual_id} not found in database")
        
        # Random option
        elif selection_method == "Random ID":
            if st.button("Generate Random ID"):
                random_id = recommender.resume_df['ID'].sample(1).iloc[0]
                st.session_state.selected_resume_id = int(random_id)
                st.success(f"Selected Random Resume ID: {random_id}")
        
        # Display current selection
        if st.session_state.selected_resume_id:
            st.info(f"Current Resume ID: {st.session_state.selected_resume_id}")
            
            # Show resume information
            resume_info = recommender.resume_df[recommender.resume_df['ID'] == st.session_state.selected_resume_id]
            if not resume_info.empty:
                st.markdown(f"**Category:** {resume_info['Category'].iloc[0]}")
                
                # Get recommendations
                if st.button("Get Recommendations"):
                    with st.spinner("Finding recommendations..."):
                        recommendations = recommender.get_top_recommendations(st.session_state.selected_resume_id)
                    
                    if recommendations is not None and not recommendations.empty:
                        st.success(f"Found {len(recommendations)} recommendations")
                        
                        # Display recommendations with safe styling
                        display_dataframe_with_styling(
                            recommendations, 
                            ['Job_Title', 'Company', 'Score', 'Quality'],
                            'Score'
                        )
                        
                        # Allow downloading results
                        st.markdown(
                            get_table_download_link(recommendations, "recommendations.csv", "recommendations as CSV"),
                            unsafe_allow_html=True
                        )
                        
                        # Visual representation
                        st.subheader("Match Quality Distribution")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        quality_counts = recommendations['Quality'].value_counts()
                        quality_counts.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
                        st.pyplot(fig)
                    else:
                        st.error("No recommendations found for this resume")
    
    # Get recommendations for a job category
    elif page == "Category Recommendations":
        st.header("Category Recommendations")
        
        # Category selector
        categories = sorted(recommender.jd_df['category'].unique().tolist())
        selected_category = st.selectbox("Select job category", categories)
        
        # Number of recommendations
        num_results = st.slider("Number of results", 5, 30, 10)
        
        if st.button("Find Top Matches"):
            with st.spinner("Finding matches..."):
                matches = recommender.get_recommendations_for_category(selected_category, top_n=num_results)
            
            if matches is not None and not matches.empty:
                st.success(f"Found {len(matches)} matches for category: {selected_category}")
                
                # Display matches with safe styling
                display_dataframe_with_styling(
                    matches, 
                    ['Resume_ID', 'Job_Title', 'Score', 'Quality'],
                    'Score'
                )
                
                # Allow downloading results
                st.markdown(
                    get_table_download_link(matches, f"{selected_category}_matches.csv", "matches as CSV"),
                    unsafe_allow_html=True
                )
                
                # Visual representation
                st.subheader("Match Scores by Quality")
                fig, ax = plt.subplots(figsize=(10, 6))
                # Use hue parameter correctly to avoid warnings
                sns.boxplot(data=matches, x='Quality', y='Score', ax=ax, palette='viridis')
                st.pyplot(fig)
            else:
                st.error(f"No matches found for category: {selected_category}")
    
    # Provide feedback
    elif page == "Provide Feedback":
        st.header("Provide Feedback on Matches")
        
        # EXTREMELY SIMPLIFIED SELECTION APPROACH
        # Initialize session state variables
        if 'feedback_page_step' not in st.session_state:
            st.session_state.feedback_page_step = 1  # Step 1: Select resume, Step 2: Show recommendations
        
        if 'feedback_resume_id' not in st.session_state:
            st.session_state.feedback_resume_id = None
        
        # Step 1: Select a resume ID
        if st.session_state.feedback_page_step == 1:
            st.subheader("Step 1: Select a Resume ID")
            
            # Two simple buttons for selection method
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Use a random resume"):
                    random_id = recommender.resume_df['ID'].sample(1).iloc[0]
                    st.session_state.feedback_resume_id = int(random_id)
                    st.session_state.feedback_page_step = 2
                    st.rerun()
            
            with col2:
                # Simple dropdown of 10 sample IDs
                sample_ids = recommender.resume_df['ID'].sample(10).tolist()
                sample_selection = st.selectbox("Or select from samples:", sample_ids)
                if st.button("Use selected ID"):
                    st.session_state.feedback_resume_id = int(sample_selection)
                    st.session_state.feedback_page_step = 2
                    st.rerun()
            
            with col3:
                # Manual ID entry
                st.write("Or enter ID manually:")
                manual_id = st.number_input("Resume ID:", min_value=1, value=10000000, step=1)
                if st.button("Verify & use this ID"):
                    if manual_id in recommender.resume_df['ID'].values:
                        st.session_state.feedback_resume_id = int(manual_id)
                        st.session_state.feedback_page_step = 2
                        st.rerun()
                    else:
                        st.error(f"Resume ID {manual_id} not found.")
        
        # Step 2: Show recommendations and collect feedback
        elif st.session_state.feedback_page_step == 2:
            resume_id = st.session_state.feedback_resume_id
            
            # Show header with resume info
            resume_info = recommender.resume_df[recommender.resume_df['ID'] == resume_id]
            if not resume_info.empty:
                st.subheader(f"Providing feedback for Resume ID: {resume_id}")
                st.info(f"Resume Category: {resume_info['Category'].iloc[0]}")
            
            # Button to go back
            if st.button("← Select a different resume"):
                st.session_state.feedback_page_step = 1
                st.rerun()
            
            # Get recommendations
            with st.spinner("Finding matches..."):
                recommendations = recommender.get_top_recommendations(resume_id, top_n=5)
            
            if recommendations is not None and not recommendations.empty:
                st.success(f"Found {len(recommendations)} potential matches")
                
                # Display each match with feedback options
                for i, (_, rec) in enumerate(recommendations.iterrows()):
                    with st.container():
                        st.markdown(f"### Match {i+1}: {rec['Job_Title']}")
                        
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            st.markdown(f"**Company:** {rec['Company']}")
                            st.markdown(f"**Match Score:** {rec['Score']:.3f} (Quality: {rec['Quality']})")
                        
                        with col2:
                            # Create a unique key for each feedback radio
                            feedback_key = f"feedback_{resume_id}_{rec['JD_idx']}"
                            
                            feedback = st.radio(
                                "Rate this match:",
                                ["good", "average", "poor"],
                                key=feedback_key,
                                horizontal=True
                            )
                            
                            # Create a unique key for each submit button
                            submit_key = f"submit_{resume_id}_{rec['JD_idx']}"
                            
                            if st.button("Submit Feedback", key=submit_key):
                                success = recommender.collect_user_feedback(
                                    rec['Resume_ID'], 
                                    rec['JD_idx'], 
                                    feedback
                                )
                                if success:
                                    st.success("Feedback recorded successfully!")
                                else:
                                    st.error("Failed to record feedback")
                
                # Display existing feedback
                st.subheader("Your Feedback History")
                
                # Get feedback for this resume
                existing_feedback = {k: v for k, v in recommender.user_feedback.items() 
                                if k[0] == resume_id}
                
                if existing_feedback:
                    feedback_data = []
                    for (r_id, jd_idx), fb in existing_feedback.items():
                        # Find job title
                        job_title = "Unknown"
                        jd_row = recommender.matches_df[
                            (recommender.matches_df['Resume_ID'] == r_id) & 
                            (recommender.matches_df['JD_idx'] == jd_idx)
                        ]
                        if not jd_row.empty:
                            job_title = jd_row['JD_title'].iloc[0]
                        
                        feedback_data.append({
                            'Job Title': job_title,
                            'Feedback': fb
                        })
                    
                    if feedback_data:
                        st.table(pd.DataFrame(feedback_data))
                else:
                    st.info("No feedback recorded yet for this resume.")
            else:
                st.error("No matches found for this resume ID.")
                # Button to go back
                if st.button("Select a different resume"):
                    st.session_state.feedback_page_step = 1
                    st.rerun()
    
    # Analyze feedback
    elif page == "Analyze Feedback":
        st.header("Feedback Analysis")
        
        if not recommender.user_feedback:
            st.warning("No user feedback available for analysis.")
            if st.button("Run Simulation to Generate Sample Feedback"):
                st.session_state.run_simulation = True
                st.rerun()
        else:
            # Analyze and display results
            with st.spinner("Analyzing feedback..."):
                feedback_analysis = recommender.analyze_feedback()
            
            if feedback_analysis is not None:
                st.subheader("Feedback Analysis Results")
                
                # Display feedback statistics
                st.dataframe(
                    feedback_analysis.style.format({
                        'mean': '{:.3f}',
                        'std': '{:.3f}'
                    })
                )
                
                # Plot feedback distribution
                st.subheader("Score Distribution by Feedback Category")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Convert feedback to DataFrame
                feedback_df = pd.DataFrame([
                    {'Resume_ID': k[0], 'JD_idx': k[1], 'Feedback': v} 
                    for k, v in recommender.user_feedback.items()
                ])
                
                # Merge with matches to get scores
                merged_df = pd.merge(
                    feedback_df,
                    recommender.matches_df,
                    on=['Resume_ID', 'JD_idx']
                )
                
                # Use correct seaborn parameter format
                sns.boxplot(data=merged_df, x='Feedback', y='Score', 
                           order=['good', 'average', 'poor'], ax=ax, palette='viridis')
                ax.set_title('Distribution of Match Scores by User Feedback')
                st.pyplot(fig)
                
                # Download options
                st.markdown(
                    get_image_download_link(fig, "feedback_analysis.png", "Download plot as PNG"),
                    unsafe_allow_html=True
                )
                
                # Display threshold information
                st.subheader("Recommended Thresholds Based on Feedback")
                if all(category in merged_df['Feedback'].values for category in ['good', 'average', 'poor']):
                    good_threshold = merged_df[merged_df['Feedback'] == 'good']['Score'].quantile(0.25)
                    poor_threshold = merged_df[merged_df['Feedback'] == 'poor']['Score'].quantile(0.75)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Good Match Threshold", f"> {good_threshold:.3f}")
                    col2.metric("Average Match Range", f"{poor_threshold:.3f} - {good_threshold:.3f}")
                    col3.metric("Poor Match Threshold", f"< {poor_threshold:.3f}")
                    
                    # Update thresholds button
                    if st.button("Update System Thresholds Based on Feedback"):
                        recommender.good_threshold = good_threshold
                        recommender.poor_threshold = poor_threshold
                        st.success("Thresholds updated successfully!")
                        st.rerun()
                else:
                    st.warning("Need feedback for all categories (good, average, poor) to calculate thresholds")
    
    # Simulation
    elif page == "Simulation":
        st.header("Simulation")
        
        st.markdown("""
        Run a simulation to generate sample user feedback. 
        This is useful for testing the system and generating data for analysis.
        """)
        
        # Simulation parameters
        num_samples = st.slider("Number of samples", 10, 100, 30)
        
        # Run simulation button
        if st.button("Run Simulation") or getattr(st.session_state, 'run_simulation', False):
            if hasattr(st.session_state, 'run_simulation'):
                st.session_state.run_simulation = False
            
            with st.spinner(f"Simulating {num_samples} user feedback entries..."):
                feedback_analysis = recommender.simulate_user_validation(num_samples)
            
            if feedback_analysis is not None:
                st.success(f"Successfully simulated {num_samples} feedback entries")
                
                # Display simulation results
                st.subheader("Simulation Results")
                
                # Display feedback statistics
                st.dataframe(
                    feedback_analysis.style.format({
                        'mean': '{:.3f}',
                        'std': '{:.3f}'
                    })
                )
                
                # Convert feedback to DataFrame for visualization
                feedback_df = pd.DataFrame([
                    {'Resume_ID': k[0], 'JD_idx': k[1], 'Feedback': v} 
                    for k, v in recommender.user_feedback.items()
                ])
                
                # Merge with matches to get scores
                merged_df = pd.merge(
                    feedback_df,
                    recommender.matches_df,
                    on=['Resume_ID', 'JD_idx']
                )
                
                # Plot feedback distribution with correct seaborn parameters
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=merged_df, x='Feedback', y='Score', 
                           order=['good', 'average', 'poor'], ax=ax, palette='viridis')
                ax.set_title('Distribution of Match Scores by Simulated Feedback')
                st.pyplot(fig)
    
    # Statistics
    elif page == "Statistics":
        st.header("Match Score Statistics")
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Matches", len(recommender.matches_df))
        col2.metric("Total Resumes with Matches", recommender.matches_df['Resume_ID'].nunique())
        col3.metric("Total Job Descriptions", recommender.matches_df['JD_idx'].nunique())
        
        # Score distribution
        scores = recommender.matches_df['Score']
        st.subheader("Score Distribution")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Score", f"{scores.mean():.3f}")
        col2.metric("Median Score", f"{scores.median():.3f}")
        col3.metric("Min Score", f"{scores.min():.3f}")
        col4.metric("Max Score", f"{scores.max():.3f}")
        
        # Current thresholds
        st.subheader("Current Quality Thresholds")
        col1, col2, col3 = st.columns(3)
        col1.metric("Good Match", f"> {recommender.good_threshold:.3f}")
        col2.metric("Average Match", f"{recommender.poor_threshold:.3f} - {recommender.good_threshold:.3f}")
        col3.metric("Poor Match", f"< {recommender.poor_threshold:.3f}")
        
        # Quality distribution
        quality_counts = {
            'good': len(recommender.matches_df[recommender.matches_df['Score'] >= recommender.good_threshold]),
            'average': len(recommender.matches_df[
                (recommender.matches_df['Score'] < recommender.good_threshold) & 
                (recommender.matches_df['Score'] > recommender.poor_threshold)
            ]),
            'poor': len(recommender.matches_df[recommender.matches_df['Score'] <= recommender.poor_threshold])
        }
        
        # Plot quality distribution
        st.subheader("Quality Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        quality_df = pd.DataFrame({
            'Quality': list(quality_counts.keys()),
            'Count': list(quality_counts.values())
        })
        
        total = sum(quality_counts.values())
        quality_df['Percentage'] = quality_df['Count'] / total * 100
        
        # Use correct seaborn parameters
        sns.barplot(data=quality_df, x='Quality', y='Count', ax=ax, palette='viridis')
        
        # Add percentage labels
        for i, row in enumerate(quality_df.itertuples()):
            ax.text(i, row.Count/2, f"{row.Percentage:.1f}%", 
                    horizontalalignment='center', size='large', color='white', weight='bold')
        
        st.pyplot(fig)
        
        # Score histogram
        st.subheader("Score Histogram")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(scores, bins=30, ax=ax)
        ax.axvline(recommender.good_threshold, color='green', linestyle='--', label=f'Good Threshold ({recommender.good_threshold:.3f})')
        ax.axvline(recommender.poor_threshold, color='red', linestyle='--', label=f'Poor Threshold ({recommender.poor_threshold:.3f})')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main() 