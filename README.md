# Resume-JD Matching System

This project automatically matches resumes to job descriptions (JDs) and categorizes them based on relevance using natural language processing techniques.

## Live Demo

Try the live deployed version of our application:
[Resume-JD Recommendation System](https://ds--job-resume-match-system.streamlit.app/)

## Setup Instructions

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Prepare data:
   - Resume data should be in `Resume.csv`
   - Job description data should be in `jd.csv`

3. Run the matching algorithm (if matches don't exist yet):
```
python match.py
```

## Running the Application

### Option 1: Streamlit Web Interface (Recommended)

Run the Streamlit web application for an easy-to-use interface:
```
streamlit run streamlit_app.py
```

Or use the provided shell script:
```
chmod +x run_app.sh
./run_app.sh
```

The Streamlit app provides a user-friendly interface with:
- Resume recommendation lookup
- Category-based browsing
- Interactive feedback collection
- Visual analytics
- Statistics dashboard
- Dynamic threshold adjustment

### Option 2: Command Line Interface

Use the command-line interface for scripting or automation:
```
python recommendation_app.py
```

## Project Components

- `match.py`: Computes similarity scores between resumes and job descriptions with domain-specific term handling
- `recommendation.py`: Core logic for the recommendation system with adaptive threshold adjustment
- `recommendation_app.py`: Command-line interface
- `streamlit_app.py`: Web-based user interface with interactive visualization
- `preprocess.ipynb`: Data cleaning and preparation
- `get-jd.ipynb`: Web scraping for job descriptions

## Key Features

### Advanced Matching Algorithm
- Domain-specific term preservation for technical skills, business terms, legal terms, etc.
- Category-based weighting system for cross-category matching
- Skill matching bonus system to boost scores for shared technical skills

### Adaptive Learning System
- Collects and analyzes user feedback on match quality
- Dynamically adjusts match quality thresholds based on feedback
- Uses robust statistical methods to handle outliers and improve accuracy
- Optional machine learning model training for match quality prediction

### Interactive User Interface
- Simple resume selection with multiple options
- Clear visualization of match distributions
- Feedback collection with intuitive rating system
- Comprehensive statistics and analytics
- Simulation capability for testing

## Using the Recommendation System

The recommendation system allows you to:
1. Get job recommendations for specific resumes
2. Find top matches for job categories
3. Provide feedback on match quality
4. Analyze user feedback to improve recommendations
5. Run simulations to generate sample feedback
6. View detailed statistics about match quality distribution

The system improves over time as more user feedback is collected, helping to fine-tune the matching thresholds.

## Data Files

- `Resume.csv`: Dataset containing resume information
- `jd.csv`: Job descriptions collected from LinkedIn
- `resume_jd_keyword_matches.csv`: Computed matches between resumes and job descriptions
- `user_feedback.csv`: User feedback on match quality (created when using the recommendation system)

## Project Status

This project has been fully implemented with:
- Advanced data collection and preprocessing
- Enhanced resume-JD matching algorithm with domain-specific optimizations
- Adaptive recommendation system with dynamic threshold adjustment
- Comprehensive user feedback collection and analysis
- Feature-rich web-based user interface
- Live deployment on Streamlit Cloud

## Adding Your Own Resumes

To add your own resumes to the system:
1. Format your resume data to match the structure in `Resume.csv`
2. Add the new entries to the CSV file
3. Restart the application to load the updated data

The system will automatically process new resumes and generate matches with existing job descriptions.
