# Resume-JD Matching System

This project automatically matches resumes to job descriptions (JDs) and categorizes them based on relevance.

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

### Option 2: Command Line Interface

Use the command-line interface for scripting or automation:
```
python recommendation_app.py
```

## Project Components

- `match.py`: Computes similarity scores between resumes and job descriptions
- `recommendation.py`: Core logic for the recommendation system
- `recommendation_app.py`: Command-line interface
- `streamlit_app.py`: Web-based user interface
- `preprocess.ipynb`: Data cleaning and preparation
- `get-jd.ipynb`: Web scraping for job descriptions

## Using the Recommendation System

The recommendation system allows you to:
1. Get job recommendations for specific resumes
2. Find top matches for job categories
3. Provide feedback on match quality
4. Analyze user feedback to improve recommendations

The system improves over time as more user feedback is collected, helping to fine-tune the matching thresholds.

## Data Files

- `Resume.csv`: Dataset containing resume information
- `jd.csv`: Job descriptions collected from LinkedIn
- `resume_jd_keyword_matches.csv`: Computed matches between resumes and job descriptions
- `user_feedback.csv`: User feedback on match quality (created when using the recommendation system)

## Project Status

This project is currently at the midpoint checkpoint. Key features implemented:
- Data collection and preprocessing
- Resume-JD matching algorithm
- Initial recommendation system
- User feedback collection and analysis
- Web-based user interface
