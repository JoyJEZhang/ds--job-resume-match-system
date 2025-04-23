#!/bin/bash

# Install dependencies if needed
if [ "$1" == "--install" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the Streamlit app
echo "Starting Resume-JD Recommendation System..."
streamlit run streamlit_app.py 