import streamlit as st
import numpy as np
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page title
st.title("MSA Finder")
st.write("Enter a city name to find the matching Metropolitan Statistical Area (MSA)")

# Function to load and parse MSA data from GitHub
@st.cache_data
def load_msa_data_from_github(github_url):
    try:
        # Make a request to the GitHub raw content URL
        response = requests.get(github_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        content = response.text
        
        # Extract MSA entries using regex
        # Pattern matches text like "IL - Chicago (12345)"
        pattern = r'([A-Z]{2})\s*-\s*([^(]+)\s*\((\d+)\)'
        matches = re.findall(pattern, content)
        
        msa_list = []
        for state, city, code in matches:
            city = city.strip()
            msa_list.append({
                'state': state,
                'city': city,
                'code': code,
                'full_name': f"{state} - {city} ({code})"
            })
        
        return msa_list
    except Exception as e:
        st.error(f"Error loading MSA data: {str(e)}")
        return []

# Function to create text embeddings for MSAs
@st.cache_resource
def create_embeddings(msa_list):
    # Extract city names for embedding
    city_texts = [msa['city'] for msa in msa_list]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(city_texts)
    
    return vectorizer, tfidf_matrix

# Function to find the closest matching MSA
def find_closest_msa(query, vectorizer, tfidf_matrix, msa_list, top_n=5):
    # Transform the query using the same vectorizer
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get indices of top matches
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Return the top matches with their similarity scores
    return [(msa_list[idx], similarities[idx]) for idx in top_indices]

# Provide option to use GitHub file or upload
data_source = st.radio(
    "Choose your MSA data source:",
    ["GitHub URL", "Upload file"]
)

msa_list = []

if data_source == "GitHub URL":
    github_url = st.text_input(
        "Enter the GitHub raw URL for your MSA file:", 
        value="https://raw.githubusercontent.com/username/repo/main/msa_data.txt",
        help="Make sure to use the 'raw' GitHub URL (https://raw.githubusercontent.com/...)"
    )
    
    if github_url:
        # Load MSA data from GitHub
        msa_list = load_msa_data_from_github(github_url)
        
        if msa_list:
            st.success(f"Successfully loaded {len(msa_list)} MSA entries from GitHub")

else:
    # File uploader for the MSA text file
    uploaded_file = st.file_uploader("Upload your MSA text file", type=["txt"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location to read it
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        # Load MSA data
        msa_list = load_msa_data(file_path)
        
        if msa_list:
            st.success(f"Successfully loaded {len(msa_list)} MSA entries from uploaded file")

if msa_list:
    # Create embeddings
    vectorizer, tfidf_matrix = create_embeddings(msa_list)
    
    # User input for city search
    query = st.text_input("Enter a city name:")
    
    if query:
        # Find closest matches
        matches = find_closest_msa(query, vectorizer, tfidf_matrix, msa_list)
        
        # Display results
        st.subheader("Matching MSAs:")
        for msa, score in matches:
            # Calculate a confidence percentage for UI purposes
            confidence = int(score * 100)
            st.write(f"{msa['full_name']} (Match confidence: {confidence}%)")
            
            # Create a progress bar to visualize match confidence
            st.progress(score)
else:
    if data_source == "Upload file" and not uploaded_file:
        st.info("Please upload a text file containing MSA information")
    
    # Show a sample of expected format
    st.markdown("""
    **Expected file format:**
    ```
    IL - Chicago (12345)
    CA - Los Angeles (23423)
    NY - New York (34535)
    ```
    """)

# Function to load MSA data from local file (kept for file upload option)
@st.cache_data
def load_msa_data(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            
        # Extract MSA entries using regex
        pattern = r'([A-Z]{2})\s*-\s*([^(]+)\s*\((\d+)\)'
        matches = re.findall(pattern, content)
        
        msa_list = []
        for state, city, code in matches:
            city = city.strip()
            msa_list.append({
                'state': state,
                'city': city,
                'code': code,
                'full_name': f"{state} - {city} ({code})"
            })
        
        return msa_list
    except Exception as e:
        st.error(f"Error loading MSA data: {str(e)}")
        return []
