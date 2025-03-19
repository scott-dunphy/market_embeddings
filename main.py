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
        seen_codes = set()  # Track unique MSA codes
        
        for state, city_area, code in matches:
            city_area = city_area.strip()
            
            # Only add if we haven't seen this MSA code before
            if code not in seen_codes:
                # Extract individual city names from the full area name
                # Split on common delimiters like hyphens, commas, and "and"
                city_parts = re.split(r'[-,/&]|\band\b', city_area)
                city_parts = [part.strip() for part in city_parts if part.strip()]
                
                msa_entry = {
                    'state': state,
                    'full_area_name': city_area,
                    'code': code,
                    'full_name': f"{state} - {city_area} ({code})",
                    'city_parts': city_parts,
                    'search_text': f"{state} {city_area} {' '.join(city_parts)}"
                }
                
                msa_list.append(msa_entry)
                seen_codes.add(code)
        
        return msa_list
    except Exception as e:
        st.error(f"Error loading MSA data: {str(e)}")
        return []

# Function to create text embeddings for MSAs
@st.cache_resource
def create_embeddings(msa_list):
    # Create a list of documents to vectorize
    # Each document combines all text we want to match against
    documents = [msa['search_text'] for msa in msa_list]
    
    # Create TF-IDF vectorizer with more features to improve matching
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Use 1-3 word phrases
        min_df=1,
        max_df=0.9,
        sublinear_tf=True  # Apply sublinear tf scaling
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return vectorizer, tfidf_matrix

# Function to find the closest matching MSA
def find_closest_msa(query, vectorizer, tfidf_matrix, msa_list, top_n=5):
    # Clean and expand the query
    # Try to handle common variations like "Manhattan" for "New York"
    expanded_query = query.strip()
    
    # List of common city name mappings
    city_mappings = {
        'manhattan': 'new york',
        'brooklyn': 'new york',
        'queens': 'new york',
        'bronx': 'new york',
        'staten island': 'new york',
        'hollywood': 'los angeles',
        'venice beach': 'los angeles',
        'south beach': 'miami',
        'the loop': 'chicago'
    }
    
    # Check if the query matches any known neighborhood/borough
    query_lower = query.lower()
    if query_lower in city_mappings:
        # If yes, add the major city name to the query
        expanded_query = f"{query} {city_mappings[query_lower]}"
    
    # Transform the expanded query using the same vectorizer
    query_vec = vectorizer.transform([expanded_query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get indices of sorted similarities (highest to lowest)
    sorted_indices = similarities.argsort()[::-1]
    
    # Track codes we've already added to prevent duplicates
    seen_codes = set()
    matches = []
    
    # Get top_n unique MSAs
    for idx in sorted_indices:
        code = msa_list[idx]['code']
        
        # Only add this MSA if we haven't seen it yet
        if code not in seen_codes and len(matches) < top_n:
            matches.append((msa_list[idx], similarities[idx]))
            seen_codes.add(code)
    
    return matches

# Provide option to use GitHub file or upload
data_source = st.radio(
    "Choose your MSA data source:",
    ["GitHub URL", "Upload file"]
)

msa_list = []

if data_source == "GitHub URL":
    github_url = st.text_input(
        "Enter the GitHub raw URL for your MSA file:", 
        value="https://github.com/scott-dunphy/market_embeddings/blob/main/markets.txt",
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
            
            # Show what parts of the MSA name matched
            with st.expander("See match details"):
                st.write("Individual cities in this MSA:")
                for city in msa['city_parts']:
                    st.write(f"- {city}")
            
        # Add debugging option
        if st.checkbox("Show debug info"):
            st.write("Top 10 raw matching scores:")
            # Calculate all similarities for debugging
            query_vec = vectorizer.transform([query])
            all_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            # Get indices of top 10 matches
            top10_indices = all_similarities.argsort()[-10:][::-1]
            
            # Display top 10 areas with their similarity scores
            for idx in top10_indices:
                st.write(f"{msa_list[idx]['full_name']}: {all_similarities[idx]:.4f}")
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
    NY - NJ-New York-Jersey City-White Plains (35614)
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
        seen_codes = set()  # Track unique MSA codes
        
        for state, city_area, code in matches:
            city_area = city_area.strip()
            
            # Only add if we haven't seen this MSA code before
            if code not in seen_codes:
                # Extract individual city names from the full area name
                # Split on common delimiters like hyphens, commas, and "and"
                city_parts = re.split(r'[-,/&]|\band\b', city_area)
                city_parts = [part.strip() for part in city_parts if part.strip()]
                
                msa_entry = {
                    'state': state,
                    'full_area_name': city_area,
                    'code': code,
                    'full_name': f"{state} - {city_area} ({code})",
                    'city_parts': city_parts,
                    'search_text': f"{state} {city_area} {' '.join(city_parts)}"
                }
                
                msa_list.append(msa_entry)
                seen_codes.add(code)
        
        return msa_list
    except Exception as e:
        st.error(f"Error loading MSA data: {str(e)}")
        return []
