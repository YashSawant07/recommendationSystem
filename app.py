from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

app = Flask(__name__)

# Load and preprocess data
def load_data():
    try:
        filepath = os.path.join('data', 'dataset.csv')
        df = pd.read_csv(filepath)

        # Data cleaning and preprocessing
        df['Cuisines'] = df['Cuisines'].fillna('Unknown').str.strip()
        df['Rating text'] = df['Rating text'].fillna('Average').str.strip().str.lower()
        df['City'] = df['City'].fillna('Unknown').str.strip()

        # Handle numeric columns
        df['Average Cost for two'] = pd.to_numeric(df['Average Cost for two'], errors='coerce')
        df['Average Cost for two'] = df['Average Cost for two'].fillna(df['Average Cost for two'].median())
        df['Price range'] = pd.to_numeric(df['Price range'], errors='coerce')
        df['Price range'] = df['Price range'].fillna(df['Price range'].median())

        # Create derived columns
        df['Cost_single_person'] = df['Average Cost for two'] / 2
        df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)
        df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)

        # Create features for recommendation
        df['features'] = df['Cuisines'] + ' ' + df['Rating text'] + ' ' + df['City']
        df['features'] = df['features'].str.replace('[^\w\s]', '').str.lower()

        # Initialize TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['features'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        return df, tfidf, cosine_sim

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Load data at startup
try:
    df, tfidf, cosine_sim = load_data()
except:
    print("Failed to load data. Please check your dataset.")
    exit()

# Recommendation function
def get_recommendations(user_preferences, top_n=5):
    try:
        # Get user's city preference
        user_city = user_preferences.get('city', '').strip()
        
        # First filter by city
        city_df = df[df['City'].str.lower() == user_city.lower()].copy()
        
        if city_df.empty:
            return pd.DataFrame()

        # Create dummy restaurant based on user preferences
        dummy_features = ' '.join([
            user_preferences.get('cuisines', ''),
            user_preferences.get('rating', 'good'),
            user_preferences.get('city', '')
        ]).lower().replace('[^\w\s]', '')

        # Transform dummy features using existing TF-IDF
        dummy_tfidf = tfidf.transform([dummy_features])

        # Calculate similarity with city-filtered restaurants
        sim_scores = linear_kernel(dummy_tfidf, tfidf.transform(city_df['features']))
        sim_scores = sim_scores[0]  # Get the first (and only) row

        # Create a Series with similarity scores
        sim_series = pd.Series(sim_scores, index=city_df.index)

        # Filter by max price if specified
        max_price = user_preferences.get('max_price', float('inf'))
        filtered_df = city_df[city_df['Cost_single_person'] <= max_price].copy()

        # Get similarity scores for filtered restaurants
        filtered_sim = sim_series[filtered_df.index]

        # Sort by similarity and get top N
        recommendations = filtered_df.iloc[filtered_sim.sort_values(ascending=False).index[:top_n]]
        recommendations['similarity_score'] = filtered_sim[recommendations.index]

        return recommendations.sort_values('similarity_score', ascending=False)

    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return pd.DataFrame()

# Routes
@app.route('/cuisines')
def cuisines():
    try:
        # Extract all cuisines and split them
        all_cuisines = df['Cuisines'].str.split(',').explode()
        # Clean and get top 50 unique cuisines
        top_cuisines = all_cuisines.str.strip().value_counts().head(50).index.tolist()
        return render_template('cuisines.html', cuisines=top_cuisines)
    except Exception as e:
        return render_template('error.html', message=f"Failed to load cuisines. Error: {str(e)}")

@app.route('/')
def index():
    try:
        cities = sorted(df['City'].unique().tolist())
        ratings = ['excellent', 'very good', 'good', 'average', 'poor', 'very poor']
        return render_template('index.html', cities=cities, ratings=ratings)
    except:
        return render_template('error.html', message="Failed to load page. Please try again later.")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Validate and get user preferences
        preferences = {
            'cuisines': request.form.get('cuisines', '').strip(),
            'rating': request.form.get('rating', 'good').strip().lower(),
            'city': request.form.get('city', '').strip(),
            'max_price': float(request.form.get('max_price', 1000))
        }

        # Basic validation
        if not preferences['cuisines'] or not preferences['city']:
            return render_template('error.html', message="Please provide both cuisines and city.")

        # Get recommendations
        recommendations = get_recommendations(preferences)

        if recommendations.empty:
            return render_template('results.html', 
                                 results=None, 
                                 preferences=preferences,
                                 message="No restaurants match your criteria. Please try different preferences.")

        # Prepare results
        results = recommendations[[
            'Restaurant Name', 'Cuisines', 'Cost_single_person',
            'Aggregate rating', 'City', 'Address', 'similarity_score'
        ]].to_dict('records')

        return render_template('results.html',
                           results=results,
                           preferences=preferences)

    except ValueError:
        return render_template('error.html', message="Invalid input. Please check your values.")
    except Exception as e:
        return render_template('error.html', message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)