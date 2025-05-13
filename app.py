from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)


# Load and preprocess data
def load_data():
    filepath = os.path.join('data', 'Dataset .csv')
    df = pd.read_csv(filepath)

    # Preprocessing (same as before)
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    df['Average Cost for two'] = df['Average Cost for two'].fillna(
        df['Average Cost for two'].median())
    df['Price range'] = df['Price range'].fillna(df['Price range'].median())
    df['Cost_single_person'] = df['Average Cost for two'] / 2
    df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
    df['Has Online delivery'] = df['Has Online delivery'].map({
        'Yes': 1,
        'No': 0
    })

    # Create features
    df['features'] = df['Cuisines'] + ' ' + df['Rating text'] + ' ' + df['City']
    df['features'] = df['features'].str.replace('[^\w\s]', '').str.lower()

    # TF-IDF Vectorizer
    global tfidf, cosine_sim
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return df


df = load_data()


# Recommendation function (same as before)
def get_recommendations(user_preferences, df=df, top_n=5):
    dummy_restaurant = pd.DataFrame([{
        'Restaurant Name':
        'User Preferences',
        'Cuisines':
        user_preferences.get('cuisines', ''),
        'Rating text':
        user_preferences.get('rating', 'good'),
        'City':
        user_preferences.get('city', ''),
        'Cost_single_person':
        user_preferences.get('max_price', df['Cost_single_person'].max()),
        'features':
        ' '.join([
            user_preferences.get('cuisines', ''),
            user_preferences.get('rating', 'good'),
            user_preferences.get('city', '')
        ]).lower().replace('[^\w\s]', '')
    }])

    temp_df = pd.concat([df, dummy_restaurant], ignore_index=True)
    temp_tfidf = tfidf.transform(temp_df['features'])
    temp_cosine_sim = linear_kernel(temp_tfidf, temp_tfidf)

    sim_scores = list(enumerate(temp_cosine_sim[-1]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    restaurant_indices = [i[0] for i in sim_scores[1:top_n + 1]]

    max_price = user_preferences.get('max_price', float('inf'))
    recommendations = df.iloc[restaurant_indices]
    recommendations = recommendations[recommendations['Cost_single_person'] <=
                                      max_price]

    similarity_scores = [i[1] for i in sim_scores[1:len(recommendations) + 1]]
    recommendations['similarity_score'] = similarity_scores

    return recommendations.sort_values('similarity_score', ascending=False)


# Routes
@app.route('/')
def index():
    # Get unique values for dropdowns
    cities = sorted(df['City'].unique().tolist())
    ratings = [
        'excellent', 'very good', 'good', 'average', 'poor', 'very poor'
    ]
    return render_template('index.html', cities=cities, ratings=ratings)


@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user preferences from form
    preferences = {
        'cuisines': request.form.get('cuisines'),
        'rating': request.form.get('rating'),
        'city': request.form.get('city'),
        'max_price': float(request.form.get('max_price', 1000))
    }

    # Get recommendations
    recommendations = get_recommendations(preferences)

    # Convert to list of dictionaries for template
    results = recommendations[[
        'Restaurant Name', 'Cuisines', 'Cost_single_person',
        'Aggregate rating', 'City', 'Address'
    ]].to_dict('records')

    return render_template('results.html',
                           results=results,
                           preferences=preferences)


if __name__ == '__main__':
    app.run(debug=True)
