from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load and preprocess data
def load_data():
    # Load your dataset - make sure the path is correct
    df = pd.read_csv("Dataset.csv")

    # Data cleaning
    df.dropna(inplace=True)

    # Currency conversion
    conversion_rates = {
        'Botswana Pula(P)': 6.15,
        'Brazilian Real(R$)': 16.50,
        'Dollar($)': 83.50,
        'Emirati Diram(AED)': 22.73,
        'Indian Rupees(Rs.)': 1,
        'Indonesian Rupiah(IDR)': 0.0053,
        'NewZealand($)': 51.50,
        'Pounds(£)': 105.50,
        'Qatari Rial(QR)': 22.93,
        'Rand(R)': 4.50,
        'Sri Lankan Rupee(LKR)': 0.27,
        'Turkish Lira(TL)': 2.80
    }

    # Clean and convert cost column
    df['Average Cost for two'] = df['Average Cost for two'].astype(str).str.replace(',', '')
    df['Average Cost for two'] = pd.to_numeric(df['Average Cost for two'], errors='coerce')

    # Apply currency conversion
    df['Conversion_Rate'] = df['Currency'].map(conversion_rates)
    df['Cost_INR'] = df['Average Cost for two'] * df['Conversion_Rate']

    # Remove any rows with missing values
    df.dropna(inplace=True)

    # Calculate cost per person
    df['Cost_single_person'] = df['Cost_INR'] / 2

    # Select necessary columns - make sure 'Address' exists in your dataset
    necessary_cols = ['Restaurant Name', 'City', 'Cuisines', 'Has Table booking', 
                     'Has Online delivery', 'Aggregate rating', 'Rating text', 
                     'Cost_single_person', 'Address']

    # Filter only columns that exist in the dataframe
    necessary_cols = [col for col in necessary_cols if col in df.columns]
    df = df[necessary_cols]

    # Convert yes/no to binary
    if 'Has Table booking' in df.columns:
        df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
    if 'Has Online delivery' in df.columns:
        df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

    # Normalize features
    scaler = MinMaxScaler()
    df['Cost_single_person_norm'] = scaler.fit_transform(df[['Cost_single_person']])
    df['Aggregate rating_norm'] = scaler.fit_transform(df[['Aggregate rating']])

    # Create features for TF-IDF
    df['features'] = df['Cuisines'] + ' ' + df['Rating text'] + ' ' + df['City']
    df['features'] = df['features'].str.replace('[^\w\s]', '').str.lower()

    return df

# Load data and prepare TF-IDF matrix
df = load_data()

# Get top cities and cuisines
top_cities = df['City'].value_counts().head(80).index.tolist()
all_cuisines = df['Cuisines'].str.split(', ').explode()
top_cuisines = all_cuisines.value_counts().head(50).index.tolist()
rating_options = sorted(df['Rating text'].unique().tolist())

# Prepare TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(user_preferences, cosine_sim=cosine_sim, df=df, top_n=5):
    # Create a dummy restaurant based on user preferences
    dummy_features = ' '.join([
        user_preferences.get('cuisines', ''),
        user_preferences.get('rating', 'good'),
        user_preferences.get('city', '')
    ]).lower().replace('[^\w\s]', '')

    # Transform the dummy features using the existing TF-IDF vectorizer
    dummy_tfidf = tfidf.transform([dummy_features])

    # Calculate cosine similarity between dummy and all restaurants
    sim_scores = linear_kernel(dummy_tfidf, tfidf_matrix).flatten()

    # Get the indices of top_n most similar restaurants
    restaurant_indices = sim_scores.argsort()[-top_n-1:-1][::-1]

    # Filter by price if specified
    max_price = user_preferences.get('max_price', float('inf'))
    recommendations = df.iloc[restaurant_indices]
    recommendations = recommendations[recommendations['Cost_single_person'] <= max_price]

    # Add similarity score to results
    recommendations['similarity_score'] = sim_scores[restaurant_indices]

    return recommendations.head(top_n)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user preferences from form
        user_prefs = {
            'city': request.form.get('city'),
            'cuisines': request.form.get('cuisines'),
            'rating': request.form.get('rating'),
            'max_price': float(request.form.get('max_price', 1000))
        }

        # Get recommendations
        recommendations = get_recommendations(user_prefs)

        # Convert to list of dicts for template
        rec_list = recommendations[['Restaurant Name', 'Cuisines', 'Cost_single_person', 
                                   'Aggregate rating', 'Address', 'similarity_score']].to_dict('records')

        return render_template('index.html', 
                             cities=top_cities, 
                             cuisines=top_cuisines, 
                             ratings=rating_options,
                             recommendations=rec_list,
                             form_data=user_prefs)

    return render_template('index.html', 
                         cities=top_cities, 
                         cuisines=top_cuisines, 
                         ratings=rating_options,
                         recommendations=None,
                         form_data=None)

if __name__ == '__main__':
    app.run(debug=True)