import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import openai

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# GPT-4 functions using new openai>=1.0.0 interface
def gpt4_sentiment_analysis(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that performs sentiment analysis."},
            {"role": "user", "content": f"Analyze the sentiment of this review: \"{text}\". Provide a detailed sentiment analysis. If the sentiment seems negative, provide recommendations on how the product can be improved."}
        ],
        max_tokens=450
    )
    return response.choices[0].message.content.strip()

def gpt4_review_insight(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who gives product improvement advice based on user reviews."},
            {"role": "user", "content": f"Based on the sentiment and the review, provide recommendations on how the product can be improved: \"{text}\""}
        ],
        max_tokens=450
    )
    return response.choices[0].message.content.strip()

def preprocess_text(text):
    return text.lower()

model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
df = pd.read_csv('train_subset.csv')

def search_and_analyze(query, page=0, results_per_page=10):
    query_processed = preprocess_text(query)
    matches = df[df['cleaned_text'].str.contains(query_processed, na=False)]
    start_idx = page * results_per_page
    end_idx = start_idx + results_per_page
    paginated_matches = matches.iloc[start_idx:end_idx]
    results = []
    for _, row in paginated_matches.iterrows():
        text = row['cleaned_text']
        vectorized_text = vectorizer.transform([text])
        prediction = model.predict(vectorized_text)
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        results.append((text, sentiment))
    return results, matches.shape[0]

def visualize_sentiment(sentiment_scores):
    fig, ax = plt.subplots()
    sns.barplot(x=['Positive', 'Negative'], y=[sentiment_scores.count('Positive'), sentiment_scores.count('Negative')], ax=ax)
    st.pyplot(fig)

# Session state defaults
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0
if 'nav_selection' not in st.session_state:
    st.session_state.nav_selection = "Home"

st.title("Conscious Consumer Hub")
st.sidebar.subheader("Navigation")

nav_icons = {
    "Home": "🏠",
    "Sentiment Analysis": "📊",
    "Search Amazon Reviews": "🔍",
    "Business Profiles": "🏢"
}

for page, icon in nav_icons.items():
    if st.sidebar.button(f"{icon} {page}"):
        st.session_state.nav_selection = page

# Pages
if st.session_state.nav_selection == "Home":
    st.subheader("Welcome to the Conscious Consumer Hub MVP!")
    st.image("https://media.giphy.com/media/l378c04F2fjeZ7vH2/giphy.gif", use_column_width=True)
    st.subheader("Interesting Facts")
    st.write("- Did you know that over 70% of consumers say they would pay a premium for products that are ethically produced?")
    st.write("- According to a survey, 81% of millennials expect their favorite brands to make public declarations of corporate responsibility.")

elif st.session_state.nav_selection == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    user_input = st.text_area("Type your review text here...", "")
    if st.button("Analyze Sentiment"):
        processed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([processed_input])
        prediction = model.predict(vectorized_input)
        basic_sentiment = 'Positive' if prediction == 1 else 'Negative'
        if basic_sentiment == 'Positive':
            st.success(f"Basic Sentiment Prediction: {basic_sentiment}")
        else:
            st.error(f"Basic Sentiment Prediction: {basic_sentiment}")
        try:
            detailed_sentiment = gpt4_sentiment_analysis(user_input)
            st.write("Detailed Sentiment Analysis from GPT-4:", detailed_sentiment)
        except Exception as e:
            st.error(f"Failed to get GPT-4 sentiment analysis: {str(e)}")

elif st.session_state.nav_selection == "Search Amazon Reviews":
    st.subheader("Search Amazon Reviews")
    query = st.text_input("Enter a keyword to search for reviews:")
    
    if query:
        search_results, total_results = search_and_analyze(query, st.session_state.page_number, results_per_page=10)
        total_pages = total_results // 10 + (total_results % 10 > 0)
        sentiment_scores = []

        for index, (text, sentiment) in enumerate(search_results):
            st.write(f"Review: {text}")
            if sentiment == 'Positive':
                st.success(f"Sentiment: {sentiment}")
            else:
                st.error(f"Sentiment: {sentiment}")
            sentiment_scores.append(sentiment)
            if st.button('Show GPT-4 Insights', key=f'btn_insight_{index}'):
                try:
                    insight = gpt4_review_insight(text)
                    st.text_area(f"GPT-4 Insight for Review {index + 1}", insight, height=150)
                except Exception as e:
                    st.error(f"Failed to get GPT-4 insights: {str(e)}")
            st.markdown("---")

        if sentiment_scores:
            visualize_sentiment(sentiment_scores)

        col1, col2 = st.columns(2)
        with col1:
            if st.button('Previous'):
                if st.session_state.page_number > 0:
                    st.session_state.page_number -= 1
                    st.experimental_rerun()
        with col2:
            if st.button('Next'):
                if st.session_state.page_number < total_pages - 1:
                    st.session_state.page_number += 1
                    st.experimental_rerun()

elif st.session_state.nav_selection == "Business Profiles":
    st.subheader("Business Profiles")
    with st.form("business_profile_form", clear_on_submit=True):
        business_name = st.text_input("Business Name", key="business_name")
        business_description = st.text_area("Description", key="business_description")
        business_category = st.text_input("Category", key="business_category")
        business_location = st.text_input("Location", key="business_location")
        business_website = st.text_input("Website", key="business_website")
        submit_business_button = st.form_submit_button("Submit Business")
    if submit_business_button:
        new_business_profile = pd.DataFrame({
            'Business Name': [business_name],
            'Description': [business_description],
            'Category': [business_category],
            'Location': [business_location],
            'Website': [business_website]
        })
        new_business_profile.to_csv('business_profiles.csv', mode='a', header=not os.path.exists('business_profiles.csv'), index=False)
        st.success("Business profile submitted successfully!")
    try:
        if os.path.exists('business_profiles.csv'):
            business_profiles_df = pd.read_csv('business_profiles.csv')
            for index, row in business_profiles_df.iterrows():
                st.subheader(row['Business Name'])
                st.write(row['Description'])
                st.write(f"Category: {row['Category']}")
                st.write(f"Location: {row['Location']}")
                st.write(f"Website: {row['Website']}")
                st.markdown("---")
        else:
            st.write("No business profiles to display yet.")
    except pd.errors.ParserError as e:
        st.error(f"An error occurred while reading the business profiles: {e}")
