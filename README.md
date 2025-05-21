# Conscious Consumer Hub

The Conscious Consumer Hub is a minimal sentiment analysis platform for exploring product reviews, evaluating sentiment, and submitting ethical business profiles. It leverages a traditional machine learning model trained on Amazon reviews for basic classification and integrates optional GPT-enhanced feedback for product insight generation.

**Live Demo:** [https://consciousconsumerapp.streamlit.app](https://consciousconsumerapp.streamlit.app)

## Features

- **Sentiment Analysis**  
  Analyze user-provided text for positive or negative sentiment using a pre-trained logistic regression model on TF-IDF features.

- **Review Search**  
  Query a subset of Amazon reviews based on keywords. Display sentiment predictions and optional insights generated via OpenAI's GPT API.

- **Business Profiles**  
  Submit business profiles promoting ethical, sustainable, or conscious business practices. View aggregated entries in a structured format.

- **GPT-4 Integration** (Optional)  
  Enhances raw model output with nuanced insights when API quota and access are available.

## Application Architecture

- **Frontend**: Streamlit-based UI with simple routing logic via session state.
- **Backend**: Pretrained `joblib`-serialized `scikit-learn` model using TF-IDF vectors.
- **Data**: Cleaned subset of Amazon product reviews used for predictions and search.
- **External API**: OpenAI ChatCompletion endpoint (optional fallback).

## Running Locally

1. Clone the repository:
   ```bash
   git clone git@github.com:JoelAlumasa/Conscious_consumer_hub.git
   cd Conscious_consumer_hub
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key (optional, for GPT-4 insights):
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

5. Launch the application:
   ```bash
   streamlit run conscious_consumer_hub.py
   ```

## File Structure

- `conscious_consumer_hub.py` — Main Streamlit app
- `sentiment_model.joblib` — Trained ML model for sentiment prediction
- `tfidf_vectorizer.joblib` — TF-IDF vectorizer used in training
- `train_subset.csv` — Labeled training data
- `test_subset.csv` — Evaluation/test data
- `business_profiles.csv` — Stored business profile entries
- `requirements.txt` — Dependency list

## GPT-4 Integration

The app includes optional calls to GPT-4 for deeper analysis and recommendations. You must set your `OPENAI_API_KEY` as an environment variable to enable this feature.

The GPT-4 section gracefully degrades if your key is missing or your quota is exceeded.

## License

This project is under the MIT License.