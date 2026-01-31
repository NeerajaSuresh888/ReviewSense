import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="ReviewSense - AI Review Analyzer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
    }
    .positive { background-color: #d4edda; border-color: #28a745; }
    .negative { background-color: #f8d7da; border-color: #dc3545; }
    .neutral { background-color: #fff3cd; border-color: #ffc107; }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        sentiment_model = pickle.load(f)
    with open("purchase_intent_model.pkl", "rb") as f:
        intent_model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return sentiment_model, intent_model, vectorizer

sentiment_model, intent_model, vectorizer = load_model()


def predict_review(text):
    # Preprocess text (same as training)
    import re
    import nltk
    from nltk.tokenize import TweetTokenizer, word_tokenize
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    
    try:
        stopwords.words('english')
    except:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    
    tk = TweetTokenizer()
    st_stemmer = SnowballStemmer('english')
   # stop = set(stopwords.words('english'))
    no_stem = {'not', 'no', 'never', 'wont', 'dont', 'didnt', 'very', 'too', 'really'}

    def smart_stem(word):
        word_lower = word.lower()
        if word_lower in no_stem:
            return word_lower
        return st_stemmer.stem(word_lower)
    # Tokenization
    text = ' '.join(tk.tokenize(text))
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # Keep words >= 3 chars
    text = ' '.join([w for w in word_tokenize(text) if len(w) >= 3])
    # Stemming
    text = ' '.join([smart_stem(i) for i in tk.tokenize(text)])
    # Remove stopwords
    #text = ' '.join([i for i in tk.tokenize(text) if i not in stop])
    
    # Transform text
    text_vector = vectorizer.transform([text])
    
    # Sentiment prediction
    sentiment_pred = sentiment_model.predict(text_vector)[0]
    sentiment_proba = sentiment_model.predict_proba(text_vector)[0]
    sentiment_confidence = max(sentiment_proba)
    
    # Map sentiment
    if sentiment_pred == 1:
        sentiment = "Positive"
        score = 1
    elif sentiment_pred == 0:
        sentiment = "Neutral"
        score = 0
    else:
        sentiment = "Negative"
        score = -1

    # Purchase intent prediction
    intent_pred = intent_model.predict(text_vector)[0]

    intent_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return {
        "sentiment": sentiment,
        'confidence': sentiment_confidence,
        "sentiment_score": score,
        "purchase_intent": intent_map.get(intent_pred, "Medium")
    }


# Header
st.markdown('<h1 class="main-header">📦 ReviewSense</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Product Review Sentiment Analysis with Purchase Intent Prediction</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔗 Navigation")
    page = st.radio("Select Mode", ["Single Review Analysis", "Batch Analysis", "Analytics Dashboard", "About"])
    
    st.markdown("---")
    st.header(" Model Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "80%", "2.1%")
    with col2:
        st.metric("Reviews", "19.9K", "841")
    
    st.markdown("---")
    st.info(" **Tip**: For best results, ensure reviews are in English and contain at least 10 words.")

# Main Content
if page == "Single Review Analysis":
    st.header("📝 Analyze Single Review")
    
    # Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        review_text = st.text_area(
            "Enter Product Review",
            height=200,
            placeholder="Paste or type a product review here...\n\nExample: This product is amazing! Great quality and fast delivery. Highly recommend!"
        )
    
    with col2:
        st.markdown("### Quick Stats")
        if review_text:
            word_count = len(review_text.split())
            char_count = len(review_text)
            st.metric("Words", word_count)
            st.metric("Characters", char_count)
        else:
            st.info("Enter a review to see stats")
    
    # Analyze Button
    if st.button("🔍 Analyze Review", type="primary"):
        if review_text.strip():
            with st.spinner("Analyzing review..."):
                # Simulate processing time
                import time
                time.sleep(1.5)
                
                # Get prediction
                result = predict_review(review_text)
                
                # Display Results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Main Sentiment Card
                sentiment_class = result['sentiment'].lower()
                st.markdown(f"""
                <div class="result-box {sentiment_class}">
                    <h2 style="margin: 0;">Sentiment: {result['sentiment']}</h2>
                    <h3>Confidence: {result['confidence']*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics Row
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Sentiment Score", f"{result['sentiment_score']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Purchase Intent", result['purchase_intent'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    recommendation = " Recommend" if result['sentiment'] == 'Positive' else ("⚠️ Cautious" if result['sentiment'] == 'Neutral' else "❌ Not Recommended")
                    st.metric("Recommendation", recommendation)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed Analysis
                st.subheader("📈 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confidence Gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = result['confidence'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence Level"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, 50], 'color': "#ffebee"},
                                {'range': [50, 75], 'color': "#fff3cd"},
                                {'range': [75, 100], 'color': "#d4edda"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Purchase Intent Gauge
                    intent_value_map = {"Low": 33, "Medium": 66, "High": 100}
                    intent_value = intent_value_map.get(result['purchase_intent'], 50)
                    
                    fig2 = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = intent_value,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Purchase Intent: {result['purchase_intent']}"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#764ba2"},
                            'steps': [
                                {'range': [0, 30], 'color': "#ffebee"},
                                {'range': [30, 70], 'color': "#fff3cd"},
                                {'range': [70, 100], 'color': "#d4edda"}
                            ]
                        }
                    ))
                    fig2.update_layout(height=300)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Export Results
                st.markdown("---")
                if st.button(" Export Results as JSON"):
                    import json
                    results_json = {
                        'review': review_text,
                        'timestamp': datetime.now().isoformat(),
                        **result
                    }
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(results_json, indent=2),
                        file_name=f"review_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.error("Please enter a review to analyze!")

elif page == "Batch Analysis":
    st.header("📊 Batch Review Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file with reviews", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} reviews")
        
        st.dataframe(df.head())
        
        if st.button("🚀 Analyze All Reviews"):
            progress_bar = st.progress(0)
            results = []
            
            for idx, row in df.iterrows():
                result = predict_review(str(row.get('reviewText', '')))
                results.append(result)
                progress_bar.progress((idx + 1) / len(df))
            
            df['sentiment'] = [r['sentiment'] for r in results]
            df['confidence'] = [r['confidence'] for r in results]
            df['purchase_intent'] = [r['purchase_intent'] for r in results]
            
            st.success("✅ Analysis Complete!")
            
            # Summary Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", f"{(df['sentiment'] == 'Positive').sum()}")
            with col2:
                st.metric("Neutral", f"{(df['sentiment'] == 'Neutral').sum()}")
            with col3:
                st.metric("Negative", f"{(df['sentiment'] == 'Negative').sum()}")
            
            # Download Results
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

elif page == "Analytics Dashboard":
    st.header("📈 Analytics Dashboard")
    
    # Create sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-01-25', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'positive': np.random.randint(50, 100, len(dates)),
        'neutral': np.random.randint(10, 30, len(dates)),
        'negative': np.random.randint(5, 20, len(dates))
    })
    
    # Sentiment Trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_data['date'], y=sample_data['positive'], mode='lines+markers', name='Positive', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=sample_data['date'], y=sample_data['neutral'], mode='lines+markers', name='Neutral', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=sample_data['date'], y=sample_data['negative'], mode='lines+markers', name='Negative', line=dict(color='red')))
    fig.update_layout(title='Sentiment Trends Over Time', xaxis_title='Date', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution
        sentiment_dist = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [sample_data['positive'].sum(), sample_data['neutral'].sum(), sample_data['negative'].sum()]
        })
        fig2 = px.pie(sentiment_dist, values='Count', names='Sentiment', title='Overall Sentiment Distribution',
                     color='Sentiment', color_discrete_map={'Positive':'green', 'Neutral':'orange', 'Negative':'red'})
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Average Confidence
        fig3 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 85,
            title = {'text': "Average Confidence"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "#667eea"}}
        ))
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

else:  # About Page
    st.header("ℹ️ About ReviewSense")
    
    st.markdown("""
    ##  Project Overview
    ReviewSense is an advanced AI-powered system for analyzing product reviews and predicting customer sentiment and purchase intent.
    
    ###  Technical Details
    - **Model**: Logistic Regression with TF-IDF Vectorization
    - **Accuracy**: 80%
    - **Training Data**: 19,994 product reviews
    - **Classes**: Positive, Neutral, Negative
    
    ### Features
    1. **Sentiment Classification**: Accurately categorize reviews as positive, neutral, or negative
    2. **Purchase Intent Prediction**: Estimate likelihood of purchase based on review sentiment
    3. **Confidence Scoring**: Provide reliability metrics for predictions
    4. **Batch Processing**: Analyze multiple reviews simultaneously
    5. **Interactive Visualizations**: Real-time analytics dashboard
    
    ###  Technology Stack
    - Python 3.11
    - Scikit-learn
    - NLTK
    - StreamLit
    - Plotly
    - Pandas & NumPy
    
    ---
    
   
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>ReviewSense v1.0 | Powered by Machine Learning | © 2024</p>
</div>
""", unsafe_allow_html=True)