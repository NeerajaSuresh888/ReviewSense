# 📦 ReviewSense

**AI-Powered Product Review Sentiment Analyzer with Purchase Intent Prediction**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Overview

ReviewSense is an advanced machine learning system designed to analyze product reviews and predict customer sentiment and purchase intent. Built with state-of-the-art NLP techniques, it provides actionable insights for e-commerce platforms, product managers, and businesses seeking to understand customer feedback at scale.

### Key Capabilities
- **Sentiment Classification**: Automatically categorizes reviews as Positive, Neutral, or Negative
- **Purchase Intent Prediction**: Estimates likelihood of purchase (High/Medium/Low)
- **Confidence Scoring**: Provides reliability metrics for each prediction
- **Batch Processing**: Analyze hundreds of reviews simultaneously
- **Interactive Dashboard**: Real-time analytics with beautiful visualizations

---

##  Features

### 1. Single Review Analysis
- Instant sentiment detection
- Purchase intent prediction
- Confidence metrics and gauges
- Word count and character statistics
- Exportable results (JSON format)

### 2. Batch Processing
- CSV file upload support
- Process multiple reviews at once
- Aggregate statistics and distribution
- Downloadable analyzed results

### 3. Analytics Dashboard
- Sentiment trend visualization over time
- Interactive pie charts and gauges
- Average confidence metrics
- Historical performance tracking

### 4. User-Friendly Interface
- Clean, modern Streamlit UI
- Responsive design
- Real-time processing feedback
- Intuitive navigation

---

##  Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11 |
| **ML Framework** | Scikit-learn |
| **NLP Processing** | NLTK |
| **Web Framework** | Streamlit |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Pickle |

---

## Model Architecture

### Sentiment Analysis
- **Algorithm**: Logistic Regression with balanced class weights
- **Features**: TF-IDF vectorization with trigrams (1-3)
- **Accuracy**: ~80%
- **Classes**: Negative (-1), Neutral (0), Positive (1)

### Purchase Intent Prediction
- **Algorithm**: Logistic Regression (multinomial)
- **Features**: Same TF-IDF vectors as sentiment model
- **Classes**: Low, Medium, High
- **Training Data**: 19,994 Amazon product reviews

### Preprocessing Pipeline
1. **Tokenization**: TweetTokenizer for handling social media-style text
2. **Cleaning**: Special character removal, preserving alphanumeric content
3. **Filtering**: Minimum 3-character word length
4. **Smart Stemming**: SnowballStemmer with negation preservation
5. **Stopword Removal**: Custom list preserving negations (not, never, won't, etc.)
6. **Vectorization**: TF-IDF with n-gram range (1,3), max_df=0.9, min_df=3

---

##  Installation

### Prerequisites
- Python 3.11 
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/ReviewSense.git
cd ReviewSense
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

5. **Run the application**
```bash
streamlit run frontend.py
```

The app will open in your browser at `http://localhost:8501`

---

##  Requirements

```txt
streamlit==1.28.0
pandas==2.1.0
numpy==1.25.0
scikit-learn==1.3.0
nltk==3.8.1
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0
wordcloud==1.9.2
```

---

##  Usage

### Single Review Analysis

1. Navigate to **"Single Review Analysis"** in the sidebar
2. Enter or paste a product review in the text area
3. Click **" Analyze Review"**
4. View results including:
   - Sentiment classification
   - Confidence score
   - Purchase intent
   - Recommendation status
5. Export results as JSON if needed

### Batch Analysis

1. Go to **"Batch Analysis"** section
2. Upload a CSV file with a `reviewText` column
3. Click **" Analyze All Reviews"**
4. View summary statistics
5. Download analyzed results as CSV

### Analytics Dashboard

- View sentiment trends over time
- Analyze overall sentiment distribution
- Check average model confidence
- Monitor performance metrics

---

## 📁 Project Structure

```
ReviewSense/
│
├── frontend.py                 # Streamlit web application
├── review.ipynb               # Model training notebook
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── models/
│   ├── sentiment_model.pkl        # Trained sentiment classifier
│   ├── purchase_intent_model.pkl  # Trained intent predictor
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
│
├── data/
│   └── reviews.csv            # Training dataset (Amazon reviews)
│
└── notebooks/
    └── review_fixed.ipynb     # Updated training notebook
```

---

## 🔬 Model Training

To retrain the models with your own data:

1. **Prepare your dataset**
   - CSV format with columns: `reviewText`, `overall` (rating), `verified`
   - Minimum ~10,000 reviews recommended

2. **Open the Jupyter notebook**
```bash
jupyter notebook review_fixed.ipynb
```

3. **Run all cells sequentially**
   - Data loading and exploration
   - Preprocessing pipeline
   - Model training and evaluation
   - Model serialization

4. **Models will be saved as**:
   - `sentiment_model.pkl`
   - `purchase_intent_model.pkl`
   - `tfidf_vectorizer.pkl`

---

##  Performance Metrics

### Sentiment Analysis
- **Overall Accuracy**: 80%
- **Precision** (Positive): 94%
- **Recall** (Positive): 84%
- **F1-Score** (Weighted): 0.82

### Purchase Intent Prediction
- **Overall Accuracy**: 88%
- **Class Balance**: Handled via stratified sampling
- **Validation**: 80-20 train-test split

---

##  Use Cases

### E-Commerce Platforms
- Automatically flag negative reviews for customer support
- Identify products with declining sentiment
- Prioritize high-intent customer feedback

### Product Management
- Track sentiment trends over product lifecycle
- Identify areas for product improvement
- Measure impact of product updates

### Market Research
- Analyze competitor product reviews
- Understand customer pain points
- Identify market opportunities

### Customer Support
- Prioritize urgent negative feedback
- Route reviews to appropriate teams
- Measure customer satisfaction

---

##  Configuration

### Model Parameters

Adjust in `review_fixed.ipynb`:

```python
# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),    # Trigrams for better context
    max_df=0.9,            # Ignore terms in >90% of documents
    min_df=3               # Minimum document frequency
)

# Sentiment Model
sentiment_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# Intent Model
intent_model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    random_state=42
)
```

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  Future Enhancements

- [ ] Multi-language support
- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Aspect-based sentiment analysis
- [ ] Real-time review monitoring
- [ ] API endpoint for integration
- [ ] Mobile application
- [ ] Database integration for history tracking
- [ ] A/B testing framework

---

##  Known Issues

- Short reviews (<5 words) may have lower accuracy
- Sarcasm detection is limited
- Requires retraining for domain-specific vocabulary

---

## Author

**Your Name**
- GitHub: Neeraja Suresh(https://github.com/NeerajaSuresh888)
- LinkedIn:https://www.linkedin.com/in/neerajasuresh/
- Email: neerajasuresh888@gmail.com

---

##  Acknowledgments

- Dataset: Amazon Product Reviews Dataset
- Inspiration: E-commerce sentiment analysis research
- Libraries: Scikit-learn, NLTK, Streamlit communities

---

