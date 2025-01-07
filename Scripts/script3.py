from textblob import TextBlob
import pandas as pd
import re

# Sample social media posts
sample_posts = [
    "This new phone is absolutely amazing! Best purchase ever! 😍",
    "Can't stand this terrible customer service. Complete waste of time 😠",
    "The weather is okay today. Might go for a walk later.",
    "Really disappointed with the quality of this product... not worth the money 😞",
    "Just had the most incredible dinner at the new restaurant! Highly recommend! ⭐⭐⭐⭐⭐",
    "Feeling neutral about the recent updates to the app.",
    "This movie was a total disaster! Worst two hours of my life 👎",
    "Super excited about my upcoming vacation! Can't wait! ✈️",
]

def clean_text(text):
    """Clean the text by removing URLs, special characters, etc."""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    # Keep emojis as they can be important for sentiment
    return text

def analyze_sentiment(text):
    """Analyze sentiment of given text using TextBlob"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Create TextBlob object
    blob = TextBlob(cleaned_text)
    
    # Get sentiment polarity (-1 to 1)
    polarity = blob.sentiment.polarity
    
    # Get sentiment subjectivity (0 to 1)
    subjectivity = blob.sentiment.subjectivity
    
    # Classify sentiment
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'sentiment': sentiment
    }

def analyze_posts(posts):
    """Analyze sentiment for a list of posts"""
    results = []
    for post in posts:
        results.append(analyze_sentiment(post))
    
    return pd.DataFrame(results)

# Analyze the posts
df = analyze_posts(sample_posts)

# Calculate overall statistics
print("\nSentiment Analysis Results:")
print("=" * 80)
print(f"\nTotal Posts Analyzed: {len(df)}")
print(f"Average Sentiment Polarity: {df['polarity'].mean():.3f}")
print(f"Average Subjectivity: {df['subjectivity'].mean():.3f}")
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# Display detailed results
print("\nDetailed Analysis:")
print("=" * 80)
for _, row in df.iterrows():
    print(f"\nOriginal Text: {row['text']}")
    print(f"Sentiment: {row['sentiment'].upper()}")
    print(f"Polarity Score: {row['polarity']:.3f}")
    print(f"Subjectivity Score: {row['subjectivity']:.3f}")
    print("-" * 80)

# Create visualization
import matplotlib.pyplot as plt

# Sentiment distribution pie chart
plt.figure(figsize=(8, 6))
sentiment_counts = df['sentiment'].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.savefig('sentiment_distribution.png')
plt.close()

# Polarity vs Subjectivity scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['polarity'], df['subjectivity'])
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.title('Polarity vs Subjectivity')
plt.grid(True)
plt.savefig('polarity_subjectivity.png')
plt.close()

# Add confidence scoring
def get_confidence_score(polarity, subjectivity):
    """Calculate a confidence score for the sentiment analysis"""
    # Higher confidence for more extreme polarity and higher subjectivity
    return abs(polarity) * (1 + subjectivity) / 2

# Add to the analysis results
df['confidence'] = df.apply(lambda x: get_confidence_score(x['polarity'], x['subjectivity']), axis=1)

# Add threshold-based classification
def get_sentiment_strength(polarity):
    """Classify sentiment strength"""
    if abs(polarity) < 0.3:
        return "weak"
    elif abs(polarity) < 0.6:
        return "moderate"
    else:
        return "strong"

df['sentiment_strength'] = df['polarity'].apply(get_sentiment_strength)