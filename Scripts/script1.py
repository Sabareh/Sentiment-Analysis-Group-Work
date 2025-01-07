import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Sample social media posts
sample_posts = [
    "OMG loving these new shoes I just bought!!! #shopping #blessed",
    "Can't wait 2 see u tomorrow! It's gonna be amazing 😊",
    "Just posted my new blog post about machine learning... check it out!",
    "Running late for work again smh... traffic is the worst 😫"
]

def preprocess_with_nltk(text):
    # Initialize tools
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and numbers
    tokens = [token for token in tokens if token.isalnum()]
    
    # Stemming and Lemmatization
    stemmed = [stemmer.stem(token) for token in tokens]
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    
    return {
        'tokens': tokens,
        'stemmed': stemmed,
        'lemmatized': lemmatized
    }

def preprocess_with_spacy(text):
    # Process text with spaCy
    doc = nlp(text)
    
    # Tokenization
    tokens = [token.text.lower() for token in doc if token.text.isalnum()]
    
    # Lemmatization (spaCy doesn't include stemming)
    lemmatized = [token.lemma_.lower() for token in doc if token.text.isalnum()]
    
    return {
        'tokens': tokens,
        'lemmatized': lemmatized
    }

# Process each post with both libraries
print("NLTK Processing:\n")
for i, post in enumerate(sample_posts, 1):
    print(f"Post {i}: {post}")
    result = preprocess_with_nltk(post)
    print("Tokens:", result['tokens'])
    print("Stemmed:", result['stemmed'])
    print("Lemmatized:", result['lemmatized'])
    print()

print("\nspaCy Processing:\n")
for i, post in enumerate(sample_posts, 1):
    print(f"Post {i}: {post}")
    result = preprocess_with_spacy(post)
    print("Tokens:", result['tokens'])
    print("Lemmatized:", result['lemmatized'])
    print()