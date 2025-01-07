import spacy
from textblob import TextBlob
import pandas as pd

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Sample social media posts
sample_posts = [
    "The new iPhone camera takes stunning photos in low light conditions!",
    "Customer service was terrible and the manager was very rude.",
    "The restaurant's pasta was delicious but their prices are too high.",
    "Really impressed with the fast delivery and careful packaging.",
]

def extract_key_phrases(doc):
    """Extract key phrases using dependency parsing"""
    key_phrases = []
    
    # Extract subject-verb-object combinations
    for token in doc:
        if token.dep_ == "ROOT":  # Main verb
            phrase = []
            # Get subject
            subjects = [tok for tok in token.lefts if tok.dep_ == "nsubj"]
            # Get object
            objects = [tok for tok in token.rights if tok.dep_ in ("dobj", "pobj")]
            
            # Construct phrase with modifiers
            for subject in subjects:
                # Get subject modifiers
                subj_modifiers = [tok for tok in subject.lefts]
                phrase.extend([mod.text for mod in subj_modifiers])
                phrase.append(subject.text)
            
            phrase.append(token.text)
            
            for obj in objects:
                # Get object modifiers
                obj_modifiers = [tok for tok in obj.lefts]
                phrase.extend([mod.text for mod in obj_modifiers])
                phrase.append(obj.text)
            
            if phrase:
                key_phrases.append(" ".join(phrase))
    
    # Extract noun phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    return {
        'key_phrases': key_phrases,
        'noun_phrases': noun_phrases
    }

def analyze_phrase_sentiment(phrase):
    """Analyze sentiment of a phrase"""
    blob = TextBlob(phrase)
    return {
        'phrase': phrase,
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

def integrated_analysis(text):
    """Perform integrated syntactic and semantic analysis"""
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract key phrases
    phrases = extract_key_phrases(doc)
    
    # Analyze sentiment for each phrase
    key_phrase_sentiment = [analyze_phrase_sentiment(phrase) 
                          for phrase in phrases['key_phrases']]
    noun_phrase_sentiment = [analyze_phrase_sentiment(phrase) 
                           for phrase in phrases['noun_phrases']]
    
    return {
        'text': text,
        'key_phrase_analysis': key_phrase_sentiment,
        'noun_phrase_analysis': noun_phrase_sentiment,
        'doc': doc
    }

# Analyze each post
for i, post in enumerate(sample_posts, 1):
    print(f"\nAnalyzing Post {i}:")
    print(f"Text: {post}")
    print("-" * 80)
    
    analysis = integrated_analysis(post)
    
    # Display key phrase analysis
    print("\nKey Phrase Analysis:")
    for phrase_data in analysis['key_phrase_analysis']:
        print(f"\nPhrase: {phrase_data['phrase']}")
        print(f"Sentiment: {phrase_data['polarity']:.3f}")
        print(f"Subjectivity: {phrase_data['subjectivity']:.3f}")
    
    # Display noun phrase analysis
    print("\nNoun Phrase Analysis:")
    for phrase_data in analysis['noun_phrase_analysis']:
        print(f"\nPhrase: {phrase_data['phrase']}")
        print(f"Sentiment: {phrase_data['polarity']:.3f}")
        print(f"Subjectivity: {phrase_data['subjectivity']:.3f}")
    
    print("\n" + "="*80)

    def analyze_aspect_sentiment(doc):
        aspects = {}
        
        for token in doc:
            # Look for nouns that have sentiment-bearing modifiers
            if token.pos_ == "NOUN":
                modifiers = [mod for mod in token.children if mod.pos_ in ("ADJ", "ADV")]
                if modifiers:
                    aspect_phrase = f"{' '.join([mod.text for mod in modifiers])} {token.text}"
                    sentiment = analyze_phrase_sentiment(aspect_phrase)
                    aspects[token.text] = sentiment
        
        return aspects

def extract_opinion_patterns(doc):
    """Extract opinion patterns using dependency relations"""
    opinions = []
    
    for token in doc:
        if token.pos_ == "ADJ":  # Find opinion words
            # Find what the opinion is about
            if token.dep_ == "amod":
                target = token.head.text
                opinions.append({
                    'opinion_word': token.text,
                    'target': target,
                    'pattern': f"amod({target}, {token.text})"
                })
                
    return opinions

# Add to main analysis
def enhanced_analysis(text):
    doc = nlp(text)
    basic_analysis = integrated_analysis(text)
    
    # Add aspect-based sentiment
    aspect_sentiments = analyze_aspect_sentiment(doc)
    
    # Add opinion patterns
    opinions = extract_opinion_patterns(doc)
    
    return {
        **basic_analysis,
        'aspect_sentiments': aspect_sentiments,
        'opinion_patterns': opinions
    }