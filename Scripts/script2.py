import spacy
from spacy import displacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Sample social media posts
sample_posts = [
    "My new phone has an amazing camera!",
    "Just finished reading that book you recommended.",
    "Can't believe how quickly time flies these days.",
    "The local cafe serves the best coffee in town."
]

def analyze_syntax(text):
    # Process text with spaCy
    doc = nlp(text)
    
    # POS Tagging Analysis
    pos_analysis = []
    for token in doc:
        pos_analysis.append({
            'text': token.text,
            'pos': token.pos_,
            'tag': token.tag_,
            'explanation': spacy.explain(token.tag_)
        })
    
    # Dependency Parsing
    dep_analysis = []
    for token in doc:
        dep_analysis.append({
            'text': token.text,
            'dep': token.dep_,
            'head': token.head.text,
            'explanation': spacy.explain(token.dep_)
        })
    
    return {
        'pos_analysis': pos_analysis,
        'dep_analysis': dep_analysis,
        'doc': doc
    }

# Process and display results
for i, post in enumerate(sample_posts, 1):
    print(f"\nAnalyzing Post {i}: '{post}'\n")
    
    analysis = analyze_syntax(post)
    
    # Display POS Tagging Results
    print("Part of Speech (POS) Analysis:")
    print("-" * 80)
    print(f"{'Token':<15} {'POS':<10} {'Tag':<10} {'Explanation'}")
    print("-" * 80)
    for item in analysis['pos_analysis']:
        print(f"{item['text']:<15} {item['pos']:<10} {item['tag']:<10} {item['explanation']}")
    
    # Display Dependency Parsing Results
    print("\nDependency Parsing Analysis:")
    print("-" * 80)
    print(f"{'Token':<15} {'Dependency':<15} {'Head':<15} {'Explanation'}")
    print("-" * 80)
    for item in analysis['dep_analysis']:
        print(f"{item['text']:<15} {item['dep']:<15} {item['head']:<15} {item['explanation']}")
    
    print("\n" + "="*80 + "\n")

# Visual dependency visualization for the first post
doc = nlp(sample_posts[0])
# Generate and save dependency visualization
options = {"compact": True, "font": "Tahoma"}
html = displacy.render(doc, style="dep", options=options)

# Save visualization to HTML file
with open("dependency_plot.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Dependency visualization has been saved to 'dependency_plot.html'")

def analyze_sentence_structure(doc):
    """Analyze the syntactic structure of a sentence"""
    # Find main verb and subject
    root = [token for token in doc if token.dep_ == 'ROOT'][0]
    subjects = [token for token in doc if 'subj' in token.dep_]
    objects = [token for token in doc if 'obj' in token.dep_]
    
    print("\nSentence Structure Analysis:")
    print(f"Main Verb (ROOT): {root.text}")
    if subjects:
        print(f"Subject(s): {', '.join(token.text for token in subjects)}")
    if objects:
        print(f"Object(s): {', '.join(token.text for token in objects)}")

# Add this to your analysis loop
for post in sample_posts:
    doc = nlp(post)
    analyze_sentence_structure(doc)