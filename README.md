# Semantic Analysis Project
## Natural Language Processing for Social Media Analytics

This repository contains implementation and documentation of NLP techniques for analyzing social media text, developed as part of the Semantic Analysis unit for BSc in Data Science and Analytics.

### Team Members
- Victor Oketch Sabare (SCT213-C002-0061/2021)
- Robert Steve Onyango (SCT213-C002-0025/2021)
- Nelly Nyaboke (SCT213-C002-0033/2021)

### Project Structure
```
semantic-analysis/
├── reports/
│   ├── text_preprocessing.pdf
│   ├── syntactic_analysis.pdf
│   ├── semantic_analysis.pdf
│   └── integrated_analysis.pdf
├── scripts/
│   ├── preprocessing.py
│   ├── syntactic_analysis.py
│   ├── sentiment_analysis.py
│   └── integrated_analysis.py
└── README.md
```

### Requirements
Install the required Python packages:
```bash
pip install -r requirements.txt
```

Required packages:
- nltk
- spacy
- textblob
- pandas
- matplotlib
- python-dotenv

You'll also need to download specific NLTK and spaCy models:
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt wordnet stopwords
```

### Features
- Text preprocessing (tokenization, lemmatization, stemming)
- Syntactic analysis (POS tagging, dependency parsing)
- Semantic analysis (sentiment analysis)
- Integrated analysis combining syntactic and semantic features

### Usage
Each script can be run independently:
```bash
python scripts/preprocessing.py
python scripts/syntactic_analysis.py
python scripts/sentiment_analysis.py
python scripts/integrated_analysis.py
```

### Documentation
Detailed documentation for each component can be found in the respective report files in the `reports/` directory.

### License
This project is part of academic coursework for Semantic Analysis unit at Jomo Kenyatta University of Agriculture and Technology.

### Contact
For any queries regarding this project, please contact any of the team members:
- Victor Oketch Sabare
- Robert Steve Onyango
- Nelly Nyaboke

---
**Note**: This project was developed as part of the 4.1 BSc in Data Science and Analytics coursework.
