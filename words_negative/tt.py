import nltk
from nltk.corpus import wordnet
from textblob import TextBlob

# Ensure necessary NLTK resources are downloaded
nltk.download('wordnet')

# Method to get synonyms of a given word
def get_synonyms(word):
    """Retrieve a set of synonyms for the given word."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Method to classify sentiment as negative or positive
def classify_sentiment(word):
    """Classify sentiment of the word as negative or positive."""
    # Create a TextBlob object for the word
    blob = TextBlob(word)
    
    # Sentiment polarity ranges from -1 (negative) to 1 (positive)
    sentiment_score = blob.sentiment.polarity
    
    # If the sentiment score is negative, classify as negative
    if sentiment_score < 0:
        return "Negative"
    # Otherwise, classify as positive
    elif sentiment_score > 0:
        return "Positive"
    else:
        return "Neutral"

# Get synonyms for 'avoid'
synonyms_avoid = get_synonyms("avoid")

# Analyze sentiment of these synonyms
sentiment_results = {}
for synonym in synonyms_avoid:
    sentiment_results[synonym] = classify_sentiment(synonym)

# Display the sentiment classification for each synonym
for synonym, sentiment in sentiment_results.items():
    print(f"Synonym: {synonym}, Sentiment: {sentiment}")
