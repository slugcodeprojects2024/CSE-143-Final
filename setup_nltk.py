import nltk

def download_nltk_resources():
    """Download all required NLTK resources"""
    resources = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'punkt_tab',
        'vader_lexicon'
    ]
    
    print("Downloading NLTK resources...")
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

if __name__ == "__main__":
    download_nltk_resources()