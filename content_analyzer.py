import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from textblob import TextBlob
import os
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class ContentAnalyzer:
    def __init__(self):
        base_stops = set(stopwords.words('english'))
        custom_stops = {
            # Fillers and common words
            'get', 'go', 'got', 'going', 'gone', 'gon', 'na', 'well', 'like', 'one',
            'th', 'us', 'want', 'come', 'see', 'say', 'said', 'would', 'could',
            'thing', 'way', 'yeah', 'hey', 'hi', 'hello', 'gonna', 'wanna',
            
            # Show-specific
            'alex', 'jones', 'show', 'infowars', 'years', 'ago', 'radio', 'network',
            'gcn', 'broadcast', 'caller', 'break', 'commercial',
            
            # Time references
            'today', 'tomorrow', 'yesterday', 'night', 'morning', 'evening',
            'day', 'week', 'month', 'year', 'time',
            
            # Numbers and measurements
            'one', 'two', 'three', 'first', 'second', 'third', 'thousand',
            'million', 'billion', 'number',
            
            # Common verbs
            'make', 'take', 'came', 'mean', 'need', 'give', 'look', 'know',
            'think', 'believe', 'try', 'keep', 'let', 'tell', 'told',
            
            # Audio transcript artifacts
            'uh', 'um', 'ah', 'eh', 'mm', 'hm', 'pause', 'silence', 'laughter',
            'applause', 'music', 'break'
        }
        
        self.stop_words = base_stops.union(custom_stops)
        self.min_word_length = 3

    def is_meaningful_word(self, word):
        return (
            word.isalnum() and
            len(word) >= self.min_word_length and
            word not in self.stop_words
        )

    def analyze_text(self, text):
        if not isinstance(text, str):
            return None
            
        blob = TextBlob(text)
        words = word_tokenize(text.lower())
        words = [word for word in words if self.is_meaningful_word(word)]
        
        # Get meaningful phrases only
        bigrams = [bg for bg in ngrams(words, 2) 
                  if all(self.is_meaningful_word(w) for w in bg)]
        trigrams = [tg for tg in ngrams(words, 3) 
                   if all(self.is_meaningful_word(w) for w in tg)]
        
        return {
            'sentiment': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'words': words,
            'bigrams': bigrams,
            'trigrams': trigrams
        }

    def analyze_year(self, year_file):
        df = pd.read_csv(os.path.join('yearly_transcripts', year_file))
        
        all_words = []
        all_bigrams = []
        all_trigrams = []
        sentiments = []
        subjectivities = []
        
        for text in tqdm(df['cleaned_text'], desc=f"Processing {year_file}"):
            analysis = self.analyze_text(text)
            if analysis:
                all_words.extend(analysis['words'])
                all_bigrams.extend(analysis['bigrams'])
                all_trigrams.extend(analysis['trigrams'])
                sentiments.append(analysis['sentiment'])
                subjectivities.append(analysis['subjectivity'])
        
        word_freq = Counter(all_words)
        bigram_freq = Counter(all_bigrams)
        trigram_freq = Counter(all_trigrams)
        
        return {
            'year': year_file.split('_')[1].split('.')[0],
            'avg_sentiment': sum(sentiments) / len(sentiments) if sentiments else 0,
            'avg_subjectivity': sum(subjectivities) / len(subjectivities) if subjectivities else 0,
            'top_words': dict(word_freq.most_common(50)),
            'top_bigrams': dict(bigram_freq.most_common(30)),
            'top_trigrams': dict(trigram_freq.most_common(20)),
            'sentiment_distribution': sentiments
        }

def main():
    analyzer = ContentAnalyzer()
    results = []
    os.makedirs('analysis_results', exist_ok=True)
    
    yearly_files = [f for f in os.listdir('yearly_transcripts') 
                   if f.startswith('infowars_') and f.endswith('.csv')]
    
    for year_file in sorted(yearly_files):
        year_results = analyzer.analyze_year(year_file)
        results.append(year_results)
        
        year = year_results['year']
        
        # Save frequencies
        freq_df = pd.DataFrame({
            'words': pd.Series(year_results['top_words']),
            'bigrams': pd.Series({' '.join(k): v for k, v in year_results['top_bigrams'].items()}),
            'trigrams': pd.Series({' '.join(k): v for k, v in year_results['top_trigrams'].items()})
        })
        freq_df.to_csv(f'analysis_results/frequencies_{year}.csv')
        
        # Save sentiment distribution
        pd.DataFrame({'sentiment': year_results['sentiment_distribution']}).to_csv(
            f'analysis_results/sentiment_{year}.csv')
    
    # Create overall summary
    summary_df = pd.DataFrame([{
        'year': r['year'],
        'avg_sentiment': r['avg_sentiment'],
        'avg_subjectivity': r['avg_subjectivity'],
        'top_5_words': ', '.join(list(r['top_words'].keys())[:5]),
        'top_3_bigrams': ', '.join(' '.join(bg) for bg in list(r['top_bigrams'].keys())[:3]),
        'top_2_trigrams': ', '.join(' '.join(tg) for tg in list(r['top_trigrams'].keys())[:2])
    } for r in results])
    
    summary_df.to_csv('analysis_results/yearly_content_summary.csv', index=False)

if __name__ == "__main__":
    main()