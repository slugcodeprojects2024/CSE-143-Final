import zipfile
import os
import pandas as pd
import re
import numpy as np
from collections import Counter
import nltk
from nltk.util import ngrams
from scipy.stats import zscore
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
from datetime import datetime

class DynamicTranscriptCleaner:
    def __init__(self, repetition_threshold_zscore=2.0):
        self.repetition_threshold_zscore = repetition_threshold_zscore
        
        # Common patterns for stage directions and timestamps
        self.patterns = {
            'timestamps': r'\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]',
            'stage_directions': [
                r'\[.*?\]',                    # [text]
                r'\(.*?\)',                    # (text)
                r'<.*?>',                      # <text>
                r'\*.*?\*',                    # *text*
                r'\{.*?\}'                     # {text}
            ]
        }
        
        self.disfluency_pattern = re.compile(
            r'\b(um+|uh+|er+|ah+|mm+|hm+|eh+)\b|'
            r'(\w+)(?:\s+\2\b)+|'
            r'(?:i mean|you know|like|so|well|right|okay|yeah)\s+(?:\1\s+)*'
        , re.IGNORECASE)
        
        self.timestamp_pattern = re.compile(self.patterns['timestamps'])
        self.stage_patterns = [re.compile(pattern) for pattern in self.patterns['stage_directions']]

    def remove_timestamps(self, text):
        """Remove timestamp markers and count them"""
        timestamps = len(self.timestamp_pattern.findall(text))
        cleaned_text = self.timestamp_pattern.sub('', text)
        return cleaned_text, timestamps

    def detect_stage_directions(self, text):
        """Detect and remove stage directions, count them"""
        stage_directions = 0
        for pattern in self.stage_patterns:
            directions = pattern.findall(text)
            stage_directions += len(directions)
            text = pattern.sub(' ', text)
        return text, stage_directions

    def find_repetitive_phrases(self, text, min_count=3):
        """Find repetitive phrases and count them"""
        words = text.lower().split()
        if len(words) < min_count:
            return set(), 0
            
        phrase_counts = {}
        
        for n in range(2, min(5, len(words))):
            phrase_list = list(ngrams(words, n))
            counts = Counter(phrase_list)
            phrase_counts.update({' '.join(phrase): count 
                               for phrase, count in counts.items()})

        if len(phrase_counts) > 1:
            counts_array = np.array(list(phrase_counts.values()))
            z_scores = zscore(counts_array)
            
            repetitive_phrases = {
                phrase for i, phrase in enumerate(phrase_counts.keys())
                if z_scores[i] > self.repetition_threshold_zscore
            }
            
            return repetitive_phrases, len(repetitive_phrases)
        return set(), 0

    def clean_text(self, text):
        """Apply cleaning steps and collect metadata"""
        if not isinstance(text, str):
            return "", {}

        metadata = {
            'original_length': len(text),
            'timestamps_removed': 0,
            'stage_directions_removed': 0,
            'repetitive_phrases_found': 0,
            'disfluencies_removed': 0
        }
        
        # Remove timestamps
        text, metadata['timestamps_removed'] = self.remove_timestamps(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove stage directions
        text, metadata['stage_directions_removed'] = self.detect_stage_directions(text)
        
        # Count disfluencies
        disfluencies = len(self.disfluency_pattern.findall(text))
        metadata['disfluencies_removed'] = disfluencies
        
        # Remove disfluencies
        text = self.disfluency_pattern.sub(' ', text)
        
        # Handle repetitive phrases
        if len(text.split()) > 10:
            repetitive_phrases, count = self.find_repetitive_phrases(text)
            metadata['repetitive_phrases_found'] = count
            for phrase in repetitive_phrases:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                occurrences = list(pattern.finditer(text))
                if len(occurrences) > 2:
                    for match in occurrences[2:-1]:
                        text = text[:match.start()] + text[match.end():]
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        metadata['cleaned_length'] = len(text)
        metadata['reduction_percentage'] = round((1 - len(text)/metadata['original_length']) * 100, 2)
        
        return text, metadata

def extract_date_from_filename(filename):
    """Extract date from filename if present"""
    date_pattern = re.compile(r'(\d{4}[-_]?\d{2}[-_]?\d{2})')
    match = date_pattern.search(filename)
    if match:
        try:
            date_str = match.group(1).replace('_', '-')
            return pd.to_datetime(date_str).strftime('%Y-%m-%d')
        except:
            return None
    return None

def process_file(args):
    """Process a single file with enhanced metadata"""
    file_path, show_dir = args
    cleaner = DynamicTranscriptCleaner()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned_text, metadata = cleaner.clean_text(content)
            
            result = {
                'show': show_dir,
                'filename': os.path.basename(file_path),
                'date': extract_date_from_filename(file_path),
                'raw_text': content,
                'cleaned_text': cleaned_text
            }
            
            # Add all metadata to result
            result.update(metadata)
            
            return result
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_infowars_data():
    start_time = time.time()
    
    print("Extracting zip file...")
    with zipfile.ZipFile('infowars-main.zip', 'r') as zip_ref:
        zip_ref.extractall('infowars-data')

    base_path = 'infowars-data/infowars-main/transcripts'
    
    files_to_process = []
    for show_dir in os.listdir(base_path):
        show_path = os.path.join(base_path, show_dir)
        if os.path.isdir(show_path):
            for root, _, files in os.walk(show_path):
                for file in files:
                    if file.endswith('.txt') or file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        files_to_process.append((file_path, show_dir))

    print(f"\nFound {len(files_to_process)} files to process")
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Processing using {num_cores} cores...")
    
    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for result in tqdm(
            executor.map(process_file, files_to_process),
            total=len(files_to_process),
            desc="Processing transcripts"
        ):
            if result is not None:
                results.append(result)

    if not results:
        print("No transcripts were successfully processed!")
        return None

    df = pd.DataFrame(results)
    
    # Additional analysis
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Save detailed and summary data
    df.to_csv('cleaned_infowars_transcripts.csv', index=False)
    
    # Create summary statistics
    summary_stats = {
        'total_transcripts': len(df),
        'total_words_original': df['original_length'].sum(),
        'total_words_cleaned': df['cleaned_length'].sum(),
        'average_reduction': df['reduction_percentage'].mean(),
        'total_timestamps_removed': df['timestamps_removed'].sum(),
        'total_stage_directions': df['stage_directions_removed'].sum(),
        'total_repetitive_phrases': df['repetitive_phrases_found'].sum(),
        'total_disfluencies': df['disfluencies_removed'].sum(),
    }
    
    if 'date' in df.columns:
        summary_stats.update({
            'date_range_start': df['date'].min(),
            'date_range_end': df['date'].max(),
        })
    
    # Save summary stats
    with open('transcript_analysis_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=4, default=str)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Processed {len(df)} transcripts")
    print("\nSummary Statistics:")
    for key, value in summary_stats.items():
        print(f"{key}: {value}")
    
    return df

if __name__ == "__main__":
    cleaned_data = process_infowars_data()