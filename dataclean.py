import pandas as pd
import os
from tqdm import tqdm

def create_spreadsheet_friendly_files():
    print("Starting summary file creation...")
    
    # Create output directory
    output_dir = 'yearly_transcripts_summary'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of yearly files
    yearly_dir = 'yearly_transcripts'
    yearly_files = [f for f in os.listdir(yearly_dir) 
                   if f.endswith('.csv') and 'summary' not in f]
    
    # Process each yearly file
    for filename in tqdm(yearly_files, desc="Processing yearly files"):
        try:
            # Read the full file
            df = pd.read_csv(os.path.join(yearly_dir, filename))
            
            # Create a summary version
            summary_df = df.copy()
            
            # Add useful metrics
            summary_df['text_preview'] = summary_df['cleaned_text'].str[:500] + '...'
            summary_df['word_count'] = summary_df['cleaned_text'].str.split().str.len()
            
            # Extract month and day from date for easier analysis
            if 'date' in summary_df.columns:
                summary_df['date'] = pd.to_datetime(summary_df['date'])
                summary_df['month'] = summary_df['date'].dt.month
                summary_df['day'] = summary_df['date'].dt.day
            
            # Calculate additional statistics
            summary_df['sentence_count'] = summary_df['cleaned_text'].str.count('[.!?]+')
            summary_df['question_count'] = summary_df['cleaned_text'].str.count(r'\?')
            summary_df['exclamation_count'] = summary_df['cleaned_text'].str.count('!')
            
            # Drop the full text columns to reduce file size
            summary_df = summary_df.drop(columns=['cleaned_text', 'raw_text'])
            
            # Reorder columns for better viewing
            column_order = ['date', 'month', 'day', 'show', 'filename', 
                          'text_preview', 'word_count', 'sentence_count',
                          'question_count', 'exclamation_count',
                          'timestamps_removed', 'stage_directions_removed',
                          'repetitive_phrases_found', 'disfluencies_removed']
            
            # Only include columns that exist in the dataframe
            existing_columns = [col for col in column_order if col in summary_df.columns]
            summary_df = summary_df[existing_columns + 
                                 [col for col in summary_df.columns if col not in column_order]]
            
            # Save the spreadsheet-friendly version
            output_file = os.path.join(output_dir, f'summary_{filename}')
            summary_df.to_csv(output_file, index=False)
            
            print(f"\nCreated summary file: {output_file}")
            print(f"Number of transcripts: {len(summary_df)}")
            print(f"Average word count: {summary_df['word_count'].mean():.0f}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print("\nSummary file creation complete!")
    print(f"Files saved in: {output_dir}")
    
    # Create overall summary
    try:
        all_years_summary = []
        for filename in os.listdir(output_dir):
            if filename.startswith('summary_infowars_'):
                year = filename.split('_')[2].split('.')[0]
                df = pd.read_csv(os.path.join(output_dir, filename))
                summary = {
                    'year': year,
                    'total_transcripts': len(df),
                    'avg_word_count': df['word_count'].mean(),
                    'avg_questions_per_transcript': df['question_count'].mean(),
                    'avg_exclamations_per_transcript': df['exclamation_count'].mean(),
                    'total_words': df['word_count'].sum()
                }
                all_years_summary.append(summary)
        
        # Save overall summary
        summary_df = pd.DataFrame(all_years_summary)
        summary_df.to_csv(os.path.join(output_dir, 'overall_yearly_summary.csv'), index=False)
        print("\nCreated overall summary file with year-by-year statistics")
        
    except Exception as e:
        print(f"Error creating overall summary: {str(e)}")

if __name__ == "__main__":
    create_spreadsheet_friendly_files()