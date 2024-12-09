import pandas as pd
import os
from tqdm import tqdm

def split_data_by_year():
    print("Starting to read the large CSV file...")
    
    # Create output directory if it doesn't exist
    output_dir = 'yearly_transcripts'
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and process the CSV in chunks
    chunk_size = 1000  # Adjust if needed
    chunks = pd.read_csv('cleaned_infowars_transcripts.csv', 
                        chunksize=chunk_size, 
                        parse_dates=['date'])
    
    # Dictionary to store DataFrames for each year
    yearly_data = {}
    
    print("Processing chunks and splitting by year...")
    # Process each chunk
    for chunk in tqdm(chunks):
        # Group the chunk by year
        for year, group in chunk.groupby(chunk['date'].dt.year):
            if year in yearly_data:
                yearly_data[year] = pd.concat([yearly_data[year], group])
            else:
                yearly_data[year] = group
    
    print("\nSaving yearly files...")
    # Save each year's data to a separate file
    for year in tqdm(yearly_data.keys()):
        output_file = os.path.join(output_dir, f'infowars_{year}.csv')
        yearly_data[year].to_csv(output_file, index=False)
        print(f"Created {output_file} with {len(yearly_data[year])} transcripts")
    
    # Create a summary file
    summary = {
        year: {
            'number_of_transcripts': len(data),
            'total_words': data['cleaned_length'].sum(),
            'average_transcript_length': data['cleaned_length'].mean(),
            'date_range': f"{data['date'].min()} to {data['date'].max()}"
        }
        for year, data in yearly_data.items()
    }
    
    # Save summary as CSV
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.to_csv(os.path.join(output_dir, 'yearly_summary.csv'))
    
    print("\nYearly split complete!")
    print(f"Files saved in: {output_dir}")
    print("\nSummary of files created:")
    for year in sorted(yearly_data.keys()):
        print(f"{year}: {len(yearly_data[year])} transcripts")

if __name__ == "__main__":
    split_data_by_year()