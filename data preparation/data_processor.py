import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

def create_output_directory():
    """Create the processed_data directory if it doesn't exist"""
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def process_csv_file(csv_path, output_dir):
    """Process a CSV file with error handling and data validation"""
    try:
        # Try different encodings if the default fails
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding, on_bad_lines='warn')
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            logging.error(f"Could not read {csv_path} with any of the attempted encodings")
            return
        
        # Basic data cleaning
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        
        # Save processed data
        output_filename = f"{csv_path.stem}_processed.csv"
        output_path = output_dir / output_filename
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Generate summary statistics
        summary = {
            'file_name': csv_path.name,
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'column_types': df.dtypes.to_dict(),
            'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logging.info(f"Successfully processed {csv_path.name}")
        return summary
        
    except Exception as e:
        logging.error(f"Error processing {csv_path}: {str(e)}")
        return None

def process_excel_file(excel_path, output_dir):
    """Process an Excel file with error handling and data validation"""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(excel_path)
        summaries = []
        
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                # Basic data cleaning
                df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                df = df.replace(r'^\s*$', pd.NA, regex=True)
                
                # Save processed data
                output_filename = f"{excel_path.stem}_{sheet_name}_processed.csv"
                output_path = output_dir / output_filename
                df.to_csv(output_path, index=False, encoding='utf-8')
                
                # Generate summary statistics
                summary = {
                    'file_name': excel_path.name,
                    'sheet_name': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'missing_values': df.isnull().sum().sum(),
                    'column_types': df.dtypes.to_dict(),
                    'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                summaries.append(summary)
                
                logging.info(f"Successfully processed sheet {sheet_name} from {excel_path.name}")
                
            except Exception as e:
                logging.error(f"Error processing sheet {sheet_name} in {excel_path}: {str(e)}")
                continue
        
        return summaries
        
    except Exception as e:
        logging.error(f"Error processing {excel_path}: {str(e)}")
        return None

def process_files():
    """Process all Excel and CSV files in the data directory"""
    output_dir = create_output_directory()
    data_dir = Path("data")
    
    # Create a summary file
    summary_file = output_dir / "processing_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Data Processing Summary\n")
        f.write("=====================\n\n")
        
        # Process CSV files
        csv_files = list(data_dir.glob("*.csv"))
        f.write(f"CSV Files Processed: {len(csv_files)}\n")
        for csv_path in csv_files:
            summary = process_csv_file(csv_path, output_dir)
            if summary:
                f.write(f"\nFile: {summary['file_name']}\n")
                f.write(f"Rows: {summary['rows']}\n")
                f.write(f"Columns: {summary['columns']}\n")
                f.write(f"Missing Values: {summary['missing_values']}\n")
                f.write(f"Processing Time: {summary['processing_time']}\n")
        
        # Process Excel files
        excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
        f.write(f"\nExcel Files Processed: {len(excel_files)}\n")
        for excel_path in excel_files:
            summaries = process_excel_file(excel_path, output_dir)
            if summaries:
                for summary in summaries:
                    f.write(f"\nFile: {summary['file_name']}\n")
                    f.write(f"Sheet: {summary['sheet_name']}\n")
                    f.write(f"Rows: {summary['rows']}\n")
                    f.write(f"Columns: {summary['columns']}\n")
                    f.write(f"Missing Values: {summary['missing_values']}\n")
                    f.write(f"Processing Time: {summary['processing_time']}\n")

if __name__ == "__main__":
    logging.info("Starting data processing...")
    process_files()
    logging.info("Data processing completed.") 