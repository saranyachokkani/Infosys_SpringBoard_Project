import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def load_csv_data(file_path, data_type='heart_rate'):
    """
    Load fitness data from CSV file
    
    Args:
        file_path: Path to CSV file
        data_type: Type of data (heart_rate, steps, sleep)
    
    Returns:
        pandas DataFrame with standardized columns
    """
    
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        
        # Display first few rows to understand structure
        print("First 5 rows:")
        print(df.head())
        
        # Basic validation
        if 'timestamp' not in df.columns:
            raise ValueError("CSV must contain 'timestamp' column")
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None
def load_json_data(file_path, extract_type='heart_rate'):
    """
    Load fitness data from JSON file
    
    Args:
        file_path: Path to JSON file  
        extract_type: Type to extract (heart_rate, steps, sleep)
    
    Returns:
        pandas DataFrame with extracted data
    """
    
    try:
        # Load JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"JSON file loaded. Available data types: {data.get('data_types', [])}")
        
        # Extract specific data type
        if extract_type == 'heart_rate':
            records = data.get('heart_rate_data', [])
            df = pd.DataFrame(records)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        elif extract_type == 'steps':
            records = data.get('step_data', [])
            df = pd.DataFrame(records)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
        else:
            raise ValueError(f"Unsupported extract_type: {extract_type}")
            
        print(f"Extracted {len(df)} {extract_type} records")
        return df
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        return None

class FitnessDataIngester:
    """Complete fitness data ingestion pipeline"""
    
    def _init_(self):
        self.supported_formats = ['.csv', '.json']
        self.supported_data_types = ['heart_rate', 'steps', 'sleep']
        
    def detect_file_format(self, file_path):
        """Detect file format from extension"""
        import os
        _, ext = os.path.splitext(file_path.lower())
        return ext
    
    def validate_data_type(self, data_type):
        """Validate requested data type"""
        if data_type not in self.supported_data_types:
            raise ValueError(f"Unsupported data type. Use: {self.supported_data_types}")
    
    def load_data(self, file_path, data_type='heart_rate'):
        """
        Universal data loader - handles both CSV and JSON
        
        Args:
            file_path: Path to data file
            data_type: Type of data to extract
            
        Returns:
            pandas DataFrame with standardized format
        """
        
        print(f"Loading {data_type} data from {file_path}")
        
        # Validate inputs
        self.validate_data_type(data_type)
        file_format = self.detect_file_format(file_path)
        
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Load based on format
        if file_format == '.csv':
            df = self._load_csv(file_path, data_type)
        elif file_format == '.json':
            df = self._load_json(file_path, data_type)
            
        if df is not None and not df.empty:
            df = self._standardize_dataframe(df, data_type)
            
        return df
    
    def _load_csv(self, file_path, data_type):
        """Load CSV data (using our previous function)"""
        return load_csv_data(file_path, data_type)
    
    def _load_json(self, file_path, data_type):
        """Load JSON data (using our previous function)"""
        return load_json_data(file_path, data_type)
    
    def _standardize_dataframe(self, df, data_type):
        """Standardize column names and data types"""
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Standardize column names based on data type
        if data_type == 'heart_rate':
            # Map common heart rate column names
            hr_columns = ['bpm', 'heart_rate_bpm', 'heartrate', 'hr']
            for col in hr_columns:
                if col in df.columns:
                    df['heart_rate'] = df[col]
                    break
                    
        elif data_type == 'steps':
            # Map common step column names  
            step_columns = ['steps', 'step_count', 'steps_per_minute']
            for col in step_columns:
                if col in df.columns:
                    df['steps'] = df[col]
                    break
        
        print(f"Standardized DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    
    

# Test the complete pipeline
ingester = FitnessDataIngester()

csv_data = ingester.load_data('sample_heart_rate.csv', 'heart_rate')
print("\nCSV Data loaded:")
print(csv_data.head())

# Test with JSON
json_hr_data = ingester.load_data('sample_fitness_data.json', 'heart_rate') 
print("\nJSON Data loaded:")
print(json_hr_data.head())

def perform_data_quality_check(df, data_type):
    """
    Comprehensive data quality assessment
    
    Args:
        df: pandas DataFrame
        data_type: Type of data being checked
        
    Returns:
        dict with quality metrics
    """
    
    quality_report = {
        'total_records': len(df),
        'date_range': None,
        'missing_values': {},
        'data_issues': [],
        'quality_score': 0
    }
    
    # Basic checks
    if df.empty:
        quality_report['data_issues'].append("No data found")
        return quality_report
    
    # Timestamp analysis
    if 'timestamp' in df.columns:
        quality_report['date_range'] = {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max(),
            'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        }
        
        # Check for timestamp gaps
        time_diffs = df['timestamp'].diff().dropna()
        median_interval = time_diffs.median().total_seconds()
        large_gaps = time_diffs[time_diffs > timedelta(minutes=10)]
        
        if len(large_gaps) > 0:
            quality_report['data_issues'].append(f"Found {len(large_gaps)} time gaps > 10 minutes")
    
    # Missing value analysis
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            quality_report['missing_values'][column] = {
                'count': missing_count,
                'percentage': (missing_count / len(df)) * 100
            }
    
    # Data type specific checks
    if data_type == 'heart_rate':
        hr_col = 'heart_rate' if 'heart_rate' in df.columns else 'bpm'
        if hr_col in df.columns:
            hr_values = df[hr_col].dropna()
            if len(hr_values) > 0:
                # Check for unrealistic values
                unrealistic_low = (hr_values < 30).sum()
                unrealistic_high = (hr_values > 220).sum()
                
                if unrealistic_low > 0:
                    quality_report['data_issues'].append(f"{unrealistic_low} heart rate values < 30 BPM")
                if unrealistic_high > 0:
                    quality_report['data_issues'].append(f"{unrealistic_high} heart rate values > 220 BPM")
    
    # Calculate quality score (0-100)
    score = 100
    score -= len(quality_report['data_issues']) * 10  # -10 for each issue
    score -= sum([info['percentage'] for info in quality_report['missing_values'].values()]) / 2  # Penalize missing data
    quality_report['quality_score'] = max(0, score)
    
    return quality_report

# Test data quality check
quality = perform_data_quality_check(json_hr_data, 'heart_rate')
print("\nData Quality Report:")
print(f"Total records: {quality['total_records']}")
print(f"Quality score: {quality['quality_score']:.1f}/100")
print(f"Date range: {quality['date_range']['start']} to {quality['date_range']['end']}")
if quality['data_issues']:
    print("Issues found:")
    for issue in quality['data_issues']:
        print(f"  - {issue}")
