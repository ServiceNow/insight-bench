import pandas as pd

def convert_csv_encoding(file_path):
    """
    Reads a CSV file with ISO-8859-1 encoding and replaces it with UTF-8 encoding.
    
    :param file_path: Path to the CSV file that needs encoding conversion.
    """
    try:
        # Load CSV with ISO-8859-1 encoding
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        
        # Save CSV with UTF-8 encoding, replacing the original file
        df.to_csv(file_path, encoding="utf-8", index=False)
        
        print(f"Successfully converted {file_path} to UTF-8 encoding.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
file_path = "data/csvs/142/data.csv"  # Change to your actual file path
convert_csv_encoding(file_path)

 #142,159,167,301,305,306,326,340