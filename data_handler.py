import pandas as pd
from pathlib import Path


class DataLoader:
    """Simple class for loading CSV and XLSX files."""

    def load_data(self, file_path):
        """
        Load data from CSV or XLSX file and drop rows with missing data.

        Args:
            file_path: Path to the CSV or XLSX file

        Returns:
            pd.DataFrame: Loaded dataframe with missing rows removed
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # # If 'median_house_value' is a column, drop rows where it is missing
        # if 'median_house_value' in df.columns:
        #     df = df.dropna(subset=['median_house_value'])
        # else:
        #     # For other datasets, drop all rows with any NA to maintain original behavior
        #     df = df.dropna()

        return df