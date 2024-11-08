import os
import pandas as pd
from abc import ABC, abstractmethod
from kaggle.api.kaggle_api_extended import KaggleApi

# Abstract Data Downloader Class
class DataDownloader(ABC):
    @abstractmethod
    def download(self, dataset_name: str, download_path: str) -> str:
        """Abstract method to download data from a source."""
        pass

# Concrete Kaggle Data Downloader Class
class KaggleDataDownloader(DataDownloader):
    def __init__(self):
        # Initialize and authenticate the Kaggle API
        self.api = KaggleApi()
        self.api.authenticate()

    def download(self, dataset_name: str, download_path: str) -> str:
        """Downloads a dataset from Kaggle and extracts it to the specified path."""
        self.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print(f"Dataset downloaded and extracted to: {download_path}")
        return download_path

# CSV Data Ingestor Class
class CSVDataIngestor:
    def ingest(self, directory_path: str) -> pd.DataFrame:
        """Reads a CSV file from a specified directory and returns it as a pandas DataFrame."""
        # List all files in the directory
        files = os.listdir(directory_path)
        
        # Find CSV files in the directory
        csv_files = [f for f in files if f.endswith(".csv")]
        
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the directory.")
        if len(csv_files) > 1:
            print("Multiple CSV files found. Using the first one.")

        # Read the first CSV file into a DataFrame
        csv_file_path = os.path.join(directory_path, csv_files[0])
        df = pd.read_csv(csv_file_path)
        
        # Return the DataFrame
        return df

# Factory for Data Downloaders
class DataDownloaderFactory:
    @staticmethod
    def get_data_downloader(source_type: str) -> DataDownloader:
        """Returns the appropriate DataDownloader based on the source type."""
        if source_type == "kaggle":
            return KaggleDataDownloader()
        else:
            raise ValueError(f"No downloader available for source type: {source_type}")

# Putting it all together
if __name__ == "__main__":
    # Define the source type, dataset name, and download path
    # source_type = "kaggle"
    # dataset_name = 'brijbhushannanda1979/bigmart-sales-data'
    # download_path = "/Users/mac/Desktop/My-Data-Science-Project/BigMart-Project/dataset"

    # # Use the factory to get the appropriate downloader
    # data_downloader = DataDownloaderFactory.get_data_downloader(source_type)

    # # Download the dataset
    # downloaded_path = data_downloader.download(dataset_name, download_path)

    # # Use the ingestor to load the unzipped CSV data into a DataFrame
    # csv_ingestor = CSVDataIngestor()
    # df = csv_ingestor.ingest(downloaded_path)
    
    # # Display the DataFrame
    # print(df.head())
    pass