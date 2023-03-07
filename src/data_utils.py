import os
import pandas as pd
from urllib import request
from zipfile import ZipFile

def raw_data_on_disk():
    """
    Checks that the datafiles are present locally. If not, they are downloaded and unzipped.
    """
    data_dir = "../data/raw"
    data_files = [
        "credit_card_transactions-ibm_v2.csv",
        "sd254_cards.csv",
        "sd254_users.csv",
        "User0_credit_card_transactions.csv"
    ]
    # Check that directory structure is present
    if "data" not in os.listdir(".."):
        os.mkdir("../data")
    if "raw" not in os.listdir("../data"):
        os.mkdir("../data/raw")
    if "tmp" not in os.listdir(".."):
        os.mkdir("../tmp")
    # Check whether the data is already there
    if set(os.listdir(data_dir)).issuperset(data_files):
        print("Data already present. Skipping download.")
        return
    # If not, download the data archive
    print("Downloading data.")
    url = "https://drive.google.com/uc?export=download&id=1hQR9dMRUv-E0g1zYvPcOYVLk1UH_7OP8&confirm=t&uuid=8b31db11-9b77-4e9d-b104-797d165bbda0"
    zip_file_name = "../tmp/data_archive.zip"
    downloaded, _ = request.urlretrieve(url, zip_file_name)
    print(f"Downloaded {downloaded}")   
    # Unzip the archive
    print("Unzipping data files.")
    with ZipFile(zip_file_name) as zipfile:
        zipfile.extractall(data_dir)
    # Make sure everything got unzipped as expected.
    try:
        assert set(os.listdir(data_dir)).issuperset(data_files)
    except AssertionError:
        print("The expected files not present after unzipping. Leaving archive undeleted.")
        return
    # Remove the archive file
    os.remove(zip_file_name)

def read_sample_transaction() -> pd.DataFrame:
    pass

def read_users() -> pd.DataFrame:
    pass

def read_cards() -> pd.DataFrame:
    pass

def make_txdata_reader():
    pass
