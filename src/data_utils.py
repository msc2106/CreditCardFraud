import os
import pandas as pd
from urllib import request
from zipfile import ZipFile

# file locations for general use
data_dir = "../data/raw"
data_files = {
    "full_tx": "credit_card_transactions-ibm_v2.csv",
    "cards": "sd254_cards.csv",
    "users": "sd254_users.csv",
    "sample_tx": "User0_credit_card_transactions.csv"
}

def confirm_dirs(path):
    dirs = path.split('/')
    if dirs[0] == '..':
        dirs = dirs[1:]
    base_dir = ".."
    for dirname in dirs:
        if dirname not in os.listdir(base_dir):
            os.mkdir(base_dir+'/'+dirname)
        base_dir += '/' + dirname

def data_files_present() -> bool:
    return set(os.listdir(data_dir)).issuperset(data_files.values())

def raw_data_on_disk():
    """
    Checks that the datafiles are present locally. If not, they are downloaded and unzipped.
    """
    # if "data" not in os.listdir(".."):
    #     os.mkdir("../data")
    # if "raw" not in os.listdir("../data"):
    #     os.mkdir("../data/raw")
    # if "tmp" not in os.listdir(".."):
    #     os.mkdir("../tmp")
    confirm_dirs(data_dir)
    # Check whether the data is already there
    if data_files_present():
        print("Data already present. Skipping download.")
        return
    # If not, download the data archive
    print("Downloading data.")
    confirm_dirs("tmp")
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
        assert data_files_present()
    except AssertionError:
        print("The expected files not present after unzipping. Leaving archive undeleted.")
        return
    # Remove the archive file
    os.remove(zip_file_name)

def read_sample_transactions() -> pd.DataFrame:
    """
    Loads the transactions for User 0 into a data frame.
    """
    return pd.read_csv(data_dir + '/' + data_files['sample_tx'])

def read_users() -> pd.DataFrame:
    pass

def read_cards() -> pd.DataFrame:
    pass

def make_txdata_reader(**kwargs):
    """
    Returns an iterator to read chunks of the main transaction data file. The default chunk size is 10,000, but alternative values for `chunksize` or for any other parameter of `pd.read_csv` can be pased as named arguments.
    """
    params = {
        "chunksize": 10000
    }
    params.update(kwargs)
    return pd.read_csv(data_dir+'/'+data_files["full_tx"], **params)

def save_data(dfdict):
    """
    Saves data to `../data/processed`. The keys of the `datafiles` dict are taken to be the filenames, and the values are assumed to be data frames
    """
    dirname = "../data/processed"
    confirm_dirs(dirname)
    for filename, df in dfdict.items():
        df.to_csv(dirname+'/'+filename)
