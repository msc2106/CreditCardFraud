import os
import pandas as pd
from urllib import request
from zipfile import ZipFile
from typing import Tuple, Set

# file locations for general use
data_dir = "../data/raw"
data_files = {
    "full_tx": "credit_card_transactions-ibm_v2.csv",
    "cards": "sd254_cards.csv",
    "users": "sd254_users.csv",
    "sample_tx": "User0_credit_card_transactions.csv"
}

###########################
# LOADING AND SAVING DATA #
###########################

def prepend_dir(filename):
    """
    Prepends `../data/processed/` to the filename
    """
    return '../data/processed/' + filename

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
    """
    Loads the users table into a data frame.
    """
    return pd.read_csv(data_dir+'/'+data_files['users'])

def read_cards() -> pd.DataFrame:
    """
    Loads the cards table into a data frame.
    """
    return pd.read_csv(data_dir+'/'+data_files['cards'])

def make_txdata_reader(**kwargs):
    """
    Returns an iterator to read chunks of the main transaction data file. The default chunk size is 10,000, but alternative values for `chunksize` or for any other parameter of `pd.read_csv` can be pased as named arguments.
    """
    params = {
        "chunksize": 100_000
    }
    params.update(kwargs)
    return pd.read_csv(data_dir+'/'+data_files["full_tx"], **params)


def save_data(dfdict, **kwargs):
    """
    Saves data to `../data/processed`. The keys of the `datafiles` dict are taken to be the filenames, and the values are assumed to be data frames
    """
    dirname = "../data/processed"
    confirm_dirs(dirname)
    for filename, df in dfdict.items():
        df.to_csv(dirname+'/'+filename, **kwargs)


def make_training_sets(subset_ids, pos_filename, neg_filename, rate):
    """
    Constructs and saves two training data sets from the full training data recorded in `pos_filename` and `neg_filename`:
    - An unbalanced set containing users in `subset_ids`
    - A balanced set combining all positive cases and negative cases selected at `rate`
    """
    subset_name = prepend_dir('tx_train_set.csv')
    balanced_name = prepend_dir('tx_train_balanced.csv')
    balanced_df = pd.read_csv(pos_filename, index_col=0)
    subset_df = balanced_df[balanced_df.user.isn(subset_ids)].copy()
    ...
    # TODO
  

def clean_split_tx(users_cards_df: pd.DataFrame, test_ids):
    """
    Processes raw transaction data and splits into test set (comprised of users in `test_ids`) and full training set (the rest).
    """
    raw_data_on_disk()
    save_file_pos = prepend_dir('tx_train_pos.csv')
    save_file_neg = prepend_dir('tx_train_neg.csv')
    test_file = prepend_dir('tx_test.csv')
    pos_count = 0
    n = 0
    reader = make_txdata_reader()
    # in first iteration, overwrite file and write header
    writing_args = {'mode':'w', 'header':True}
    for df in reader:
        df = next(reader)
        df = clean_tx_df(df)
        df = df.merge(users_cards_df, how='left', on=['user', 'card'])
        df = user_features(df)

        test_df = df[df.user.isin(test_ids)]
        test_df.to_csv(test_file, **writing_args)

        train_df = df[~df.user.isin(test_ids)]
        pos_count += train_df.is_fraud.sum()
        n += train_df.is_fraud.count()
        train_df[train_df.is_fraud].to_csv(save_file_pos, **writing_args)
        train_df[~train_df.is_fraud].to_csv(save_file_neg, **writing_args)
        
        # after first iteration mode is append and header should not be written
        writing_args = {'mode': 'a', 'header':False}
    metadata = {
        "pos_filename": save_file_pos,
        "neg_filename": save_file_neg,
        "rate": pos_count / n
    }
    return metadata


#################
# DATA CLEANING #
#################

def update_colnames(old_colnames):
    return (
        old_colnames
        .str.lower()
        .str.replace(' - ', '_', regex=False)
        .str.replace(' ', '_', regex=False)
        .str.replace('?', '', regex=False)
    )


def mm_yyyy_to_dt(mm_yyyy: str) -> pd.Timestamp:
    """
    Takes a date string in the format `mm/yyyy` and returns a Pandas Timestamp object for 1/m/y
    """
    m, y = map(int, mm_yyyy.split('/'))
    return pd.Timestamp(
        year=y,
        month=m,
        day=1
    )


def convert_monthyear_dates(column: pd.Series):
    return column.apply(mm_yyyy_to_dt)


def convert_dollar_amounts(column: pd.Series) -> pd.Series:
    if column.dtype in ['float', 'int']:
        return column
    else:
        return column.str.slice(1).astype('float')


def make_timestamps(df: pd.DataFrame):
    datetime_df = df[['year', 'month', 'day']].copy()
    if len(datetime_df) > 0:
        datetime_df[['hour', 'minute']] = df.time.str.split(':', expand=True)
        datetime_df['hour'] = datetime_df.hour.apply(int)
        datetime_df['minute'] = datetime_df.minute.apply(int)
    return pd.to_datetime(datetime_df)


def clean_tx_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Carries out the operations to clean the transactions data frame:
    - change column names
    - convert `amount` to float
    - make a timestamp column
    - make `tx_type` categorical feature
    - convert `is_fraud` to bool
    """
    # Updating the column names
    df.columns = update_colnames(df.columns)
    # if df.shape[0] == 0:
    #     return df
    # Converting transactions to float
    df.amount = df.amount.str.slice(1)\
        .astype('float')
    # Making a timestamp column
    df['timestamp'] = make_timestamps(df)
    # Creating the tx_type categorical feature
    df['tx_type'] = df.use_chip\
        .str.strip(" Transaction")\
        .str.lower()\
        .astype("category")
    df.drop(['year', 'month', 'day', 'time', 'merchant_name', 'use_chip'], axis=1, inplace=True)
    # Converting the is_fraud to bool
    df.is_fraud = df.is_fraud == 'Yes'
    return df


def user_features(df: pd.DataFrame) -> pd.DataFrame:
    #location-based features
    merged_df = df.copy()
    merged_df['home_city'] = (merged_df.city == merged_df.merchant_city) & (merged_df.state == merged_df.merchant_state)
    merged_df['home_state'] = merged_df.state == merged_df.merchant_state
    merged_df['home_zip'] = merged_df.zip == merged_df.zipcode
    merged_df['overseas'] = (merged_df.tx_type != 'online') & merged_df.zip.isna()
    merged_df = merged_df[~(merged_df.merchant_state.isna() & (merged_df.tx_type != 'online'))]

    # time-based features
    merged_df['user_age'] = (merged_df.timestamp - merged_df.birthdate).dt.days/365
    merged_df['retired'] = merged_df.user_age >= merged_df.retirement_age
    merged_df['until_expired'] = (merged_df.expires - merged_df.timestamp).dt.days
    merged_df['since_opened'] = (merged_df.timestamp - merged_df.acct_open_date).dt.days

    # dropping unneeded columns
    redundant_cols = ['merchant_city', 'merchant_state', 'zip', 'city', 'state', 'zipcode', 'birthdate', 'retirement_age', 'expires', 'acct_open_date', 'timestamp']
    merged_df.drop(columns=redundant_cols, inplace=True)

    return merged_df


#TODO make into a scikitlearn transformer class
def convert_multicat(df: pd.DataFrame, colname: str) -> Tuple[pd.DataFrame, list]:
    '''Copies `df`  and converts the categorical column `colname` has been converted into dummies. Allows for membership in multiple categories separated by a single comma, e.g. entry "a,b" will be converted into `True` for columns `a` and `b`'''
    dummy_df = df.copy()
    cats = set()
    for entry in dummy_df[colname].dropna().unique().tolist():
        for cat in entry.split(','):
            cats.add(cat)
    cats = list(cats)
    for cat in cats:
        dummy_df[colname] = dummy_df[colname].str.contains(cat).fillna(False)
    dummy_df.drop(columns=colname, inplace=True)
    return dummy_df, cats



def get_MCC_codes() -> pd.DataFrame:
    # as plaintext csv: https://github.com/greggles/mcc-codes/blob/main/mcc_codes.csv
    # or a python package with some more data: https://pypi.org/project/iso18245/
    ...

