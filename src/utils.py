import os
import pandas as pd
import numpy as np
from urllib import request
from zipfile import ZipFile
from typing import Tuple, Union, List
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

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


def clean_split_tx(users_cards_df: pd.DataFrame, test_ids) -> Tuple[str, str, float]:
    """
    Processes raw transaction data and splits into test set (comprised of users in `test_ids`) and full training set (the rest). Returns: the file names for the positive and negative training data, and the rate of positive fraud cases in the training data.
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
        # read in the next training data
        df = next(reader)
        # clean the data
        df = clean_tx_df(df)
        # merge in users and cards features
        df = df.merge(users_cards_df, how='left', on=['user', 'card'])
        # construct needed features
        df = user_features(df)

        # filter and save data from users in the test set
        test_df = df[df.user.isin(test_ids)]
        test_df.to_csv(test_file, **writing_args)

        # split the training pool into fraud and non-fraud observations
        train_df = df[~df.user.isin(test_ids)]
        pos_count += train_df.is_fraud.sum()
        n += train_df.is_fraud.count()
        train_df[train_df.is_fraud].to_csv(save_file_pos, **writing_args)
        train_df[~train_df.is_fraud].to_csv(save_file_neg, **writing_args)
        
        # after first iteration mode is append and header should not be written
        writing_args = {'mode': 'a', 'header':False}

    return save_file_pos, save_file_neg, pos_count / n


def make_training_sets(subset_ids, pos_filename, neg_filename, rate) -> Tuple[str, str, str]:
    """
    Constructs and saves two training data sets from the full training data recorded in `pos_filename` and `neg_filename`, returning 3 file name:
    - An unbalanced set containing users in `subset_ids`
    - A balanced set combining all positive cases and negative cases selected at `rate`
    - The fraud rates by MCC code
    """
    rng = np.random.default_rng(22)
    subset_name = prepend_dir('tx_train_set.csv')
    balanced_name = prepend_dir('tx_train_balanced.csv')
    mcc_rates_name = prepend_dir('mcc_rates.csv')

    balanced_df = pd.read_csv(pos_filename, index_col=0)
    subset_df = balanced_df[balanced_df.user.isin(subset_ids)].copy()

    mcc_dict = {}
    update_mcc(balanced_df, mcc_dict)

    reader = pd.read_csv(neg_filename, index_col=0, chunksize=100_000)
    for df in reader:
        n_records = len(df.index)
        # choose negative cases to include based on given fraud rate
        idx_for_balance = rng.choice([False, True], n_records, replace=True, p=[1-rate, rate])
        balanced_df = pd.concat([balanced_df, df.loc[idx_for_balance]])
        
        update_mcc(df, mcc_dict)

        subset_df = pd.concat([subset_df, df[df.user.isin(subset_ids)]])
    
    subset_df.sort_values(['user', 'card'], inplace=True)
    subset_df.to_csv(subset_name)

    balanced_df.sort_values(['user', 'card'], inplace=True)
    balanced_df.to_csv(balanced_name)

    mcc_rates = pd.DataFrame.from_dict(mcc_dict, orient='index')
    # print(mcc_rates.head())
    mcc_rates['mcc_fraud_rate'] = mcc_rates['pos'] / mcc_rates['n']
    mcc_rates[['mcc_fraud_rate']].to_csv(mcc_rates_name)

    return subset_name, balanced_name, mcc_rates_name
    

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


def convert_multicat(df: pd.DataFrame, colname: str, copy:bool=True) -> Tuple[pd.DataFrame, List[str]]:
    '''(Optionally copies `df` and) converts the categorical column `colname` into dummies. Allows for membership in multiple categories separated by a single comma, e.g. entry "a,b" will be converted into `True` for columns `a` and `b`'''
    if copy:
        dummy_df = df.copy()
    else:
        dummy_df = df
    cats = set()
    for entry in dummy_df[colname].dropna().unique().tolist():
        for cat in entry.split(','):
            cats.add(cat)
    cats = list(cats)
    for cat in cats:
        dummy_df[cat] = dummy_df[colname].str.contains(cat).fillna(False)
    dummy_df.drop(columns=colname, inplace=True)

    return dummy_df, cats


def update_mcc(new_data, mcc_dict):
    fraud_totals = (
        new_data[['mcc', 'is_fraud']]
            .groupby('mcc')
            .agg(['sum', 'count'])
            ['is_fraud']
    )
    # fraud_totals = fraud_totals['is_fraud']
    # print(fraud_totals)
    for mcc, data in fraud_totals.iterrows():
        # print(data)
        if mcc in mcc_dict:
            mcc_dict[mcc]['pos'] += data['sum']
            mcc_dict[mcc]['n'] += data['count']
        else:
            mcc_dict[mcc] = {
                'pos': data['sum'],
                'n': data['count']
            }


class MakeDummies(BaseEstimator, TransformerMixin):
    """Transforms categorical columns into dummies. Can handle multi-category columns"""
    def __init__(self, multicat_col: str, drop_first=True, dummy_cols:Union[List[str], str]='auto') -> None:
        super().__init__()
        self.multicat_col = multicat_col
        self.drop_first = drop_first
        self.dummy_cols= dummy_cols
    
    def fit(self, X, y=None):
        _, self.cats = convert_multicat(X, self.multicat_col)
        return self
    
    def fit_transform(self, X, y=None):
        dummy_df, self.cats = convert_multicat(X, self.multicat_col)
        dummy_df = pd.get_dummies(dummy_df, drop_first=self.drop_first)
        return dummy_df
    
    def transform(self, X, y=None):
        dummy_df = X.copy()
        for cat in self.cats:
            dummy_df[cat] = dummy_df[self.multicat_col].str.contains(cat).fillna(False)
        dummy_df.drop(columns=self.multicat_col, inplace=True)
        dummy_df = pd.get_dummies(dummy_df, drop_first=self.drop_first)

        return dummy_df


class MCCRates(BaseEstimator, TransformerMixin):
    """Transforms MCC codes into average fraud rates, based on saved data or a specific training set. Note: only accepts data frames"""
    
    mcc_rates_file = prepend_dir('mcc_rates.csv')
    
    def __init__(self, use_saved=True) -> None:
        super().__init__()
        self.use_saved = use_saved
        self.mcc_rates = None

    def fit(self, X, y=None):
        if self.use_saved:
            if self.mcc_rates is None:
                self.mcc_rates = pd.read_csv(self.mcc_rates_file, index_col=0)
        else:
            self.mcc_rates = (
                X.copy()
                .groupby('mcc')
                .agg('mean')
                .rename({'mean':'mcc_fraud_rate'}, axis=1)
            )
        return self
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        X_transformed = X.merge(self.mcc_rates, how='left', left_on='mcc', right_index=True)
        X_transformed.drop(columns='mcc', inplace=True)
        return X_transformed


def merge_mcc_rates(df: pd.DataFrame) -> pd.DataFrame:
    mcc_fraud_rates = pd.read_csv(prepend_dir('mcc_rates.csv'), index_col=0)
    merged_df = df.merge(mcc_fraud_rates, how='left', left_on='mcc', right_index=True)
    merged_df.drop(columns='mcc', inplace=True)
    return merged_df


def get_MCC_codes() -> pd.DataFrame:
    # as plaintext csv: https://github.com/greggles/mcc-codes/blob/main/mcc_codes.csv
    # or a python package with some more data: https://pypi.org/project/iso18245/
    ...


##########
# MODELS #
##########

class NaiveClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.rng = np.random.default_rng()
        return self

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype('int')
    
    def predict_proba(self, X):
        proba = self.rng.random(len(X))
        return np.array([proba, 1-proba]).T
    
    def fit_predict(self, X, y):
        _ = self.fit(X, y)
        return self.predict(X)
