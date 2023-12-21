import pandas as pd
from pathlib import Path

def main():
    data_dir = Path('.') / 'data' / 'processed'
    train_all = pd.concat([
        pd.read_csv(data_dir / "tx_train_pos.csv", index_col=0),
        pd.read_csv(data_dir / "tx_train_neg.csv", index_col=0)
    ])
    train_all.sort_values(by=["user", "timestamp"], inplace=True)
    train_all.drop(columns=['timestamp'], inplace=True)

    train_all.to_csv(data_dir / "tx_train_all.csv")

if __name__ == '__main__':
    main()
