import pandas as pd
import argparse

from sklearn.model_selection import train_test_split

from utils import seed_everything

def raw_data_split(args):
    
    seed_everything(seed=args.seed)
    
    train = pd.read_csv(f"{args.path}/train.csv")
    test = pd.read_csv(f"{args.path}/test.csv")

    train = train.drop(columns='SAMPLE_ID') 
    X_train = train.drop(columns="CI_HOUR")
    y_train = train["CI_HOUR"]
    X_test = test.drop(columns='SAMPLE_ID') 

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.valid_size)

    X_train.to_csv(f"{args.path}/X_train.csv", index=False)
    X_valid.to_csv(f"{args.path}/X_valid.csv", index=False)
    y_train.to_csv(f"{args.path}/y_train.csv", index=False)
    y_valid.to_csv(f"{args.path}/y_valid.csv", index=False)
    
    X_test.to_csv(f"{args.path}/X_test.csv", index=False)

    print("Complete !")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", default="../data", type=str, help="path to load and save data")
    parser.add_argument("--valid_size", "-v", default=0.3, type=float, help="size of validation set")
    parser.add_argument("--seed", "-s", default=0, type=int, help="setting seed number")
    args = parser.parse_args()

    raw_data_split(args=args)
