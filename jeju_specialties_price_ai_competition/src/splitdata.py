import pandas as pd
import argparse

def load_data(args):
    train = pd.read_csv(f"{args.path}/train.csv")
    test = pd.read_csv(f"{args.path}/test.csv")

    train = train.drop(columns=["ID"])
    X_test = test.drop(columns=["ID"])
    
    X_train = train[["timestamp", "supply(kg)", "item", "corporation", "location"]]
    y_train = train[["price(원/kg)", "item", "corporation", "location"]]
    
    X_train.to_csv(f"{args.path}/X_train.csv", index=False)
    y_train.to_csv(f"{args.path}/y_train.csv", index=False)
    X_test.to_csv(f"{args.path}/X_test.csv", index=False)
    
    print("Head of X train :        ")
    print(X_train.head(1))
    print("---------------------------------------------")
    print("Head of y train : ")
    print(y_train.head(1))
    print("---------------------------------------------")
    print("Head of X test :         ")
    print(X_test.head(1))
    print("---------------------------------------------")
    print("Done Save !")


parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", type=str, default="../data", help="path of dataset")
args = parser.parse_args()

if __name__ == "__main__":
    load_data(args)