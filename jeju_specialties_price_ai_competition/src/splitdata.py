import pandas as pd
import argparse

def load_data(args):
    train = pd.read_csv(f"{args.path}/train.csv")
    test = pd.read_csv(f"{args.path}/test.csv")

    train = train.drop(columns=["ID"])
    X_test = test.drop(columns=["ID"])
    
    X_train = train[["timestamp", "item", "corporation", "location"]]
    y_train_supply = train["supply(kg)"]
    y_train_price = train["price(원/kg)"]
    
        
    X_train.to_csv(f"{args.path}/X_train.csv", index=False)
    y_train_supply.to_csv(f"{args.path}/y_train_supply.csv", index=False)
    y_train_price.to_csv(f"{args.path}/y_train_price.csv", index=False)
    X_test.to_csv(f"{args.path}/X_test.csv", index=False)
    
    print("Head of X train :        ")
    print(X_train.head(1))
    print("---------------------------------------------")
    print("Head of y train supply : ")
    print(y_train_supply.head(1))
    print("---------------------------------------------")
    print("Head of y train price :  ")
    print(y_train_price.head(1))
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