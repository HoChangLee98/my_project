import argparse
import time
from tqdm.auto import tqdm

from utils import *
from feature_engineering import *
from model import *

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("--submission", "-s", default="../submission", type=str, help="path of submission folder")
parser.add_argument("--model", "-m", default="lgboost", type=str, help="select model")

args = parser.parse_args()


def main(args):

    start_time = time.time()

    submission = pd.read_csv(f"{args.submission}/sample_submission.csv")
    
    for target in tqdm(["HLM", "MLM"]):
        X_test = dataloader(target_name=target)[4]
        fe = FeatureEngineer()
        X_test = fe.drop_cat_feature(X_test)
    
        model = load_pickle(f"{target}_{args.model}")

        y_pred = model.predict(X_test)
        submission[target] = y_pred
        print(f"====== Prediction of {target} : \n", y_pred)
        print("================")

    submission.to_csv(f"{args.submission}/{args.model}.csv", index=False)
    
    end_time = time.time()

    print("===== Time : ", round(end_time - start_time, 4))


if __name__ == "__main__":
    main(args)

        