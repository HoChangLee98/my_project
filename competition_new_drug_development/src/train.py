import argparse

from utils import *
from feature_engineering import *
from model import *


parser = argparse.ArgumentParser()
parser.add_argument("--target_name", "-t", type=str, help="train model about target name")
parser.add_argument("--model_name", "-m", type=str, default="lgboost" ,help="input model name")
parser.add_argument("--opt_tune", "-o", type=bool, default=False, help="find hyper-parameters")
parser.add_argument("--seed_num", "-s", type=int, default=0, help="set seed number")
args = parser.parse_args()

def main(args):
    
    seed_everything(args.seed_num)

    if args.opt_tune == True:
        ## load data
        X_train, X_valid, y_train, y_valid = dataloader(target_name=args.target_name)[0:4]

        ## feature engineering
        fe = FeatureEngineer(X_train=X_train, X_test=X_valid)
        X_train = fe.drop_cat_feature(X_train)
        X_valid = fe.drop_cat_feature(X_valid)

 
        print(f"=====Start {args.target_name} Optuna=====")

        params = get_optuna_params(model_name=args.model_name, X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
        save_pickle(params, f"{args.target_name}_{args.model_name}_base_params")
        
        print(f"=====End {args.target_name} Oputna=====")
   

    elif args.opt_tune == False:
        
        print(f"=====Start {args.target_name} Training=====")
        ## load data
        X_train, X_valid, y_train, y_valid = dataloader(target_name=args.target_name)[0:4]

        ## feature engineering
        fe = FeatureEngineer(X_train=X_train, X_test=X_valid)
        X_train = fe.drop_cat_feature(X_train)
        X_valid = fe.drop_cat_feature(X_valid)

        ## modeling
        base_params = load_pickle(f"{args.target_name}_{args.model_name}_base_params")
        model = Models(model_name=args.model_name, params = base_params)
        fitted_model = model.fit(X_train, y_train)
        y_val_pred = fitted_model.predict(X_valid)
        rmse = mean_squared_error(y_valid, y_val_pred, squared=False)

        save_pickle(fitted_model, f"{args.target_name}_{args.model_name}")

        print(f"rmse of {args.target_name} : ", rmse)
        print(f"=====End {args.target_name} Training=====")

        return rmse
    

if __name__ == "__main__":
    main(args)

