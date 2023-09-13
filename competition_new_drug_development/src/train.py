import argparse
import warnings
warnings.filterwarnings('ignore')

from utils import *
from feature_engineering import *
from model import *


parser = argparse.ArgumentParser()
parser.add_argument("--version", "-v", type=str, default="test", help="set version")
parser.add_argument("--target_name", "-t", type=str, default="MLM", help="train model about target name")
parser.add_argument("--model_name", "-m", type=str, default="lgboost" ,help="input model name")
parser.add_argument("--opt_tune", "-o", type=bool, default=False, help="find hyper-parameters")
parser.add_argument("--seed_num", "-s", type=int, default=0, help="set seed number")
parser.add_argument("--minmax_columns", "-c", type=list, default=['AlogP', 'Molecular_Weight', 'Num_RotatableBonds', 'LogD', 'Molecular_PolarSurfaceArea'], help="input minmaxscaling columns")
args = parser.parse_args()

def main(args):
    
    seed_everything(args.seed_num)

    if args.opt_tune == True:
        ## load data
        X_train, X_valid, y_train, y_valid = dataloader(target_name=args.target_name)[0:4]
        
        ## del outlier
        X_train, y_train = outlier_remove_only_train(X_train=X_train, y_train=y_train,
                                                     columns = ["Num_H_Donors", "Num_RotatableBonds"])

        ## feature engineering
        fe = FeatureEngineer()
        minmaxscaler = fe.fit_minmaxscaler(df=X_train, columns=args.minmax_columns)
        labelencoder = fe.fit_labelencoder(df=X_train)

        X_train = fe.feature_engineering_process(
            df=X_train,
            minmaxscaler=minmaxscaler, 
            labelencoder=labelencoder, 
            minmax_columns=args.minmax_columns
            )
        
        X_valid = fe.feature_engineering_process(
            df=X_valid,
            minmaxscaler=minmaxscaler, 
            labelencoder=labelencoder, 
            minmax_columns=args.minmax_columns
            )
 
        print(f"=====Start {args.target_name}_{args.version} Optuna=====")

        params = get_optuna_params(model_name=args.model_name, X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
        save_pickle(params, f"{args.target_name}_{args.model_name}_params_{args.version}")
        
        print(f"=====End {args.target_name}_{args.version} Oputna=====")
   

    elif args.opt_tune == False:
        
        print(f"=====Start {args.target_name}_{args.version} Training=====")
        ## load data
        X_train, X_valid, y_train, y_valid = dataloader(target_name=args.target_name)[0:4]

        ## del outlier
        X_train, y_train = outlier_remove_only_train(X_train=X_train, y_train=y_train,
                                                     columns = ["Num_H_Donors", "Num_RotatableBonds"])

        ## feature engineering
        fe = FeatureEngineer()
        minmaxscaler = fe.fit_minmaxscaler(df=X_train, columns=args.minmax_columns)
        labelencoder = fe.fit_labelencoder(df=X_train)
        
        fe_object = {"minmax" : minmaxscaler, 
                     "label" : labelencoder}
        
        save_pickle(fe_object, file_name=f"{args.target_name}_fe_{args.version}")
        
        X_train = fe.feature_engineering_process(
            df = X_train,
            minmaxscaler=fe_object["minmax"], 
            labelencoder=fe_object["label"],            
            minmax_columns=args.minmax_columns
            )
        
        X_valid = fe.feature_engineering_process(
            df = X_valid,
            minmaxscaler=fe_object["minmax"], 
            labelencoder=fe_object["label"],        
            minmax_columns=args.minmax_columns
            )

        print(X_train)

        ## modeling
        model_params = load_pickle(f"{args.target_name}_{args.model_name}_params_{args.version}")
        model = Models(model_name=args.model_name, params = model_params)
        fitted_model = model.fit(X_train, y_train)
        y_val_pred = fitted_model.predict(X_valid)
        rmse = mean_squared_error(y_valid, y_val_pred, squared=False)

        save_pickle(fitted_model, f"{args.target_name}_{args.model_name}_{args.version}")

        print(f"rmse of {args.target_name}_{args.version} : ", rmse)
        print(f"=====End {args.target_name}_{args.version} Training=====")

        return rmse
    

if __name__ == "__main__":
    main(args)
