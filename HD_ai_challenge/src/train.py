import argparse
from sklearn.metrics import mean_absolute_error
from utils import *
from preprocess import PreProcessor
from model import Model, OptunaProcessor

def train(args):
    ## load data set
    X_train, y_train, X_valid, y_valid = load_dataset(mode='train')
    categorical_feature = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG', 'BREADTH', 'DEPTH', 'DRAUGHT', 'year']
    minmaxscale_feature = ['DIST', 'BUILT', 'DEADWEIGHT', 'GT', 'LENGTH', 'DUBAI', 'BRENT', 'WTI', 'BDI_ADJ', 'PORT_SIZE', ]

    ## preprocess data set
    preprocessing = PreProcessor(categorical_feature=categorical_feature, minmaxscale_feature=minmaxscale_feature)
    X_train = preprocessing.preprocess(X_train, drop=True)
    X_valid = preprocessing.preprocess(X_valid, drop=True)

    encoder_dict = preprocessing.categorical_process_fit(X_train)
    scaler = preprocessing.minmaxscale_process_fit(X_train)
    
    X_train = preprocessing.transform(X_train, encoder=encoder_dict, scaler=scaler)
    X_valid = preprocessing.transform(X_valid, encoder=encoder_dict, scaler=scaler)

    save_pickle(file=encoder_dict, file_name="labelencoder", path=f"{args.pickle_path}/{args.version}")
    save_pickle(file=scaler, file_name="minmaxscaler", path=f"{args.pickle_path}/{args.version}")

    X_train, y_train = reset_data(X_train, y_train)
    X_valid, y_valid = reset_data(X_valid, y_valid)

    ## progress optuna
    if args.mode == 'optuna':
        print("     Strat Optuna!")
        optuna = OptunaProcessor(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, categorical_feature=categorical_feature)
        optuna_process = optuna.run_optuna(n_trials=args.n_trials, model_name=args.model_name)
        best_optuna_params = optuna_process.best_trial.params
        print("     Done Optuna!")
        
        save_pickle(file=best_optuna_params, file_name=f"{args.model_name}_optuna", path=f"{args.pickle_path}/{args.version}")

    else: 
        best_optuna_params = load_pickle(file_name=f"{args.model_name}_optuna", path=f"{args.pickle_path}/{args.version}")

    ## train model
    model = Model(
        model_params=best_optuna_params, 
        categorical_feature=categorical_feature, 
        X_train=X_train, 
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        model_name=args.model_name
        )
    model = model.fit()
    
    save_pickle(file=model, file_name=args.model_name, path=f"{args.pickle_path}/{args.version}")

    mae = mean_absolute_error(model.predict(X_valid), y_valid)
    print("Model Name : ", args.model_name)
    print("Validation MAE : ", mae)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", "-pp", type=str, default="../pickle", help="path of pickle folder")
    parser.add_argument("--version", "-v", type=str, default="test_version", help="version number")
    parser.add_argument("--mode", "-m", type=str, default="", help="progress optuna")
    parser.add_argument("--n_trials", "-n", type=int, default=2, help="set number of trials")
    parser.add_argument("--model_name", "-mn", type=str, default="lgboost", help="select model")
    args = parser.parse_args()
    train(args)