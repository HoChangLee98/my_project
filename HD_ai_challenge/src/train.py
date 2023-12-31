import argparse
from sklearn.metrics import mean_absolute_error
from utils import *
from preprocess import PreProcessor
from feature_engineering import make_feature
from model import ClassificationModel, output_index, RegressionModel, OptunaProcessor

def train(args):
    ## load data set
    X_train, y_train, X_valid, y_valid = load_dataset(mode='train')
    categorical_feature = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG', 'BREADTH', 'DEPTH', 'DRAUGHT', 'year']
    minmaxscale_feature = ['DIST', 'BUILT', 'DEADWEIGHT', 'GT', 'LENGTH', 'PORT_SIZE']
    
    ## preprocess data set
    preprocessing = PreProcessor(categorical_feature=categorical_feature, minmaxscale_feature=minmaxscale_feature)
    mean_values_train = preprocessing.nan_area_mean_fit(X_train)
    ship_mean_values_train = preprocessing.nan_ship_fit(X_train)

    X_train = preprocessing.preprocess(X_train, method='area_mean', mean_values=mean_values_train, ship_mean_values=ship_mean_values_train)
    X_valid = preprocessing.preprocess(X_valid, method='area_mean', mean_values=mean_values_train, ship_mean_values=ship_mean_values_train)

    encoder_dict = preprocessing.categorical_process_fit(X_train)
    scaler = preprocessing.minmaxscale_process_fit(X_train)
    
    X_train = preprocessing.transform(X_train, encoder=encoder_dict, scaler=scaler)
    X_valid = preprocessing.transform(X_valid, encoder=encoder_dict, scaler=scaler)

    save_pickle(file=mean_values_train, file_name="nan_replace_area_mean", path=f"{args.pickle_path}/{args.version}/{args.method}")
    save_pickle(file=ship_mean_values_train, file_name="nan_replace_ship_mean", path=f"{args.pickle_path}/{args.version}/{args.method}")
    save_pickle(file=encoder_dict, file_name="labelencoder", path=f"{args.pickle_path}/{args.version}/{args.method}")
    save_pickle(file=scaler, file_name="minmaxscaler", path=f"{args.pickle_path}/{args.version}/{args.method}")

    X_train, y_train = reset_data(X_train, y_train)
    X_valid, y_valid = reset_data(X_valid, y_valid)

    ## feature engineering
    X_train = make_feature(X_train)    
    X_valid = make_feature(X_valid)
    print("Done Feature Engineering!")

    ## binary
    if args.classification_method == 'True':
        extractor = ClassificationModel(
            X_train=X_train, 
            y_train=y_train, 
            X_valid=X_valid, 
            y_valid=y_valid, 
            classifier_name=args.classification_model_name,         
            ) 
        classifier = extractor.fit() 
        
        save_pickle(file=classifier, file_name=args.classification_model_name, path=f"{args.pickle_path}/{args.version}/{args.method}/classifier")
        
        train_none_zero_index = output_index(classifier=classifier, df=X_train)
        valid_none_zero_index = output_index(classifier=classifier, df=X_valid)
        X_train_ = X_train.loc[train_none_zero_index,:]
        y_train_ = y_train.loc[train_none_zero_index]
        X_valid_ = X_valid.loc[valid_none_zero_index,:]
        y_valid_ = y_valid.loc[valid_none_zero_index]
    else:
        X_train_ = X_train.copy()
        y_train_ = y_train.copy()
        X_valid_ = X_valid.copy()
        y_valid_ = y_valid.copy()

    ## log transform
    if args.log_transform == 'True':
        y_train_ = np.log1p(y_train_)
        y_valid_ = np.log1p(y_valid_)        

    ## progress optuna
    if args.mode == 'optuna':
        print("     Strat Optuna!")
        optuna = OptunaProcessor(
            X_train=X_train_, 
            y_train=y_train_, 
            X_valid=X_valid_, 
            y_valid=y_valid_, 
            categorical_feature=categorical_feature
            )
        optuna_process = optuna.run_optuna(n_trials=args.n_trials, model_name=args.regression_model_name)
        best_optuna_params = optuna_process.best_trial.params
        print("     Done Optuna!")
        
        save_pickle(file=best_optuna_params, file_name=f"{args.regression_model_name}_optuna", path=f"{args.pickle_path}/{args.version}/{args.method}/regressor")

    else: 
        best_optuna_params = load_pickle(file_name=f"{args.regression_model_name}_optuna", path=f"{args.pickle_path}/{args.version}/{args.method}/regressor")

    ## train model
    model = RegressionModel(
        model_params=best_optuna_params, 
        categorical_feature=categorical_feature, 
        X_train=X_train_, 
        y_train=y_train_, 
        X_valid=X_valid_, 
        y_valid=y_valid_, 
        model_name=args.regression_model_name
        )
     
    model = model.fit()
    
    save_pickle(file=model, file_name=args.regression_model_name, path=f"{args.pickle_path}/{args.version}/{args.method}/regressor")

    if args.classification_method == 'True':    
        y_valid_pred = y_valid.copy()
        y_valid_pred['CI_HOUR'] = 0
        y_valid_pred.loc[valid_none_zero_index] = model.predict(X_valid.loc[valid_none_zero_index,:])
    else: 
        y_valid_pred = model.predict(X_valid_)
        
    if args.log_transform == 'True':
        y_valid_pred = np.expm1(y_valid_pred)

    if args.classification_method == 'False':
        y_valid_pred = after_process(y_pred=y_valid_pred)    

    mae = mean_absolute_error(y_true=y_valid, y_pred=y_valid_pred)
    
    print("Classification Model Name : ", args.classification_model_name)
    print("Regression Model Name : ", args.regression_model_name)
    print("Validation MAE : ", mae)
    print(y_valid_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", "-pp", type=str, default="../pickle", help="path of pickle folder")
    parser.add_argument("--version", "-v", type=str, default="test_version", help="version number")
    parser.add_argument("--method", "-md", type=str, default="test_method", help="describe method")
    parser.add_argument("--classification_method", "-c", type=lambda x:(True if x=='True' else False), default=False, help="boolean of classification method")
    parser.add_argument("--classification_model_name", "-cm", type=str, default=None, help="select classification model")
    parser.add_argument("--log_transform", "-l", type=lambda x:(True if x=='True' else False), default=False, help="boolean of log transform")
    parser.add_argument("--mode", "-m", type=str, default="", help="progress optuna")
    parser.add_argument("--n_trials", "-n", type=int, default=2, help="set number of trials")
    parser.add_argument("--regression_model_name", "-rm", type=str, default="lightgbm", help="select regression model")
    args = parser.parse_args()
    train(args)