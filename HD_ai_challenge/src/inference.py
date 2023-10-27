import argparse
from utils import *
from preprocess import PreProcessor
from feature_engineering import make_feature
from model import output_index

def inference(args):
    ## load test set
    X_test = load_dataset(mode="inference")
    categorical_feature = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG', 'BREADTH', 'DEPTH', 'DRAUGHT', 'year']
    minmaxscale_feature = ['DIST', 'BUILT', 'DEADWEIGHT', 'GT', 'LENGTH', 'DUBAI', 'BRENT', 'WTI', 'BDI_ADJ', 'PORT_SIZE', ]

    ## preprocess test set
    mean_values_train = load_pickle(file_name="nan_replace_area_mean", path=f"{args.pickle_path}/{args.version}/{args.method}")
    ship_mean_values_train = load_pickle(file_name="nan_replace_ship_mean", path=f"{args.pickle_path}/{args.version}/{args.method}")
    encoder_dict = load_pickle(file_name="labelencoder", path=f"{args.pickle_path}/{args.version}/{args.method}")
    scaler = load_pickle(file_name="minmaxscaler", path=f"{args.pickle_path}/{args.version}/{args.method}")
    
    preprocessing = PreProcessor(categorical_feature=categorical_feature, minmaxscale_feature=minmaxscale_feature)
    X_test = preprocessing.preprocess(X_test, method='area_mean', mean_values=mean_values_train, ship_mean_values=ship_mean_values_train)
    X_test = preprocessing.transform(X_test, encoder=encoder_dict, scaler=scaler)
    
    X_test = make_feature(X_test)

    if args.classification_method == 'True':    
        ## Classification Model
        classifier = load_pickle(file_name=args.classification_model_name, path=f"{args.pickle_path}/{args.version}/{args.method}/classifier")
        none_zero_index = output_index(classifier=classifier, df=X_test)
        print("Done Classification!")
    else: 
        print("No Classification!")
        
    ## Regression Model
    model = load_pickle(file_name=args.regression_model_name, path=f"{args.pickle_path}/{args.version}/{args.method}/regressor")

    if args.classification_method == 'True':    
        y_pred_none_zero = model.predict(X_test.loc[none_zero_index,:])
    
        if args.log_transform == 'True':
            y_pred_none_zero = np.expm1(y_pred_none_zero)
        

        generate_submission_file(
            y_pred=y_pred_none_zero, 
            classification_method=args.classification_method,
            y_pred_none_zero=y_pred_none_zero,
            file_name=f"{args.classification_model_name}&{args.regression_model_name}_{args.method}", 
            path=f"../submission/{args.version}"
            )
    else:
        y_pred = model.predict(X_test)
        
        if args.log_transform == 'True':
            y_pred = np.expm1(y_pred)
    
        y_pred = after_process(y_pred=y_pred)
        
        generate_submission_file(
            y_pred=y_pred, 
            file_name=f"{args.classification_model_name}&{args.regression_model_name}_{args.method}", 
            path=f"../submission/{args.version}"
            )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", "-pp", type=str, default="../pickle", help="path of pickle folder")
    parser.add_argument("--version", "-v", type=str, default="test_version", help="version number")
    parser.add_argument("--method", "-md", type=str, default="test_method", help="describe method")
    parser.add_argument("--log_transform", "-l", type=str, default='False', help="boolean of log transform")
    parser.add_argument("--classification_method", "-c", type=str, default='False', help="boolean of classification method")
    parser.add_argument("--classification_model_name", "-cm", type=str, default=None, help="select classification model")    
    parser.add_argument("--regression_model_name", "-rm", type=str, default="lightgbm", help="select regression model")
    args = parser.parse_args()
    inference(args)