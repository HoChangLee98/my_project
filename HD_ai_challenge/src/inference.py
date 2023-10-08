import argparse
from utils import *
from preprocess import PreProcessor
from model import output_index

def inference(args):
    ## load test set
    X_test = load_dataset(mode="inference")
    categorical_feature = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG', 'BREADTH', 'DEPTH', 'DRAUGHT', 'year']
    minmaxscale_feature = ['DIST', 'BUILT', 'DEADWEIGHT', 'GT', 'LENGTH', 'DUBAI', 'BRENT', 'WTI', 'BDI_ADJ', 'PORT_SIZE', ]

    ## preprocess test set
    mean_values_train = load_pickle(file_name="nan_replace_mean", path=f"{args.pickle_path}/{args.version}/{args.method}")
    encoder_dict = load_pickle(file_name="labelencoder", path=f"{args.pickle_path}/{args.version}/{args.method}")
    scaler = load_pickle(file_name="minmaxscaler", path=f"{args.pickle_path}/{args.version}/{args.method}")
    
    preprocessing = PreProcessor(categorical_feature=categorical_feature, minmaxscale_feature=minmaxscale_feature)
    X_test = preprocessing.preprocess(X_test, method='mean', mean_values=mean_values_train)
    X_test = preprocessing.transform(X_test, encoder=encoder_dict, scaler=scaler)
    
    ## Classification Model
    classifier = load_pickle(file_name=args.classification_model_name, path=f"{args.pickle_path}/{args.version}/{args.method}/classifier")
    none_zero_index = output_index(classifier=classifier, df=X_test)
    
    ## Regression Model
    model = load_pickle(file_name=args.regression_model_name, path=f"{args.pickle_path}/{args.version}/{args.method}/regressor")
    y_pred_none_zero = model.predict(X_test.loc[none_zero_index,:])

    generate_submission_file(
        y_pred_none_zero=y_pred_none_zero, 
        none_zero_index=none_zero_index,
        file_name=f"{args.classification_model_name}&{args.regression_model_name}_{args.method}", 
        path=f"../submission/{args.version}"
        )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", "-pp", type=str, default="../pickle", help="path of pickle folder")
    parser.add_argument("--version", "-v", type=str, default="test_version", help="version number")
    parser.add_argument("--method", "-md", type=str, default="test_method", help="describe method")
    parser.add_argument("--classification_model_name", "-cm", type=str, default="lightgbm", help="select classification model")    
    parser.add_argument("--regression_model_name", "-rm", type=str, default="lightgbm", help="select regression model")
    args = parser.parse_args()
    inference(args)