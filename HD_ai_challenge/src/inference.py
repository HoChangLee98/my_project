import argparse
from utils import *
from preprocess import PreProcessor

def inference(args):
    ## load test set
    X_test = load_dataset(mode="inference")
    categorical_feature = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG', 'BREADTH', 'DEPTH', 'DRAUGHT', 'year']
    minmaxscale_feature = ['DIST', 'BUILT', 'DEADWEIGHT', 'GT', 'LENGTH', 'DUBAI', 'BRENT', 'WTI', 'BDI_ADJ', 'PORT_SIZE', ]

    ## preprocess test set
    encoder_dict = load_pickle(file_name="labelencoder", path=f"{args.pickle_path}/{args.version}")
    scaler = load_pickle(file_name="minmaxscaler", path=f"{args.pickle_path}/{args.version}")
    
    preprocessing = PreProcessor(categorical_feature=categorical_feature, minmaxscale_feature=minmaxscale_feature)
    X_test = preprocessing.preprocess(X_test, drop=True)
    X_test = preprocessing.transform(X_test, encoder=encoder_dict, scaler=scaler)
    
    model = load_pickle(file_name=args.model_name, path=f"{args.pickle_path}/{args.version}")
    y_pred = model.predict(X_test)

    submission = pd.read_csv("../submission/sample_submission.csv")
    submission['CI_HOUR'] = y_pred
    submission.to_csv(f"../submission/{args.model_name}_{args.version}.csv", index=False)

    return y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", "-pp", type=str, default="../pickle", help="path of pickle folder")
    parser.add_argument("--version", "-v", type=str, default="test_version", help="version number")
    parser.add_argument("--model_name", "-mn", type=str, default="lgboost", help="select model")
    args = parser.parse_args()
    inference(args)