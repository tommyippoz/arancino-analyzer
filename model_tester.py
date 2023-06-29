import argparse
import collections
import copy
import os
import sys

import joblib
import numpy
import pandas
import sklearn.metrics
import sklearn.metrics as metrics
import sklearn.model_selection as ms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from analyzerlib.utils import binarize_label, current_ms

# Name of the file containing monitored data
DATASETS_FOLDER = 'datasets'
# Name of the label column
LABEL_NAME = 'label'
# Name of the normal tag
NORMAL_TAG = 'normal'
# Name of the file containing injections
MODELS_FOLDER = 'models'
# Tag of the timestamp
TIMESTAMP_TAG = '_timestamp'
# Type of classification (binary or multiclass)
BINARY_CLASSIFICATION = True
# True if data has to be normalized before processing
NORMALIZE = True
# verbosity level
VERBOSE = 1

if __name__ == '__main__':
    """
    Main to train and test models for anomaly detection on the ARANCINO
    """

    # Parse Arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-df", "--datafolder", type=str,
                           help="location of the datasets; default is 'datasets' in the current folder")
    argParser.add_argument("-mf", "--modfolder", type=str,
                           help="location in which we save trained models; default is 'models' in the current folder")
    argParser.add_argument("-lt", "--labeltag", type=str,
                           help="name of the label column; default is 'label'")
    argParser.add_argument("-nt", "--normaltag", type=str,
                           help="tag of the norml class; default is 'normal'")
    argParser.add_argument("-t", "--timetag", type=str,
                           help="tag of the timestamp in the monitor file; default is '_timestamp'")
    argParser.add_argument("-bc", "--binaryclassification", type=bool,
                           help="True if you aim at binary classification, False otherwise; default is True")
    argParser.add_argument("-n", "--normalize", type=bool,
                           help="True if you want to normalize data, False otherwise; default is False")
    argParser.add_argument("-v", "--verbose", type=int,
                           help="0 if all messages need to be suppressed, 2 if all have to be shown. "
                                "1 displays base info")
    args = argParser.parse_args()
    if hasattr(args, 'datafolder') and args.datafolder is not None and os.path.exists(args.datafolder):
        DATASETS_FOLDER = args.datafolder
    if hasattr(args, 'modfolder') and args.modfolder is not None:
        MODELS_FOLDER = args.modfolder
    if hasattr(args, 'labeltag') and args.labeltag is not None:
        LABEL_NAME = args.labeltag
    if hasattr(args, 'normaltag') and args.normaltag is not None:
        NORMAL_TAG = args.normaltag
    if hasattr(args, 'timetag') and args.timetag is not None:
        TIMESTAMP_TAG = args.timetag
    if hasattr(args, 'binaryclassification') and args.binaryclassification is not None:
        BINARY_CLASSIFICATION = args.binaryclassification
    if hasattr(args, 'normalize') and args.normalize is not None:
        NORMALIZE = args.normalize
    if hasattr(args, 'verbose') and args.verbose is not None:
        VERBOSE = int(args.verbose)

    if VERBOSE > 0:
        print('-----------------------------------------------------------------------')
        print('               Testing Anomaly Detectors for the ARANCINO')
        print('                  MODE: %s classification' % ('binary' if BINARY_CLASSIFICATION else 'multiclass'))
        print('-----------------------------------------------------------------------\n')

    if not os.path.exists(MODELS_FOLDER):
        print('models folder %s does not exist. Exiting' % MODELS_FOLDER)
        sys.exit(1)

    OUT_FILE = str(current_ms()) + '_output.csv'
    with open(OUT_FILE, 'w') as f:
        if BINARY_CLASSIFICATION:
            f.write('clf,train_dataset,test_dataset,test_size,an_perc,test_time,acc,mcc,tp,tn,fp,fn,fpr,p,r\n')
        else:
            f.write('clf,train_dataset,test_dataset,test_size,an_perc,test_time,acc,mcc\n')

    for file in os.listdir(DATASETS_FOLDER):
        if file.endswith(".csv"):
            df_name = file.replace('.csv', '')
            df = pandas.read_csv(os.path.join(DATASETS_FOLDER, file))
            df = df.drop(columns=[TIMESTAMP_TAG])
            if VERBOSE > 0:
                print('Dataset %s: %d rows and %d columns' % (df_name, len(df.index), len(df.columns)))

            x_train, x_test, y_train, y_test = \
                ms.train_test_split(df.drop(columns=LABEL_NAME),
                                    df[LABEL_NAME].to_numpy() if not BINARY_CLASSIFICATION else binarize_label(
                                        df[LABEL_NAME].to_numpy(), NORMAL_TAG),
                                    test_size=0.3, shuffle=True)

            # Normalize if needed
            if NORMALIZE:
                normalizer = MinMaxScaler()
                normalizer.fit(x_train, y_train)
                x_test = normalizer.transform(x_test)

            x_train = None
            y_train = None
            df = None

            an_perc = (len([ele for ele in y_test if ele != 'normal'])) / len(y_test) * 100.0

            # Training and Testing Classifiers
            for model_file in os.listdir(MODELS_FOLDER):
                if model_file.endswith('.joblib'):
                    clf_name = model_file.replace('.joblib', '').split('@')[0].strip()
                    train_dataset = model_file.replace('.joblib', '').split('@')[1].strip()
                    model_path = os.path.join(MODELS_FOLDER, model_file)
                    try:
                        clf = joblib.load(model_path)
                        if clf is None:
                            print('Something went wrong when loading the file')
                        else:
                            start_time = current_ms()
                            y_pred = clf.predict(x_test)
                            test_time = current_ms() - start_time
                            print('Testing %s/%s on %s: MCC=%.2f' %
                                  (clf_name, train_dataset, df_name, metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred)))

                            with open(OUT_FILE, 'a') as f:
                                f.write(clf_name + ',' + train_dataset + ',' + df_name + ',' + str(len(y_test))
                                        + ',' + str(an_perc) + ',' + str(test_time) + ','
                                        + str(metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
                                        + ',' + str(metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred)))
                                if BINARY_CLASSIFICATION:
                                    [[tn, fn], [fp, tp]] = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pred, labels=['normal', 'anomaly'])
                                    f.write(',' + str(tp) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn)
                                            + ',' + str(1.0 * fp / (fp + tn))
                                            + ',' + str(1.0 * tp / (fp + tp))
                                            + ',' + str(1.0 * tp / (fn + tp)))
                                f.write('\n')
                    except:
                        print('Something went wrong when loading the file')
