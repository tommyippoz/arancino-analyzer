import argparse
import collections
import copy
import os

import joblib
import numpy
import pandas
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
# random state
RANDOM_STATE = 43


# Classifiers to be tested. The model should be able to be dumped with joblib
def get_classifiers():
    return {'dt': DecisionTreeClassifier(),
            'g_nb': GaussianNB(),
            'm_nb': MultinomialNB(),
            'b_nb': BernoulliNB(),
            'lda': LinearDiscriminantAnalysis(),
            'rf_100': RandomForestClassifier(n_estimators=100),
            'rf_30': RandomForestClassifier(n_estimators=30),
            'rf_10': RandomForestClassifier(n_estimators=10),
            'gb_100': GradientBoostingClassifier(n_estimators=100),
            'gb_30': GradientBoostingClassifier(n_estimators=30),
            'gb_10': GradientBoostingClassifier(n_estimators=10),
            'sgd': SGDClassifier(),
            'mlp': Perceptron(),
            'lr': LogisticRegression(),
            'st_stat': StackingClassifier(estimators=[('nb', GaussianNB()),
                                                      ('lda', LinearDiscriminantAnalysis()),
                                                      ('lr', LogisticRegression())],
                                          final_estimator=DecisionTreeClassifier()),
            'tree_stat': StackingClassifier(estimators=[('dt', DecisionTreeClassifier()),
                                                        ('rf', RandomForestClassifier(n_estimators=10)),
                                                        ('gb', GradientBoostingClassifier(n_estimators=10))],
                                            final_estimator=LinearDiscriminantAnalysis()),
            }


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
        print('        Training and testing Anomaly Detectors for the ARANCINO')
        print('                  MODE: %s classification' % ('binary' if BINARY_CLASSIFICATION else 'multiclass'))
        print('-----------------------------------------------------------------------\n')

    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    OUT_FILE = str(current_ms()) + '_output.csv'
    with open(OUT_FILE, 'w') as f:
        f.write(
            'clf,train_dataset,train_size,train_time,model_size,train_an_perc,test_dataset,test_size,test_an_perc,test_time,acc,mcc\n')

    for file in os.listdir(DATASETS_FOLDER):
        if file.endswith(".csv"):
            train_dataset_name = file.replace('.csv', '')
            df = pandas.read_csv(os.path.join(DATASETS_FOLDER, file))
            df = df.drop(columns=[TIMESTAMP_TAG])
            if VERBOSE > 0:
                print('Dataset %s: %d rows and %d columns' % (train_dataset_name, len(df.index), len(df.columns)))

            x_train, x_test, y_train, y_test = \
                ms.train_test_split(df.drop(columns=LABEL_NAME),
                                    df[LABEL_NAME].to_numpy() if not BINARY_CLASSIFICATION else binarize_label(
                                        df[LABEL_NAME].to_numpy(), NORMAL_TAG),
                                    test_size=0.3, shuffle=True)

            # Normalize if needed
            if NORMALIZE:
                normalizer = MinMaxScaler()
                normalizer.fit(x_train, y_train)
                x_train = normalizer.transform(x_train)
                x_test = normalizer.transform(x_test)

            df = None

            train_an_perc = (len([ele for ele in y_train if ele != 'normal'])) / len(y_train) * 100.0
            test_an_perc = (len([ele for ele in y_test if ele != 'normal'])) / len(y_test) * 100.0

            # Training and Testing Classifiers
            for (clf_name, clf) in get_classifiers().items():
                if VERBOSE > 0:
                    print('\nTraining classifier %s' % clf_name)

                start_ms = current_ms()
                clf.fit(x_train, y_train)
                train_time = current_ms() - start_ms

                if VERBOSE > 0:
                    print('Dumping on file')

                # Dump, reload and measure the size of the model
                model_path = os.path.join(MODELS_FOLDER, clf_name + '@' + train_dataset_name + '.joblib')
                joblib.dump(clf, model_path, compress=9)
                model_size = -1 if not os.path.exists(model_path) else os.stat(model_path).st_size
                try:
                    clf = joblib.load(model_path)
                    if clf is None or model_size == -1:
                        print('Something went wrong when saving the file')
                    else:
                        start_time = current_ms()
                        y_pred = clf.predict(x_test)
                        test_time = current_ms() - start_time
                        print('Testing %s/%s: MCC=%.2f' %
                              (clf_name, train_dataset_name, metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred)))

                        with open(OUT_FILE, 'a') as f:
                            f.write(clf_name + ',' + train_dataset_name + ',' + str(len(y_train)) + ',' + str(
                                train_time) + ','
                                    + str(model_size) + ',' + str(train_an_perc) + ',' + train_dataset_name
                                    + ',' + str(len(y_test)) + ',' + str(test_an_perc) + ',' + str(test_time) + ','
                                    + str(metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
                                    + ',' + str(metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred)) + '\n')
                except:
                    print('Something went wrong when saving the file')
