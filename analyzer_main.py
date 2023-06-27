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
NORMALIZE = False
# verbosity level
VERBOSE = 1

# Classifiers to be tested. The model should be able to be dumped with joblib
CLASSIFIERS = {'dt': DecisionTreeClassifier(),
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
               'knn_1': KNeighborsClassifier(n_neighbors=1),
               'knn_3': KNeighborsClassifier(n_neighbors=3),
               'knn_9': KNeighborsClassifier(n_neighbors=9),
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
    clf_list = collections.OrderedDict(sorted(CLASSIFIERS.items()))

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

    dataframes = {}
    for file in os.listdir(DATASETS_FOLDER):
        if file.endswith(".csv"):
            df_name = file.replace('.csv', '')
            df = pandas.read_csv(os.path.join(DATASETS_FOLDER, file))
            df = df.drop(columns=[TIMESTAMP_TAG])
            dataframes[df_name] = df
            if VERBOSE > 0:
                print('Dataset %s: %d rows and %d columns' % (df_name, len(df.index), len(df.columns)))

    OUT_FILE = str(current_ms()) + '_output.csv'
    with open(OUT_FILE, 'w') as f:
        f.write('clf,train_dataset,train_time,model_size,train_an_perc,')
        for dn in dataframes:
            f.write(dn + '@test_size,' + dn + '@an_perc,' + dn + '@test_time,' +
                    dn + '@acc,' + dn + '@mcc,')
        f.write('\n')

    for train_dataset_name in dataframes:

        if VERBOSE > 0:
            print('\nUsing Dataset %s for training' % train_dataset_name)

        x_train, x_test, y_train, y_test = ms.train_test_split(dataframes[train_dataset_name].drop(columns=LABEL_NAME),
                                                               dataframes[train_dataset_name][
                                                                   LABEL_NAME].to_numpy() if not BINARY_CLASSIFICATION else binarize_label(
                                                                   dataframes[train_dataset_name][
                                                                       LABEL_NAME].to_numpy(), NORMAL_TAG),
                                                               test_size=0.3, shuffle=True)
        
        # Normalize if needed
        if NORMALIZE:
            normalizer = MinMaxScaler()
            normalizer.fit(x_train, y_train)
            x_train = normalizer.transform(x_train)
            x_test = normalizer.transform(x_test)
        
        test_datasets = {}
        for test_dataset_name in dataframes:
            test_datasets[test_dataset_name] = {}
            if test_dataset_name == train_dataset_name:
                test_datasets[test_dataset_name]['x'] = x_test
                test_datasets[test_dataset_name]['y'] = y_test
            else:
                test_datasets[test_dataset_name]['x'] = dataframes[test_dataset_name].drop(columns=LABEL_NAME)
                if NORMALIZE:
                    test_datasets[test_dataset_name]['x'] = normalizer.transform(test_datasets[test_dataset_name]['x'])
                test_datasets[test_dataset_name]['y'] = dataframes[test_dataset_name][
                    LABEL_NAME].to_numpy() if not BINARY_CLASSIFICATION else binarize_label(
                    dataframes[test_dataset_name][LABEL_NAME].to_numpy(), NORMAL_TAG)
            if VERBOSE > 0:
                print('New test dataset %s: %d rows and %.2f percent of anomalies'
                      % (test_dataset_name, len(test_datasets[test_dataset_name]['y']),
                         (len([ele for ele in test_datasets[test_dataset_name]['y'] if ele != 'normal'])) /
                         len(test_datasets[test_dataset_name]['y']) * 100.0))

        # Training and Testing Classifiers
        for clf_name in clf_list:
            clf = copy.deepcopy(clf_list[clf_name])
            if VERBOSE > 0:
                print('\nTraining classifier %s' % clf_name)
            file_str = clf_name + ',' + train_dataset_name + ','

            start_ms = current_ms()
            clf.fit(x_train, y_train)
            train_time = current_ms() - start_ms

            # Dump, reload and measure the size of the model
            model_path = os.path.join(MODELS_FOLDER, clf_name + '_' + train_dataset_name + '.joblib')
            joblib.dump(clf, model_path, compress=9)
            model_size = -1 if not os.path.exists(model_path) else os.stat(model_path).st_size
            clf = joblib.load(model_path)
            if clf is None or model_size == -1:
                print('Something went wrong when saving the file')
            file_str = file_str + str(train_time) + ',' + str(model_size) + ',' \
                       + str((len([ele for ele in y_train if ele != 'normal'])) / len(y_train) * 100.0) + ','

            for test_dataset in test_datasets:
                start_time = current_ms()
                y_pred = clf.predict(test_datasets[test_dataset]['x'])
                test_time = current_ms() - start_time
                y_true = test_datasets[test_dataset]['y']
                an_perc = (len([ele for ele in y_true if ele != 'normal'])) / len(y_true) * 100.0
                file_str = file_str + str(len(y_pred)) + ',' + str(an_perc) + ',' + str(test_time) + ',' \
                           + str(metrics.accuracy_score(y_true=y_true, y_pred=y_pred)) \
                           + ',' + str(metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred)) + ','
                print('Testing on %s: MCC=%.2f' %
                      (test_dataset, metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred)))

            with open(OUT_FILE, 'a') as f:
                f.write(file_str + '\n')
