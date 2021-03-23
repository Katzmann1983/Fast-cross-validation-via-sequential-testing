import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.grid_search import RandomizedSearchCV
import xgboost as xgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import classifiers_CV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("winequality-white.csv", sep=";")
df_y = df.quality.values
df_x = df.drop("quality", axis=1)
train, test, y, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)


class fast_cross_validation:
    def __init__(self, train, test, y):
        self.train = train
        self.test = test
        self.y = y

    def topConfigurations(P1, mean_performance, alpha, K):
        sum_1 = []
        sort_pre = []
        Y = P1.shape
        print(
            "Mean Performance", mean_performance
        )  # mean performance of all configuration
        t = np.argsort(mean_performance)
        P_pp = t[::-1]
        sorted_list = np.zeros(Y[1])
        for s in range(0, len(P_pp)):
            sorted_list = np.vstack(
                (sorted_list, P1[P_pp[s]])
            )  # Sort Pp according to the mean performance
        sorted_list = np.delete(sorted_list, 0, 0)
        sorted_list = np.matrix(np.array(sorted_list))
        alpha_2 = alpha / K  # K is number of Active configutation
        for k in range(1, len(sorted_list)):
            p = cochrans_q(
                sorted_list[0:k, :]
            )  # Cochrans_q test for finding the top configuration
            if p[1] <= alpha_2:
                break
        k = k + 1
        for x in range(0, k):
            sorted_list[x] = 1  # Top configuration as a 1
        for y in range(k, len(sorted_list)):
            sorted_list[y] = 0
        for s in range(0, len(sorted_list)):  # Rests are zeros
            sum_1.append(sorted_list[s].max())
        sort_pre = np.zeros(len(sum_1))
        for b, c in zip(P_pp, sum_1):
            setitem(sort_pre, b, c)  # Configurations back to their orginal sequence
        return sort_pre

    def isFlopConfiguration(T, s, S, beta=0.1, alpha=0.01):
        pi_0 = 0.05
        pi_1 = 0.0
        g = 1 / float(S)
        alpha_l = 0.01
        beta_l = 0.1
        pi_1 = (((1 - beta_l) / alpha_l) ** g) / 2
        num = math.log(beta_l / (1 - alpha_l))
        dum = math.log(pi_1 / pi_0) - math.log((1 - pi_1) / (1 - pi_0))
        num_1 = math.log((1 - pi_0) / (1 - pi_1))
        dum_1 = math.log(pi_1 / pi_0) - math.log((1 - pi_1) / (1 - pi_0))
        a = num / dum
        b = num_1 / dum_1
        if sum(T) <= (a + b * s):
            return T

    def similarPerformance(TS, alpha):
        pw = cochrans_q(TS)
        return pw[1]

    def selectWinnner(PS, isActive, wstop, s):
        Rx = []
        p = PS.shape
        Rs = np.empty(p)
        Rs[:] = np.NAN
        print("isActive[c]", isActive)
        for i in range(0, p[1]):
            Rs[:, i] = rankdata(PS[:, i])  # Gather the rank of c in step i
        print("Rs", Rs)
        Ms = np.zeros(p[0])
        print("S", s)
        for c in range(p[0]):
            if isActive[c] == 1:
                Rx = Rs[c, s - wstop + 1 : s]
                print("Rx", Rx)
                Rx = sum(Rx)
                print("Rx", Rx)
                Ms[c] = Rx / wstop  # Mean rank for the last wstop steps
        print("Ms", Ms)
        return np.argmax(Ms)  # Return configuration with minimal mean rank

    def parametersList(self, conf):
        """
        This function return the candidate parameters matrix for the classifier/model.
        :param conf: Number of configurations of paramter for classifier
        :return: Paramter matrix
        """
        train_size = len(self.train)  # Size of training set
        parameters = []
        configurations = conf  # Number of Configutions
        algo_param_numbers = 5
        parameters_list = np.zeros(
            algo_param_numbers
        )  # List for sroting the parameters of algorithm
        for c in range(0, configurations):
            params_iterate = []  # Paramters list for each iteration
            clf = (
                classifiers_CV.classifiers.random_forest_classifier()
            )  # Choosen classifier for prediction
            param_dist = (
                {  # Choose discret and continous values for parameters randomly
                    "n_estimators": sp_randint(1, 500),
                    "max_depth": sp_uniform(0.1, 1),
                    "max_features": sp_randint(1, 11),
                    "min_samples_split": sp_randint(2, 20),
                    "min_samples_leaf": sp_randint(1, 11),
                }
            )
            random_search = RandomizedSearchCV(
                clf, param_distributions=param_dist
            )  # Randomized search on hyper parameters
            random_forest_model = random_search.fit(self.train, self.y)
            for key in random_search.best_params_:
                params_iterate.append(random_search.best_params_[key])
            parameters_list = np.vstack((parameters_list, params_iterate))
        return parameters_list

    def CSVT_main_loop(self, fold, wstop, configuration, alpha, beta):
        """
         This function is a main loop of the cross validation algorithm which selects top configurations
        :param features:
        :param target:Final Configuration
        :return: none
        """
        performce_mean = np.empty(
            [configuration, fold]
        )  # Matrix for storing Mean Performance
        performce_mean[:] = np.NAN  # ?????????????????????????????????
        matrix_trace = np.zeros(configuration)
        isActive = np.ones(configuration)  # Active Configuraion
        train_size = len(self.train)
        n = (train_size + 1) / fold
        print("n", n)  # Initialize subset increment
        pp_matrix = np.zeros(train_size - 2)  # Pointwise perofrmance matrix
        top_configurations = []  # for storing the top configurations
        score = []  # ?????????????????????????????????
        Ty = np.zeros(configuration)
        print(isActive)
        parameters_list = self.parametersList(configuration)
        for fd in range(
            1, fold + 1
        ):  # To find the top performing configurations for fold fd
            print("fold Number", fd)  # Total Number of folds for cross validation
            performace_matrix = []
            for c in range(0, configuration):
                if isActive[c] == 1:
                    K = sum(isActive)
                    ind1 = (fd - 1) * int(n)
                    ind2 = (fd * int(n)) - 2
                    print("index 1......", ind1)
                    print("index 2....", ind2)
                    x = self.train.values[
                        ind1:ind2
                    ]  # Range of dataset of train in current fold
                    z = self.y[
                        ind1:ind2
                    ]  # Range of dataset of prediction in current fold
                    test_CV = self.train.drop(
                        train.index[[ind1, ind2]]
                    )  # Rest of the data for testing
                    v = pd.DataFrame(self.y)
                    y_test = v.drop(v.index[[ind1, ind2]])  # prediction of test data
                    myarray = np.array(parameters_list[c]).tolist()
                    clf = classifiers_CV.classifiers.random_forest_classifier(
                        n_estimators=myarray[0],
                        max_depth=myarray[1],
                        max_features=myarray[2],
                        min_samples_split=myarray[3],
                        min_samples_leaf=myarray[4],
                    )
                    random_forest_model = clf.fit(x, z)
                    reds = clf.predict_proba(test_CV)  # Predict probabilities
                    reds = np.array(reds[:, 1]).tolist()
                    print("y_test", y_test)
                    print("red", reds)
                    y_test = np.array(y_test)
                    roc_score = roc_auc_score(y_test, reds)
                    print("roc_score", roc_score)
                    performace_matrix = np.append(performace_matrix, roc_score)
                    performce_mean[
                        c, fd - 1
                    ] = roc_score  # Mean performance of each configuration
                    for l in range(
                        0, len(reds)
                    ):  # convertion of XGboost prediction into binary form
                        if reds[l] > 0.5:
                            reds[l] = 1
                        else:
                            reds[l] = 0
                    print("#" * 10)
                    c = c - 1
                    pp_matrix = np.vstack((pp_matrix, reds))

            pp_matrix = np.delete(pp_matrix, 0, 0)  # pointwise peroformance matrix
            top_configurations = self.topConfigurations(
                pp_matrix, performace_matrix, alpha, K
            )  # Find the top configurations
            A = np.where(isActive == 1)
            print(A[0])
            Ty[A[0]] = top_configurations
            matrix_trace = np.vstack((matrix_trace, Ty))
            # Configurations are column-wise and folds are Row-wise                                      #Top configurations are "1" in columns

            print("isActive", isActive)
            for z in range(0, len(matrix_trace[0])):
                T = self.isFlopConfiguration(
                    matrix_trace[:, z], fd, fold, beta, alpha
                )  # Checking each configuration whether its Flop or not
                if T is not None:  # D-Active Flop Configuration
                    isActive[z] = 0
            print("is ative", isActive)
            isActive_index = np.where(
                isActive == 1
            )  # Slection the index of configurations which are not flop
            isActive_index = np.array(isActive_index)
            print("isActive_index[0]", isActive_index[0])
            trace_matrix = np.delete(matrix_trace, 0, 0)
            trace_matrix = trace_matrix.T
            print("trace_matrix\n", trace_matrix)
            p = self.similarPerformance(
                trace_matrix[isActive_index[0], (f - wstop + 1) : f], alpha
            )
            if (
                p <= alpha
            ):  # checks whether all remaining configurations performed equally well in the past
                break
        Final_asnwer = self.selectWinnner(performce_mean, isActive, wstop, f)
        print("Final_answer", Final_asnwer)


r = fast_cross_validation(train, test, y)
r.CSVT_main_loop(fold=20, wstop=6, configuration=5, alpha=0.5, beta=0.1)
