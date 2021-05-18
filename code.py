# -*- coding: utf-8 -*-
# -------------------------------------------
# Rare Event prediction with classical 
# modeling approach with a linear classifier
# -------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import datetime
import sys
import statsmodels.api as sm
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
import itertools
import matplotlib.pyplot as plt


class TSAnalysis:
    def __init__(self, infile):
        try:
            raw_data = pd.read_csv(infile)
        except FileNotFoundError as Err:
            print("#" * 90)
            print(f"Check for the input file {infile}")
            print("#" * 90)
            sys.exit(1)

        raw_data['time'] = pd.to_datetime(raw_data['time'])
        # calculate time diff between two successive rows
        raw_data["diff"] = raw_data["time"].diff().apply(
            lambda x: x / np.timedelta64(1, 'm')).fillna(0).astype('int64')
        # windows are built starting from a failure and back-tracking in time
        raw_data["Window"] = TSAnalysis.find_windows(raw_data)
        self.train_data = raw_data.loc[raw_data["time"]
                                       <= datetime.datetime(1999, 5, 22)]
        self.test_data = raw_data.loc[raw_data["time"]
                                      > datetime.datetime(1999, 5, 22)]
        features = list(
            self.train_data.select_dtypes(
                include=[
                    np.number]).columns.values)
        features = [feat for feat in features if feat != 'y']
        self.features = features.copy()
        # modeling variable are finalized based whether they showed signs of non-stationarity
        self.mod_vars = self.find_modeling_vars(raw_data)

    @staticmethod
    def find_windows(indf):
        counter = 1
        window = []
        y_series = indf["y"]
        for idx, elem in enumerate(y_series):
            if idx == 0:
                window.append(1)
            else:
                if y_series[idx] == 0 and y_series[idx - 1] == 1:
                    counter += 1
                    window.append(counter)
                else:
                    window.append(window[idx - 1])
        return window

    def find_modeling_vars(self, inDat):
        """Returns list of modeling vars"""
        min_dates = inDat.groupby("Window")["time"].min()
        max_dates = inDat.groupby("Window")["time"].max()
        left_table = pd.DataFrame()
        for id, from_date, to_date in zip(
                inDat.Window.unique().tolist(), min_dates, max_dates):
            expand_ts = pd.date_range(from_date, to_date, freq="2min")
            expand_df = pd.DataFrame(expand_ts, columns=["time"])
            expand_df["Window"] = id
            left_table = pd.concat([left_table, expand_df], axis=0)

        to_impute = pd.merge(
            left_table, inDat, on=[
                "Window", "time"], how="left")
        imputed_ts = TSAnalysis.impute_ts(
            to_impute, "Window", "time", self.features)
        df = imputed_ts.copy()

        window_feat = []
        mod_vars = []
        for one_feature in self.features:
            for single_window in list(imputed_ts.Window.unique()):
                df = imputed_ts[imputed_ts["Window"] == single_window]
                try:
                    result = adfuller(df[one_feature])
                    if result[1] > 0.05:
                        window_feat.append([single_window, one_feature])
                        mod_vars.append(one_feature)
                        #print("{} - Series is not Stationary for Window - {}".format(one_feature, single_window))
                    else:
                        #print("{} - Series is Stationary for Window - {}".format(one_feature, single_window))
                        pass
                except ValueError:
                    print(
                        f"Test could not be performed for {one_feature} feature, for window {single_window}")

        return list(set(mod_vars))

    @staticmethod
    def applyParallel(dfGrouped, func):
        with Pool(cpu_count()) as p:
            ret_list = p.map(func, [group for name, group in dfGrouped])
        return pd.concat(ret_list)

    def time_series_plots(self):
        color_list = [
            "blue",
            "orange",
            "red",
            "cyan",
            "olive",
            "pink",
            "purple",
            "brown",
            "gray",
            "green"
        ]
        feature_size = int(len(self.features) / 6)
        fig, axes = plt.subplots(nrows=int(np.ceil(feature_size /
                                                   2)), ncols=2, figsize=(14, feature_size *
                                                                          2), dpi=80, facecolor="w", edgecolor="k")

        for i in range(feature_size):
            plot_data = self.train_data[self.features[i]]
            c = color_list[i % (len(color_list))]
            ax = plot_data.plot(
                ax=axes[i // 2, i % 2],
                color=c,
                title="{}".format(self.features[i]),
                rot=25,
            )
            ax.legend([self.features[i]])
        plt.tight_layout()
        plt.show()
        return fig

    def outlier_ys(self):
        df_gp = self.train_data[self.train_data["y"]
                                > 0].groupby(["diff"]).size()
        plt.figure(figsize=(8, 5))
        fig = df_gp.plot(kind='bar')
        fig.set_title('Number of failures by time diff of last reading')
        df_gp.plot(kind='bar')
        plt.show()

    @staticmethod
    def impute_ts(indf, byvar, ordvar, impute_cols):
        # sort before imputation
        ts_dat = indf.sort_values(by=[byvar, ordvar])
        # forward pass
        ts_dat[impute_cols] = ts_dat.groupby(
            byvar)[impute_cols].fillna(method="ffill")
        # backward pass
        ts_dat[impute_cols] = ts_dat.groupby(
            byvar)[impute_cols].fillna(method="bfill")
        # Impute first with global mean and then with 0 for all missings case -
        # open to discussion
        ts_dat[impute_cols] = ts_dat[impute_cols].fillna(
            ts_dat[impute_cols].mean())
        ts_dat[impute_cols] = ts_dat[impute_cols].fillna(0)
        return (ts_dat)

    @staticmethod
    def ols_features(indf, by_col, col_y, suff):
        """ extract all the OLS features"""
        slope_y = ['slope_' + suff + c for c in col_y]
        intercept_y = ['intercept_' + suff + c for c in col_y]
        sq_error_y = ['sq_error_' + suff + c for c in col_y]

        for i, col in enumerate(col_y):
            if (i == 0):
                key_feats = pd.DataFrame(
                    indf[by_col].unique(), columns=[by_col])
            groupbyDF = indf.groupby(by_col)
            split_regress_terms = pd.DataFrame(
                groupbyDF.apply(
                    regress,
                    col,
                    'xaxis').values.tolist(),
                columns=[
                    'params',
                    sq_error_y[i]])  # it was as_matrix here and in the next line
            split_regress_terms[[intercept_y[i], slope_y[i]]] = pd.DataFrame(
                split_regress_terms.params.values.tolist(), index=split_regress_terms.index)
            split_regress_terms = split_regress_terms.drop(['params'], axis=1)
            # get features upto current loop together
            key_feats = pd.concat([key_feats, split_regress_terms], axis=1)
            del (split_regress_terms)

        return (key_feats)

    @staticmethod
    def featuresByTimeWindow(indf, by_col, col_y, suff):
        print("featuresbytimewindow for..", suff)
        """ extract all the OLS features with mean, std, max, min, var, covar
            xaxis variable is created with cat.codes for utc_timestamp_ts
            features are treated as y-axis and xaxis as x-axis for running linear regression
        """
        col_x = "x_axis"

        mean_x = ['mean_' + suff + c for c in [col_x]]
        deviation_x = ['deviation_' + suff + c for c in [col_x]]
        var_x = ['var_' + suff + c for c in [col_x]]

        mean_y = ['mean_' + suff + c for c in col_y]
        deviation_y = ['deviation_' + suff + c for c in col_y]
        max_y = ['max_' + suff + c for c in col_y]
        min_y = ['min_' + suff + c for c in col_y]
        var_y = ['var_' + suff + c for c in col_y]
        slope_y = ['slope_' + suff + c for c in col_y]
        intercept_y = ['intercept_' + suff + c for c in col_y]
        predicted_y = ['predicted_' + suff + c for c in col_y]
        sq_error_y = ['sq_error_' + suff + c for c in col_y]
        covar_y = ['covar_' + suff + c for c in col_y]

        # make x-axis out of timestamps
        # indf = make_xaxis(indf)
        # dataframe having only serial number
        get_serial = pd.DataFrame(indf[by_col].unique(), columns=[by_col])
        print("Processing for time window..", suff)
        for i, col in enumerate(col_y):
            if (i == 0):
                Serials_feats = get_serial
            groupbyDF = indf.groupby(by_col)
            extracts_cov = groupbyDF.apply(lambda x: x['xaxis'].cov(x[col]))
            extracts_des = groupbyDF[col].describe(
            )[['mean', 'std', 'min', 'max']]
            extracts_des['var'] = extracts_des['std']**2
            cov_des = pd.merge(
                extracts_cov.reset_index(),
                extracts_des.reset_index(),
                how='inner',
                on=by_col)
            cov_des.columns = [
                by_col,
                covar_y[i],
                mean_y[i],
                deviation_y[i],
                min_y[i],
                max_y[i],
                var_y[i]]

            # run OLS
            def regress(data, yvar, xvars):
                Y = data[yvar]
                X = data[xvars]
                X = sm.add_constant(X)
                res = sm.OLS(Y, X).fit()
                sum_squared_errors = np.sum((Y - res.predict()) ** 2)
                return np.array(res.params), sum_squared_errors

            split_regress_terms = pd.DataFrame(
                groupbyDF.apply(
                    regress,
                    col,
                    'xaxis').values.tolist(),
                columns=[
                    'params',
                    sq_error_y[i]])  # it was as_matrix here and in the next line
            split_regress_terms[[intercept_y[i], slope_y[i]]] = pd.DataFrame(
                split_regress_terms.params.values.tolist(), index=split_regress_terms.index)
            split_regress_terms = split_regress_terms.drop(['params'], axis=1)

            # get features of current loop together
            serials_sub_feats = pd.concat(
                [cov_des.reset_index(drop=True), split_regress_terms], axis=1)

            # get features upto current loop together
            Serials_feats = pd.merge(
                Serials_feats,
                serials_sub_feats,
                how='inner',
                on=by_col)

            del (cov_des, split_regress_terms, serials_sub_feats)

        return (Serials_feats)

    @staticmethod
    def make_xaxis(indf, byvar="Window", ordvar='time'):
        # pandas-dense-rank
        indf = indf.sort_values([byvar, ordvar], ascending=True)
        indf["xaxis"] = indf.groupby(byvar)[ordvar].rank(
            method="dense", ascending=True)
        return (indf)

    @staticmethod
    def bucketizer(indf, by_col, DateColumn):
        max_dt = indf.groupby(by_col, as_index=False)[DateColumn].max()
        max_dt.columns = [by_col, 'max_dt']
        outdf = pd.merge(indf, max_dt, how='inner', on=by_col)
        outdf['minutes_between'] = (
            outdf['max_dt'] - outdf[DateColumn]) / np.timedelta64(1, 'm')
        return (outdf)

    @staticmethod
    def get_windows(indf, *args):
        """split data for different durations of features"""
        ts_wd1 = TSAnalysis.make_xaxis(
            indf[(indf['minutes_between'] <= args[1]) & (indf['minutes_between'] >= args[0])])
        ts_wd2 = TSAnalysis.make_xaxis(
            indf[(indf['minutes_between'] <= args[2]) & (indf['minutes_between'] >= args[0])])
        ts_wd3 = TSAnalysis.make_xaxis(
            indf[(indf['minutes_between'] <= args[3]) & (indf['minutes_between'] >= args[0])])
        return (ts_wd1, ts_wd2, ts_wd3)

    @staticmethod
    def prepare_Nos(df):
        return df.loc[(df["y"] == 0) & (df["minutes_between"] > 64), :]

    def get_features_by_window(self, indf, *args):
        """Extract features by different windows"""
        ts_wd1, ts_wd2, ts_wd3 = TSAnalysis.get_windows(indf, *args)
        prep_RHS_wd1 = TSAnalysis.featuresByTimeWindow(
            ts_wd1, "Window", self.mod_vars, '16m_')
        prep_RHS_wd2 = TSAnalysis.featuresByTimeWindow(
            ts_wd2, "Window", self.mod_vars, '32m_')
        prep_RHS_wd3 = TSAnalysis.featuresByTimeWindow(
            ts_wd3, "Window", self.mod_vars, '48m_')
        all_windows = prep_RHS_wd1.merge(
            prep_RHS_wd2, on=["Window"]).merge(
            prep_RHS_wd3, on=["Window"])
        return all_windows

    def data_for_modeling(self):
        train_data = self.bucketizer(self.train_data, "Window", "time")
        train_yes = train_data.copy()
        train_Nos = TSAnalysis.prepare_Nos(train_data)

        test_data = self.bucketizer(self.test_data, "Window", "time")
        test_yes = test_data.copy()
        test_Nos = TSAnalysis.prepare_Nos(test_data)

        # Get features for the datasets
        # by collapsing time dimension and extracting secondary features
        mod_train_yes = self.get_features_by_window(
            train_yes, 16, 32, 48, 64)
        mod_test_yes = self.get_features_by_window(
            test_yes, 16, 32, 48, 64)
        mod_train_yes["y"], mod_test_yes["y"] = 1, 1

        mod_train_Nos = self.get_features_by_window(
            train_Nos, 64, 80, 96, 112)
        mod_test_Nos = self.get_features_by_window(
            test_Nos, 64, 80, 96, 112)
        mod_train_Nos["y"], mod_test_Nos["y"] = 0, 0

        return (mod_train_yes, mod_test_yes, mod_train_Nos, mod_test_Nos)

    @staticmethod
    def plot_roc(y_actual, y_pred, st_label):
        fpr, tpr, threshold = roc_curve(y_actual, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label=st_label + ' = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()


def main():
    infile = "./data.csv"
    tsa = TSAnalysis(infile)

    # plot raw independent variables
    _ = tsa.time_series_plots()
    # Take a look at the outlier ys
    _ = tsa.outlier_ys()

    # prepare modeling train and test 1s and 0s datasets
    (mod_train_yes, mod_test_yes, mod_train_Nos, mod_test_Nos) = \
        tsa.data_for_modeling()

    mod_train = pd.concat([mod_train_yes, mod_train_Nos], axis=0)
    mod_test = pd.concat([mod_test_yes, mod_test_Nos], axis=0)
    
    # Take a look at the stratification
    print(mod_train.y.value_counts())
    print(mod_test.y.value_counts())
    
    # begin modeling exercise
    # tried linearSVC prior to this experiment and that failed to converge
    fit_vars = [col for col in mod_train if col != 'y']
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=10000,
        penalty='l1',
        solver='saga',
        C=0.01)

    # Get probabilities and predictions from the trained model
    # for train/test datasets
    probs = clf.fit(
        mod_train[fit_vars],
        mod_train['y']).predict_proba(
        mod_train[fit_vars])
    preds = probs[:, 1]
    _ = TSAnalysis.plot_roc(mod_train["y"], preds, "TRAINING AUC")

    probs = clf.fit(
        mod_train[fit_vars],
        mod_train['y']).predict_proba(
        mod_test[fit_vars])
    preds = probs[:, 1]
    _ = TSAnalysis.plot_roc(mod_test["y"], preds, "TEST AUC")

    preds = clf.fit(
        mod_train[fit_vars],
        mod_train['y']).predict(
        mod_test[fit_vars])

    # build confusion matrix from test results
    cnf_matrix = confusion_matrix(mod_test['y'], preds)
    np.set_printoptions(precision=2)
    print(cnf_matrix)
    plt.figure()
    _ = TSAnalysis.plot_confusion_matrix(
        cnf_matrix,
        classes=[
            'Break',
            'No-Break'],
        title='Confusion matrix, without normalization')

    # Calculate weighted F1 score of the model
    f1Score = f1_score(mod_test['y'], preds, average='weighted')
    print([f"F1 score of the model is {f1Score}"])


if __name__ == "__main__":
    main()
