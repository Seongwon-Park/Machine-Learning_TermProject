# Classification Tele Customer Churn Prediction

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ********************************* Inspection Function ********************************* #

# Load the dataset
# Customer ID column is unusable column
# So, Drop that column
def loadDataset() :
    df = pd.read_csv('cell2celltrain.csv')
    df = df.drop(columns = 'CustomerID', axis = 1)
    return df

# Data Inspection : Null count, Target attribute distribution, Dataset Info
def inspectionDataset(df) :
    # Print the dataset basic information
    print(df.head())
    print(df.describe())
    print(df.info())
    # Visualize the dataset for inspection
    countChurn(df)
    ratioChurn(df)
    nanCount(df)

# Find attribute that has the best correlation value top 9 with target attribute, continuous and categorical each
def inspectionCorr(df) :
    categorical = []
    continuous = []
    index = 0
    for col in df.columns :
        if df.dtypes[col] == object:
            if (col != "Churn") :
                categorical.append(index)
            index += 1
        else : 
            continuous.append(index)
            index += 1
    encoder = LabelEncoder()
    df = df.copy()
    for col in df.columns :
        if df.dtypes[col] == object:
            df[col] = encoder.fit_transform(df[col].astype(str))
    standard = StandardScaler()
    standard_df = standard.fit_transform(df)
    minmax = MinMaxScaler()
    minmax_df = minmax.fit_transform(df)
    robust = RobustScaler()
    robust_df = robust.fit_transform(df)
    standard_df = pd.DataFrame(standard_df)
    minmax_df = pd.DataFrame(minmax_df)
    robust_df = pd.DataFrame(robust_df)
    std_corr = abs(standard_df.corr(method='pearson'))
    minmax_corr = abs(minmax_df.corr(method='pearson'))
    robust_corr = abs(robust_df.corr(method='pearson'))
    total_corr = (std_corr + minmax_corr +  robust_corr) / 3
    target_corr = total_corr.iloc[:, 0]
    target_corr = target_corr.drop(target_corr.index[[0]])
    # Store the top 9 score about categorical and continuous attribute
    cate_max = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_max = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    cate_idx = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    cont_idx = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in categorical :
        if (target_corr[i] > min(cate_max)) :
            min_idx = cate_max.index(min(cate_max))
            cate_max[min_idx] = target_corr[i]
            cate_idx[min_idx] = i   
    for i in continuous :
        if (target_corr[i] > min(cont_max)) :
            min_idx = cont_max.index(min(cont_max))
            cont_max[min_idx] = target_corr[i]
            cont_idx[min_idx] = i    
    return cate_idx, cont_idx

# ********************************* Preprocessing Function ********************************* #

# The number of total dataset is about 50000
# But the number of NAN value is maximum 900
# So, Drop the NAN value to accurate result
def dropData(df) :
    drop_df = df.dropna(axis=0)
    return drop_df

# Fill the NAN value with ffill method
def ffillData(df) :
    # Create the correlation table
    corr_matrix = df.corr(method = 'pearson').abs()
    # Find each column's max correlation value 
    for col in corr_matrix.columns :
        corr_list = corr_matrix[col]
        max_corr = 0
        for i in range(0, len(corr_list)) :
            if (corr_list[i] != 1.0) :
                if (corr_list[i] > max_corr) :
                    max_corr = corr_list[i]
                    max_index = i
        # Sort with the max corr columns and fill with method, ffill
        df = df.sort_values(by=corr_matrix.columns[max_index])
        df = df.fillna(method = 'ffill')
    return df

# Fill the NAN value with bfill method
def bfillData(df) :
    # Create the correlation table
    corr_matrix = df.corr(method = 'pearson').abs()
    # Find each column's max correlation value 
    for col in corr_matrix.columns :
        corr_list = corr_matrix[col]
        max_corr = 0
        for i in range(0, len(corr_list)) :
            if (corr_list[i] != 1.0) :
                if (corr_list[i] > max_corr) :
                    max_corr = corr_list[i]
                    max_index = i
        # Sort with the max corr columns and fill with method, bfill
        df = df.sort_values(by=corr_matrix.columns[max_index])
        df = df.fillna(method = 'bfill')  
    return df

# Create the LabelEncoder for the categorical attribute
def labelEncoding(df) :
    encoder = LabelEncoder()
    labeled_df = df.copy()
    for col in df.columns :
        if df.dtypes[col] == object:
            labeled_df[col] = encoder.fit_transform(df[col].astype(str))
    return labeled_df

# Standard Scaler
def standardScaler(df, column_list) :
    standard = StandardScaler()
    scaled_df = standard.fit_transform(df.iloc[ : , 1:56])
    scaled_df = pd.DataFrame(scaled_df)
    scaled_df.insert(0, 'Churn', df['Churn'])
    scaled_df.columns = column_list
    return scaled_df

# MinMax Scaler
def minmaxScaler(df, column_list) :
    minmax = MinMaxScaler()
    scaled_df = minmax.fit_transform(df.iloc[ : , 1:56])
    scaled_df = pd.DataFrame(scaled_df)
    scaled_df.insert(0, 'Churn', df['Churn'])
    scaled_df.columns = column_list
    return scaled_df

# Robust Scaler
def robustScaler(df, column_list) :
    robust = RobustScaler()
    scaled_df = robust.fit_transform(df.iloc[ : , 1:56])
    scaled_df = pd.DataFrame(scaled_df)
    scaled_df.insert(0, 'Churn', df['Churn'])
    scaled_df.columns = column_list
    return scaled_df

# Set the train_X, train_y, test_x, test_y
def setAttribute(df) :
    X = df.drop(columns = ['Churn'], axis = 1)
    y = df['Churn']
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

# ********************************* Model Function ********************************* #

# Build the model and find the best parameters
def buildModel(df, X_train, X_test, y_train, y_test) :
    df_list = ['Gaussian NB', 'Decision Tree', 'Logistic Regression']
    df_params = []
    df_score = []
    gaussian_best_param, gaussian_best_score = gaussianNB(X_train, X_test, y_train, y_test)
    df_params.append(gaussian_best_param)
    df_score.append(gaussian_best_score)
    decision_best_param, decision_best_score = decisionTree(X_train, X_test, y_train, y_test)
    df_params.append(decision_best_param)
    df_score.append(decision_best_score)
    logistic_best_param, logistic_best_score = logisticRegression(X_train, X_test, y_train, y_test)
    df_params.append(logistic_best_param)
    df_score.append(logistic_best_score)
    gnb_clf = GaussianNB(var_smoothing = gaussian_best_param['var_smoothing'])
    dt_clf = DecisionTreeClassifier(criterion = decision_best_param['criterion'], max_depth = decision_best_param['max_depth'])
    lost_reg = LogisticRegression(C = logistic_best_param['C'], max_iter = logistic_best_param['max_iter'])
    # Implement Voting Classifier using parameters with the highest score of each model
    soft_voting_model = VotingClassifier(estimators=[ ('GaussianNB', gnb_clf), ('DecisionTree', dt_clf), ('LogisticRegression', lost_reg)],
                                                        voting = 'soft', flatten_transform = False)
    soft_voting_model.fit(X_train, y_train)
    voting_pred = list(soft_voting_model.predict(X_test))
    voting_pred_proba = soft_voting_model.predict_proba(X_test)
    acc_score = round(accuracy_score(y_test, voting_pred), 2)    
    # # Visualize the value obtained using the Soft Voting model by dividing the section from 0% to 100%
    hard_result = voting_pred.count(1)
    total_count = voting_pred.count(0) + voting_pred.count(1)
    df['probability'] = pd.Series(voting_pred_proba[: , 0])
    bins = list(np.arange(0.0, 1.1, 0.1))
    for i in range(0, len(bins)) :
        bins[i] = round(bins[i] , 1)
    bins_label = [str(int(x*100)) + "%" for x in bins]
    df["level"] = pd.cut(df["probability"], bins, right = False, labels = bins_label [:-1])
    total = pd.value_counts(df["level"])
    total = total.sort_index()
    ax = total.plot(kind='bar', title = "Probability Customer Churn (Hard Case: {0} / {1})".format(hard_result, total_count), rot=0,
                    color=['slateblue', 'slateblue', 'slateblue', 'slateblue', 'slateblue', 
                           'slateblue', 'slateblue', 'slateblue', 'slateblue', 'red'])
    ax.set_xticks = bins_label
    for p in ax.patches: 
        left, _, width, height = p.get_bbox().bounds 
        ax.annotate("{0}".format(int(height)), (left+width/2, height*1.01), ha='center')
    plt.savefig('image/probability_churn_{}.png'.format(acc_score))
    plt.clf()
    max_index = df_score.index(max(df_score))
    best_model = df_list[max_index]
    best_params = df_params[max_index]
    best_score = df_score[max_index]
    average_score = sum(df_score) / len(df_score)
    # At this time, Voting classifier is not reflected
    print('* Total Best Model: {}'.format(best_model))
    print('* Total Best Param: {}'.format(best_params))
    print('* Total Best Score: {}\n'.format(round(best_score, 4)))
    print('* Dataset Average Score: {}\n'.format(round(average_score, 4)))

    return best_model, best_params, best_score, average_score

# Gaussian Naive Bayes Model
def gaussianNB(X_train, X_test, y_train, y_test) :
    classifier = GaussianNB()
    parameters = {'var_smoothing' : np.logspace(0,-9, num=100)}
    grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)
    best_parameter = grid_search.best_params_
    best_score = round(grid_search.best_score_, 4)
    print('Gaussian Best Parameter: {}'.format(best_parameter))
    print('Gaussian Best Score: {}\n'.format(best_score))
    return best_parameter, best_score

# Decision Tree Classifier
def decisionTree(X_train, X_test, y_train, y_test) :
    classifier = DecisionTreeClassifier()
    parameters = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 10, 15, 30]}
    grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)
    best_parameter = grid_search.best_params_
    best_score = round(grid_search.best_score_, 4)
    print('Decision Tree Best Parameter: {}'.format(best_parameter))
    print('Decision Tree Best Score: {}\n'.format(best_score))
    return best_parameter, best_score

# Logistic Regression
def logisticRegression(X_train, X_test, y_train, y_test) :
    classifier = LogisticRegression()
    parameters = {'C': [0.01, 0.1, 1.0, 10.0], 'max_iter': [1000, 10000, 100000]}
    grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy')
    grid_search.fit(X_train, y_train)
    best_parameter = grid_search.best_params_
    best_score = round(grid_search.best_score_, 4)
    print('Logisitic Regression Best Parameter: {}'.format(best_parameter))
    print('Logisitic Regression Best Score: {}\n'.format(best_score))
    return best_parameter, best_score

# ********************************* Visualization Function ********************************* #

# Draw a bar graph that the number of each column's Nan value
def nanCount(df) :
    col = df.columns.values
    val = df.isnull().sum()
    null_count = list(zip(col, val))  
    null_list = []
    label = []
    for i in range(0, len(null_count)) :
        if (null_count[i][1] != 0) :
            label.append(null_count[i][0])
            null_list.append(null_count[i][1])
    index = np.arange(len(label))
    plt.bar(index, null_list)
    plt.title('Total Missing Value Count', fontsize=18)
    plt.xlabel('Column', fontsize=14)
    plt.ylabel('Number of NAN', fontsize=14)
    plt.xticks(index, label, fontsize=10, rotation = 40)
    plt.savefig('image/nan_count_histogram.png')
    plt.clf()

# Visualize the count value of customer churn
def countChurn(df) :
    total = pd.value_counts(df['Churn'])
    print('<Target Value Count>')
    print('Yes: {}'.format(total[1]))
    print('No: {}\n'.format(total[0]))
    ax = total.plot(kind='bar', title="Churn Value Count", rot=0, color=['darkslateblue', 'salmon'])
    for p in ax.patches: 
        left, _, width, height = p.get_bbox().bounds 
        ax.annotate("{0}".format(int(height)), (left+width/2, height*1.01), ha='center')
    plt.savefig('image/churn_counts_bargraph.png')
    plt.clf()

# Visualize the percentage of customer churn
def ratioChurn(df) :
    total = pd.value_counts(df['Churn'])
    yes_ratio = total[1] / len(df)
    no_ratio = total[0] / len(df)
    print('<Target Value Percentage>')
    print('Yes: {}%'.format(round(yes_ratio*100, 2)))
    print('No: {}%\n'.format(round(no_ratio*100, 2)))
    labels = ['Yes', 'No']
    ratio = [yes_ratio, no_ratio]
    plt.title('Churn Value Percentage')
    plt.pie(ratio, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 45, colors = ['salmon', 'darkslateblue'])
    plt.savefig('image/churn_ratio_piechart.png')
    plt.clf()

# Visualize the Top 9 correlation attribute distribution  
def visualDistribution(df, column_list, top_cate, top_cont) :
    _, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20, 15))
    counter = 0
    top_cate_columns = []
    for i in top_cate :
        top_cate_columns.append(column_list[i])
    for top_cate_column in top_cate_columns:
        value_counts = df[top_cate_column].value_counts()
        trace_x = counter // 3
        trace_y = counter % 3
        x_pos = np.arange(0, len(value_counts))
        axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label=value_counts.index, alpha=0.8)
        axs[trace_x, trace_y].set_title(top_cate_column)
        for tick in axs[trace_x, trace_y].get_xticklabels():
            tick.set_rotation(45)
        counter += 1
    plt.tight_layout()
    plt.savefig('image/categorical_distribution.png')
    plt.clf()

    _, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(14, 10))
    counter = 0
    top_cont_columns = []
    for i in top_cont :
        top_cont_columns.append(column_list[i])
    for top_cont_column in top_cont_columns:
        trace_x = counter // 3
        trace_y = counter % 3
        data = df[top_cont_column]
        sns.distplot(data, ax=axs[trace_x, trace_y])
        axs[trace_x, trace_y].set_title(top_cont_column, fontsize=12)
        counter += 1
    plt.tight_layout()
    plt.savefig('image/continuous_distribution.png')
    plt.clf()

# For each dataset, draw confusion matrix with the highest score
def confusionMatrix(dataset, model, params, X_train, X_test, y_train, y_test) :
    if (model == 'Gaussian NB') :
        classifier = GaussianNB(var_smoothing = params['var_smoothing'])
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
    elif (model == 'Decision Tree') :
        classifier = DecisionTreeClassifier(criterion = params['criterion'], max_depth = params['max_depth'])
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
    elif (model == 'Logistic Regression') :
        classifier = LogisticRegression(C = params['C'], max_iter = params['max_iter'])
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
    else :
        print('ERROR:: Invalid Model Input')
    matrix = confusion_matrix(y_test, y_predict)
    sns.heatmap(matrix, annot=True, linewidth=0.7, linecolor='black', fmt='g', cmap="BuPu")
    plt.title('{0} Confusion Matrix ({1})'.format(model, dataset))
    plt.xlabel('Y predict')
    plt.ylabel('Y test')
    plt.savefig('image/{0}_{1}_Confusion_Matrix.png'.format(dataset, model))
    plt.clf()

# ROC Curve
def rocCurve(dataset, model, params, X_train, X_test, y_train, y_test) : 
    if (model == 'Gaussian NB') :
        classifier = GaussianNB(var_smoothing = params['var_smoothing'])
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        y_predict_proba = y_pred_proba[:, 1]
    elif (model == 'Decision Tree') :
        classifier = DecisionTreeClassifier(criterion = params['criterion'], max_depth = params['max_depth'])
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        y_predict_proba = y_pred_proba[:, 1]
    elif (model == 'Logistic Regression') :
        classifier = LogisticRegression(C = params['C'], max_iter = params['max_iter'])
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        y_predict_proba = y_pred_proba[:, 1]
    else :
        print('ERROR:: Invalid Model Input')
    acc_score = metrics.accuracy_score(y_test, y_predict)	
    rec_score = metrics.recall_score(y_test, y_predict)	
    pre_score = metrics.precision_score(y_test, y_predict)	
    f1s_score = metrics.f1_score(y_test, y_predict)
    print("* Accuracy: {}".format(round(acc_score, 4)))
    print("* Precision: {}".format(round(pre_score, 4)))
    print("* Recall: {}".format(round(rec_score, 4)))
    print("* F1 score: {}".format(round(f1s_score, 4)))
    fpr, tpr, _ = roc_curve(y_test, y_predict_proba)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, label = 'ANN')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('{0} ROC curve ({1})'.format(model, dataset)) 
    plt.savefig('image/{0}_{1}_ROC_Curve.png'.format(dataset, model))
    plt.clf()

# Main Function 
if __name__ == '__main__':

    # Load the dataset and delete the unusable attribute
    df = loadDataset()
    column_list = df.columns

    # Data Inspection
    top_cate, top_cont = inspectionCorr(df)
    visualDistribution(df, column_list, top_cate, top_cont)
    inspectionDataset(df)
    df = df.drop(columns = 'HandsetPrice')
    column_list = df.columns

    # Preprocessing (drop / ffill / bfill)
    drop_df = dropData(df) 
    ffill_df = ffillData(df) 
    bfill_df = bfillData(df)
    print(drop_df.info())
    print(ffill_df.info())
    print(bfill_df.info())

    # Label encode the dataset
    drop_df = labelEncoding(drop_df)
    ffill_df = labelEncoding(ffill_df)
    bfill_df = labelEncoding(bfill_df)

    # Scale the dataset (standard / minmax / robust)
    stand_drop_df = standardScaler(drop_df, column_list)
    minmax_drop_df = minmaxScaler(drop_df, column_list)
    robust_drop_df = robustScaler(drop_df, column_list)
    stand_drop_df = stand_drop_df.dropna(axis=0)
    minmax_drop_df = minmax_drop_df.dropna(axis=0)
    robust_drop_df = robust_drop_df.dropna(axis=0)
    stand_ffill_df = standardScaler(ffill_df, column_list)
    minmax_ffill_df = minmaxScaler(ffill_df, column_list)
    robust_ffill_df = robustScaler(ffill_df, column_list)
    stand_bfill_df = standardScaler(bfill_df, column_list)
    minmax_bfill_df = minmaxScaler(bfill_df, column_list)
    robust_bfill_df = robustScaler(bfill_df, column_list)

    # Set the attribute (X and y) and Set the dataframe (train and test)
    # ex) sd = standard / drop
    sd_X_train, sd_X_test, sd_y_train, sd_y_test = setAttribute(stand_drop_df)
    md_X_train, md_X_test, md_y_train, md_y_test = setAttribute(minmax_drop_df)
    rd_X_train, rd_X_test, rd_y_train, rd_y_test = setAttribute(robust_drop_df)
    sf_X_train, sf_X_test, sf_y_train, sf_y_test = setAttribute(stand_ffill_df)
    mf_X_train, mf_X_test, mf_y_train, mf_y_test = setAttribute(minmax_ffill_df)
    rf_X_train, rf_X_test, rf_y_train, rf_y_test = setAttribute(robust_ffill_df)
    sb_X_train, sb_X_test, sb_y_train, sb_y_test = setAttribute(stand_bfill_df)
    mb_X_train, mb_X_test, mb_y_train, mb_y_test = setAttribute(minmax_bfill_df)
    rb_X_train, rb_X_test, rb_y_train, rb_y_test = setAttribute(robust_bfill_df)

    # Build the model and find the best model
    print('\n======= Used Dataset : Drop / Standard Scaler =======\n')
    sd_model, sd_params, sd_score, sd_average = buildModel(stand_drop_df, sd_X_train, sd_X_test, sd_y_train, sd_y_test) 
    print('\n======= Used Dataset : Drop / MinMax Scaler =======\n')
    md_model, md_params, md_score, md_average = buildModel(minmax_drop_df, md_X_train, md_X_test, md_y_train, md_y_test)
    print('\n======= Used Dataset : Drop / Robuster Scaler =======\n')
    rd_model, rd_params, rd_score, rd_average = buildModel(robust_drop_df, rd_X_train, rd_X_test, rd_y_train, rd_y_test)
    print('\n======= Used Dataset : ffill / Standard Scaler =======\n')
    sf_model, sf_params, sf_score, sf_average = buildModel(stand_ffill_df, sf_X_train, sf_X_test, sf_y_train, sf_y_test)
    print('\n======= Used Dataset : ffill / MinMax Scaler =======\n')
    mf_model, mf_params, mf_score, mf_average = buildModel(minmax_ffill_df, mf_X_train, mf_X_test, mf_y_train, mf_y_test)
    print('\n======= Used Dataset : ffill / Robuster Scaler =======\n')
    rf_model, rf_params, rf_score, rf_average = buildModel(robust_ffill_df, rf_X_train, rf_X_test, rf_y_train, rf_y_test)
    print('\n======= Used Dataset : bfill / Standard Scaler =======\n')
    sb_model, sb_params, sb_score, sb_average = buildModel(stand_bfill_df, sb_X_train, sb_X_test, sb_y_train, sb_y_test)
    print('\n======= Used Dataset : bfill / MinMax Scaler =======\n')
    mb_model, mb_params, mb_score, mb_average = buildModel(minmax_bfill_df, mb_X_train, mb_X_test, mb_y_train, mb_y_test)
    print('\n======= Used Dataset : bfill / Robuster Scaler =======\n')
    rb_model, rb_params, rb_score, rb_average = buildModel(robust_bfill_df, rb_X_train, rb_X_test, rb_y_train, rb_y_test)
    
    # Confusion Matrix 
    dataset = 'Drop_Standard'
    confusionMatrix(dataset, sd_model, sd_params, sd_X_train, sd_X_test, sd_y_train, sd_y_test)
    dataset = 'Drop_MinMax'
    confusionMatrix(dataset, md_model, md_params, md_X_train, md_X_test, md_y_train, md_y_test)
    dataset = 'Drop_Robust'
    confusionMatrix(dataset, rd_model, rd_params, sd_X_train, sd_X_test, sd_y_train, sd_y_test)
    dataset = 'ffill_Standard'
    confusionMatrix(dataset, sf_model, sf_params, sf_X_train, sf_X_test, sf_y_train, sf_y_test)
    dataset = 'ffill_MinMax'
    confusionMatrix(dataset, mf_model, mf_params, mf_X_train, mf_X_test, mf_y_train, mf_y_test)
    dataset = 'ffill_Robust'
    confusionMatrix(dataset, rf_model, rf_params, sf_X_train, sf_X_test, sf_y_train, sf_y_test)
    dataset = 'bfill_Standard'
    confusionMatrix(dataset, sb_model, sb_params, sb_X_train, sb_X_test, sb_y_train, sb_y_test)
    dataset = 'bfill_MinMax'
    confusionMatrix(dataset, mb_model, mb_params, mb_X_train, mb_X_test, mb_y_train, mb_y_test)
    dataset = 'bfill_Robust'
    confusionMatrix(dataset, rb_model, rb_params, sb_X_train, sb_X_test, sb_y_train, sb_y_test)

    # ROC Curve
    print('\n======= Used Dataset : Drop / Standard Scaler =======\n')
    dataset = 'Drop_Standard'
    rocCurve(dataset, sd_model, sd_params, sd_X_train, sd_X_test, sd_y_train, sd_y_test)
    print('\n======= Used Dataset : Drop / MinMax Scaler =======\n')
    dataset = 'Drop_MinMax'
    rocCurve(dataset, md_model, md_params, md_X_train, md_X_test, md_y_train, md_y_test)
    print('\n======= Used Dataset : Drop / Robuster Scaler =======\n')
    dataset = 'Drop_Robust'
    rocCurve(dataset, rd_model, rd_params, sd_X_train, sd_X_test, sd_y_train, sd_y_test)
    print('\n======= Used Dataset : ffill / Standard Scaler =======\n')
    dataset = 'ffill_Standard'
    rocCurve(dataset, sf_model, sf_params, sf_X_train, sf_X_test, sf_y_train, sf_y_test)
    print('\n======= Used Dataset : ffill / MinMax Scaler =======\n')
    dataset = 'ffill_MinMax'
    rocCurve(dataset, mf_model, mf_params, mf_X_train, mf_X_test, mf_y_train, mf_y_test)
    print('\n======= Used Dataset : ffill / Robuster Scaler =======\n')
    dataset = 'ffill_Robust'
    rocCurve(dataset, rf_model, rf_params, sf_X_train, sf_X_test, sf_y_train, sf_y_test)
    print('\n======= Used Dataset : bfill / Standard Scaler =======\n')
    dataset = 'bfill_Standard'
    rocCurve(dataset, sb_model, sb_params, sb_X_train, sb_X_test, sb_y_train, sb_y_test)
    print('\n======= Used Dataset : bfill / MinMax Scaler =======\n')
    dataset = 'bfill_MinMax'
    rocCurve(dataset, mb_model, mb_params, mb_X_train, mb_X_test, mb_y_train, mb_y_test)
    print('\n======= Used Dataset : bfill / Robuster Scaler =======\n')
    dataset = 'bfill_Robust'
    rocCurve(dataset, rb_model, rb_params, sb_X_train, sb_X_test, sb_y_train, sb_y_test)

    # Find the TOP 1 model and score with their preprocessing method
    best_preprocessing_list = ['Drop and Standard Scaler', 'Drop and MinMax Scaler', 'Drop and Robust Scaler',
                               'ffill and Standard Scaler', 'ffill and MinMax Scaler', 'ffill and Robust Scaler',
                               'bfill and Standard Scaler', 'bfill and MinMax Scaler', 'bfill and Robust Scaler']
    best_model_list = [sd_model, md_model, rd_model, sf_model, mf_model, rf_model, sb_model, mb_model, rb_model]
    best_params_list = [sd_params, md_params, rd_params, sf_params, mf_params, rf_params, sb_params, mb_params, rb_params]
    best_score_list = [sd_score, md_score, rd_score, sf_score,mf_score, rf_score, sb_score, mb_score, rb_score ]
    average_score_list = [sd_average, md_average, rd_average, sf_average, mf_average, rf_average, sb_average, mb_average, rb_average]
    best_index = best_score_list.index(max(best_score_list))
    average_index = average_score_list.index(max(average_score_list))   

    print('\n======= Total Best Score =======\n')
    print('Preprocessing Method: {}'.format(best_model_list[best_index]))
    print('Parameters: {}'.format(best_params_list[best_index]))
    print('Score: {}'.format(round(best_score_list[best_index], 4)))
    print('\n======= Best Average Score =======\n')
    print('Best Preprocessing Method: {}'.format(best_preprocessing_list[average_index]))
    print('Score: {}'.format(round(average_score_list[average_index], 4)))
    