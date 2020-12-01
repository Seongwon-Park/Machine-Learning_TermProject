import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, GridSearchCV

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage

filepath = './visualization/'


# Filtering dataset
def make_dataset():
    indicators = pd.read_csv('Indicators.csv')
    countries = pd.read_csv('Country.csv')

    # GDP
    indicator_gdp = indicators[
        (2010 <= indicators.Year) & (indicators.Year <= 2012) & (indicators.IndicatorCode == 'NY.GDP.PCAP.CD')]
    # Life expectancy
    indicator_life_expectancy = indicators[
        (2010 <= indicators.Year) & (indicators.Year <= 2012) & (indicators.IndicatorCode == 'SP.DYN.LE00.IN')]
    # Enrollment primary education
    indicator_pri_edu = indicators[
        (2010 <= indicators.Year) & (indicators.Year <= 2012) & (indicators.IndicatorCode == 'SE.PRM.ENRR')]
    # Enrollment secondary education
    indicator_sec_edu = indicators[
        (2010 <= indicators.Year) & (indicators.Year <= 2012) & (indicators.IndicatorCode == 'SE.SEC.ENRR')]
    # Enrollment tertiary education
    indicator_ter_edu = indicators[
        (2010 <= indicators.Year) & (indicators.Year <= 2012) & (indicators.IndicatorCode == 'SE.TER.ENRR')]
    # Renewable energy consumption
    indicator_renew_energy = indicators[
        (2010 <= indicators.Year) & (indicators.Year <= 2012) & (indicators.IndicatorCode == 'EG.FEC.RNEW.ZS')]
    # Fossil fuel energy consumption
    indicator_fossil_energy = indicators[
        (2010 <= indicators.Year) & (indicators.Year <= 2012) & (indicators.IndicatorCode == 'EG.USE.COMM.FO.ZS')]

    df = pd.DataFrame(
        columns=['GDP', 'Life expectancy', 'Enrollment primary education', 'Enrollment secondary education',
                 'Enrollment tertiary education', 'Renewable energy consumption', 'Fossil fuel energy consumption'])
    df = pd.concat([countries['CountryCode'], countries['ShortName'], df], axis=1)

    # Year
    year = pd.DataFrame({'Year': [2010, 2011, 2012]})
    new_year = pd.DataFrame(columns=['Year'])
    for i in range(len(df)):
        new_year = pd.concat([new_year, year])
    new_year.reset_index(inplace=True)
    new_year.drop(columns='index', inplace=True)

    df = pd.concat([df, df, df]).sort_values('CountryCode')
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)
    df = pd.concat([df, new_year], axis=1)

    for i in range(len(df)):
        country = df['CountryCode'].iloc[i]
        year = df['Year'].iloc[i]

        # GDP
        gdp = indicator_gdp[(indicator_gdp['CountryCode'] == country) & (indicator_gdp['Year'] == year)]['Value'].values
        if len(gdp) != 0:
            df['GDP'].iloc[i] = gdp[0]

        # Life expectancy
        life = indicator_life_expectancy[
            (indicator_life_expectancy['CountryCode'] == country) & (indicator_life_expectancy['Year'] == year)][
            'Value'].values
        if len(life) != 0:
            df['Life expectancy'].iloc[i] = life[0]

        # Enrollment primary education
        pri_edu = \
            indicator_pri_edu[(indicator_pri_edu['CountryCode'] == country) & (indicator_pri_edu['Year'] == year)][
                'Value'].values
        if len(pri_edu) != 0:
            df['Enrollment primary education'].iloc[i] = pri_edu[0]

        # Enrollment secondary education
        sec_edu = \
            indicator_sec_edu[(indicator_sec_edu['CountryCode'] == country) & (indicator_sec_edu['Year'] == year)][
                'Value'].values
        if len(sec_edu) != 0:
            df['Enrollment secondary education'].iloc[i] = sec_edu[0]

        # Enrollment tertiary education
        ter_edu = \
            indicator_ter_edu[(indicator_ter_edu['CountryCode'] == country) & (indicator_ter_edu['Year'] == year)][
                'Value'].values
        if len(ter_edu) != 0:
            df['Enrollment tertiary education'].iloc[i] = ter_edu[0]

        # Renewable energy consumption
        renew_energy = indicator_renew_energy[
            (indicator_renew_energy['CountryCode'] == country) & (indicator_renew_energy['Year'] == year)][
            'Value'].values
        if len(renew_energy) != 0:
            df['Renewable energy consumption'].iloc[i] = renew_energy[0]

        # Fossil fuel energy consumption
        fossil_energy = indicator_fossil_energy[
            (indicator_fossil_energy['CountryCode'] == country) & (indicator_fossil_energy['Year'] == year)][
            'Value'].values
        if len(fossil_energy) != 0:
            df['Fossil fuel energy consumption'].iloc[i] = fossil_energy[0]

    return df


# Numerical data statistics information
def statistical_info(df, title):
    df.describe().to_csv(title)

# Feature information and data types
def feature_info(df):
    print('feature names & data types')
    print(df.info())
    print('\n\n')


# Check correlation between columns
def correlation_between_columns(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data=df.corr(), annot=True, cmap='viridis')
    plt.savefig(filepath + 'inspection/heatmap.png')
    plt.clf()


# Standard Scaler
def standard_scaling(x):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    scaled_x = pd.DataFrame(sc.fit_transform(x))
    scaled_x.columns = x.columns.values
    scaled_x.index = x.index.values

    return scaled_x


# Minmax Scaler
def minmax_scaling(x):
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler()
    scaled_x = pd.DataFrame(sc.fit_transform(x))
    scaled_x.columns = x.columns.values
    scaled_x.index = x.index.values

    return scaled_x


# Normalizer
def normalize_scaling(x):
    from sklearn.preprocessing import Normalizer

    sc = Normalizer()
    scaled_x = pd.DataFrame(sc.fit_transform(x))
    scaled_x.columns = x.columns.values
    scaled_x.index = x.index.values

    return scaled_x


# Transform type of whole numerical data to float type
def type_convert(df):
    converted_df = df.copy()
    # 'Energy use'
    features = ['GDP', 'Life expectancy', 'Enrollment primary education',
                'Enrollment secondary education', 'Enrollment tertiary education',
                'Renewable energy consumption', 'Fossil fuel energy consumption', ]
    for feature in features:
        converted_df[feature] = converted_df[feature].astype('float')

    return converted_df


def cleaning_data(df):
    # If there are more than 4 missing values in a row, it is judged that it cannot be used and dropped.
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.dropna(axis=0, thresh=7)
    cleaned_df.reset_index(inplace=True, drop=True)

    '''GDP'''
    # If there is a missing value in GDP, it replaces the missing value with the national gdp average.
    cleaned_df['GDP'].fillna(cleaned_df.groupby('CountryCode')['GDP'].transform('mean'), inplace=True)

    '''Life expectancy'''
    # If there is a missing value in life expectancy, it replaces the missing value with the life expectancy average.
    cleaned_df['Life expectancy'].fillna(cleaned_df.groupby('CountryCode')['Life expectancy'].transform('mean'),
                                         inplace=True)

    '''Enrollment primary education'''
    # If there is a missing value in enrollment primary education,
    # it replaces the missing value with the enrollment primary education average.
    cleaned_df['Enrollment primary education'].fillna(
        cleaned_df.groupby('CountryCode')['Enrollment primary education'].transform('mean'), inplace=True)

    '''Enrollment secondary education'''
    # If there is a missing value in enrollment secondary education,
    # it replaces the missing value with the enrollment secondary education average.
    cleaned_df['Enrollment secondary education'].fillna(
        cleaned_df.groupby('CountryCode')['Enrollment secondary education'].transform('mean'), inplace=True)

    '''Enrollment tertiary education'''
    # If there is a missing value in enrollment tertiary education,
    # it replaces the missing value with the enrollment tertiary education average.
    cleaned_df['Enrollment tertiary education'].fillna(
        cleaned_df.groupby('CountryCode')['Enrollment tertiary education'].transform('mean'), inplace=True)

    '''Renewable energy consumption'''
    # If there is a missing value in renewable energy consumption,
    # it replaces the missing value with the renewable energy consumption average.
    cleaned_df['Renewable energy consumption'].fillna(
        cleaned_df.groupby('CountryCode')['Renewable energy consumption'].transform('mean'), inplace=True)

    '''Fossil fuel energy consumption'''
    # If there is a missing value in fossil fuel energy consumption,
    # it replaces the missing value with the Fossil fuel energy consumption average.
    cleaned_df['Fossil fuel energy consumption'].fillna(
        cleaned_df.groupby('CountryCode')['Fossil fuel energy consumption'].transform('mean'), inplace=True)

    gdp_idx = 2
    life_exp_idx = 3
    primary_idx = 4
    second_idx = 5
    tertiary_idx = 6
    renew_idx = 7
    fossil_idx = 8
    # energy_idx = 9
    drop_list = set()
    edu_data = list()
    for i in range(len(cleaned_df)):
        '''GDP'''
        # Drop the row if there is no gdp for every year in each country.
        if np.isnan(cleaned_df.iloc[i, gdp_idx]):
            drop_list.add(cleaned_df.index[i])

        '''Life expectancy'''
        # Drop the row if there is no life expectancy for every year in each country.
        if np.isnan(cleaned_df.iloc[i, life_exp_idx]):
            drop_list.add(cleaned_df.index[i])

        '''Education'''
        primary = 'Enrollment primary education'
        secondary = 'Enrollment secondary education'
        tertiary = 'Enrollment tertiary education'

        # If neither enrollment primary education, enrollment secondary education nor Enrollment tertiary education
        # is found in the sample, drop the sample.
        if np.isnan(cleaned_df.iloc[i, primary_idx]) and np.isnan(cleaned_df.iloc[i, second_idx]) and np.isnan(
                cleaned_df.iloc[i, tertiary_idx]):
            drop_list.add(cleaned_df.index[i])
        else:
            # If there are no primary education indicators and
            if np.isnan(cleaned_df.iloc[i, primary_idx]):
                # there are secondary education indicators,
                if not np.isnan(cleaned_df.iloc[i, second_idx]):
                    # If the secondary education indicator data for that row is greater than
                    # or equal to the overall secondary education indicator mean,
                    # replace with the average of data above average among all primary education indicators.
                    if cleaned_df.iloc[i, second_idx] >= cleaned_df[secondary].mean():
                        cleaned_df.iloc[i, primary_idx] = cleaned_df[cleaned_df[primary] >= cleaned_df[primary].mean()][
                            primary].mean()
                    # If the secondary education indicator data for that row is less than
                    # to the overall secondary education indicator mean,
                    # replace with the average of data below average among all primary education indicators.
                    else:
                        cleaned_df.iloc[i, primary_idx] = cleaned_df[cleaned_df[primary] < cleaned_df[primary].mean()][
                            primary].mean()
                # there are tertiary education indicators,
                elif not np.isnan(cleaned_df.iloc[i, tertiary_idx]):
                    # If the tertiary education indicator data for that row is greater than
                    # or equal to the overall tertiary education indicator mean,
                    # replace with the average of data above average among all primary education indicators.
                    if cleaned_df.iloc[i, tertiary_idx] >= cleaned_df[tertiary].mean():
                        cleaned_df.iloc[i, primary_idx] = cleaned_df[cleaned_df[primary] >= cleaned_df[primary].mean()][
                            primary].mean()
                    # If the tertiary education indicator data for that row is less than
                    # to the overall tertiary education indicator mean,
                    # replace with the average of data below average among all primary education indicators.
                    else:
                        cleaned_df.iloc[i, primary_idx] = cleaned_df[cleaned_df[primary] < cleaned_df[primary].mean()][
                            primary].mean()
            # If there are no secondary education indicators and
            if np.isnan(cleaned_df.iloc[i, second_idx]):
                # there are primary education indicators,
                if not np.isnan(cleaned_df.iloc[i, primary_idx]):
                    # If the primary education indicator data for that row is greater than
                    # or equal to the overall primary education indicator mean,
                    # replace with the average of data above average among all secondary education indicators.
                    if cleaned_df.iloc[i, primary_idx] >= cleaned_df[primary].mean():
                        cleaned_df.iloc[i, second_idx] = \
                            cleaned_df[cleaned_df[secondary] >= cleaned_df[secondary].mean()][
                                secondary].mean()
                    # If the primary education indicator data for that row is less than
                    # to the overall primary education indicator mean,
                    # replace with the average of data below average among all secondary education indicators.
                    else:
                        cleaned_df.iloc[i, second_idx] = \
                            cleaned_df[cleaned_df[secondary] < cleaned_df[secondary].mean()][
                                secondary].mean()
                # there are tertiary education indicators,
                elif not np.isnan(cleaned_df.iloc[i, tertiary_idx]):
                    # If the tertiary education indicator data for that row is greater than
                    # or equal to the overall tertiary education indicator mean,
                    # replace with the average of data above average among all secondary education indicators.
                    if cleaned_df.iloc[i, tertiary_idx] >= cleaned_df[tertiary].mean():
                        cleaned_df.iloc[i, second_idx] = \
                            cleaned_df[cleaned_df[secondary] >= cleaned_df[secondary].mean()][
                                secondary].mean()
                    # If the tertiary education indicator data for that row is less than
                    # to the overall tertiary education indicator mean,
                    # replace with the average of data above average among all secondary education indicators.
                    else:
                        cleaned_df.iloc[i, second_idx] = \
                            cleaned_df[cleaned_df[secondary] < cleaned_df[secondary].mean()][
                                secondary].mean()

            # If there are no tertiary education indicators and
            if np.isnan(cleaned_df.iloc[i, tertiary_idx]):
                # there are primary education indicators,
                if not np.isnan(cleaned_df.iloc[i, primary_idx]):
                    # If the primary education indicator data for that row is greater than
                    # or equal to the overall primary education indicator mean,
                    # replace with the average of data above average among all tertiary education indicators.
                    if cleaned_df.iloc[i, primary_idx] >= cleaned_df[primary].mean():
                        cleaned_df.iloc[i, tertiary_idx] = \
                            cleaned_df[cleaned_df[tertiary] >= cleaned_df[tertiary].mean()][
                                tertiary].mean()
                    # If the primary education indicator data for that row is less than
                    # to the overall primary education indicator mean,
                    # replace with the average of data above average among all tertiary education indicators.
                    else:
                        cleaned_df.iloc[i, tertiary_idx] = \
                            cleaned_df[cleaned_df[tertiary] < cleaned_df[tertiary].mean()][
                                tertiary].mean()
                # there are secondary education indicators,
                if not np.isnan(cleaned_df.iloc[i, second_idx]):
                    # If the secondary education indicator data for that row is greater than
                    # or equal to the overall secondary education indicator mean,
                    # replace with the average of data above average among all tertiary education indicators.
                    if cleaned_df.iloc[i, second_idx] >= cleaned_df[secondary].mean():
                        cleaned_df.iloc[i, tertiary_idx] = \
                            cleaned_df[cleaned_df[tertiary] >= cleaned_df[tertiary].mean()][
                                tertiary].mean()
                    # If the secondary education indicator data for that row is less than
                    # the overall secondary education indicator mean,
                    # replace with the average of data above average among all tertiary education indicators.
                    else:
                        cleaned_df.iloc[i, tertiary_idx] = \
                            cleaned_df[cleaned_df[tertiary] < cleaned_df[tertiary].mean()][
                                tertiary].mean()

        '''Renewable and fossil fuel energy consumption'''
        # If neither Renewable energy nor fossil fuel energy explanation is found in the sample, drop the sample.
        # If there is a value in the sample, either Renewable energy or Fossil Fuel energy explanation,
        # subtract the value from 100 and replace nan.
        if np.isnan(cleaned_df.iloc[i, renew_idx]) and np.isnan(cleaned_df.iloc[i, fossil_idx]):
            drop_list.add(cleaned_df.index[i])
        elif np.isnan(cleaned_df.iloc[i, renew_idx]):
            cleaned_df.iloc[i, renew_idx] = 100 - cleaned_df.iloc[i, fossil_idx]
        elif np.isnan(cleaned_df.iloc[i, fossil_idx]):
            cleaned_df.iloc[i, fossil_idx] = 100 - cleaned_df.iloc[i, renew_idx]

        '''Amount of renewable and fossil fuel energy consumption'''
        if not i in drop_list:
            edu_data.append(
                cleaned_df.iloc[i, primary_idx] + cleaned_df.iloc[i, second_idx] + cleaned_df.iloc[i, tertiary_idx])

    # Drop rows
    cleaned_df.drop(drop_list, inplace=True)
    cleaned_df.reset_index(inplace=True, drop=True)

    # Obtain the total school enrollment rate of primary, secondary and higher education.
    cleaned_df = pd.concat([cleaned_df, pd.DataFrame(data=edu_data, columns=['Education'])], axis=1)

    # Inspection
    print('============ After cleaning dataset ============')
    feature_info(cleaned_df)
    statistical_info(cleaned_df, 'after statistic.csv')
    verify_missing_value(cleaned_df, 'after cleaning')

    # columns visualization
    scatter(cleaned_df['GDP'], 'plot GDP')
    scatter(cleaned_df['Life expectancy'], 'plot Life expectancy')
    scatter(cleaned_df['Education'], 'plot Education')
    scatter(cleaned_df['Renewable energy consumption'], 'plot Renewable energy consumption')
    scatter(cleaned_df['Fossil fuel energy consumption'], 'plot Fossil fuel energy consumption')

    temp = cleaned_df.drop(
        columns=['CountryCode', 'ShortName', 'Year', 'Enrollment primary education', 'Enrollment secondary education',
                 'Enrollment tertiary education'])
    feature_distribution(temp)
    correlation_between_columns(temp)

    hdi_data = list()
    gdp_idx = 2
    life_exp_idx = 3
    edu_idx = 9
    for i in range(len(cleaned_df)):
        gdp = cleaned_df.iloc[i, gdp_idx]
        life_exp = cleaned_df.iloc[i, life_exp_idx]
        edu = cleaned_df.iloc[i, edu_idx]
        hdi_data.append((gdp + life_exp + edu) / 3)

    cluster_df = pd.concat([cleaned_df['ShortName'], cleaned_df['Year'], pd.DataFrame(data=hdi_data, columns=['HDI']),
                            cleaned_df['Renewable energy consumption'],
                            cleaned_df['Fossil fuel energy consumption']], axis=1)
    cluster_df.reset_index(inplace=True, drop=True)

    return cluster_df


def verify_missing_value(df, title):
    # Verifying missing values
    print('Missing value of the entire data set')
    print(df.isna().any())
    print('\n\n')

    print('Total missing value by column')
    print(df.isna().sum())
    print('\n\n')
    df.isna().sum().plot.bar(title='Total missing value by column', rot=45)
    plt.savefig(filepath + 'inspection/missing/' + title + '.jpg')
    plt.clf()

    # find percentage of missing values for each column
    missing_values = df.isnull().mean() * 100
    print('Missing value probability by column')
    print(missing_values)
    print('\n\n')

    # how many total missing values do we have?
    total_cells = np.product(df.shape)
    total_missing = df.isna().sum().sum()

    # percent of data that is missing
    total_missing_values = (total_missing / total_cells) * 100
    print('percent of data that is missing')
    print(total_missing_values)
    print('\n\n')


# Check outlier data
def check_outlier(df):
    features = df.columns
    sns.set_style("whitegrid")
    plt.figure(figsize=(24, 8))
    nonnumerical = ['Year', 'ShortName']
    for feature in features:
        if not feature in nonnumerical:
            sns.boxenplot(x=feature, orient='h', data=df)
            title = 'boxplot ' + feature
            plt.title(title)
            plt.savefig(filepath + 'inspection/boxplot/' + title + '.png')
            plt.clf()


def scatter(x, title):
    plt.figure(figsize=(12, 8))
    plt.plot(x)
    plt.title(title)
    plt.savefig(filepath + 'inspection/plot/' + title + '.png')
    # plt.show()


def feature_distribution(x):
    plt.figure(figsize=(12, 8))
    sns.set(style='whitegrid')
    sns.pairplot(x)
    plt.savefig(filepath + 'inspection/distribution/Pairplot for the Data' + '.png')
    plt.clf()


def hdi_distribution(x):
    sns.set(style='whitegrid')
    plt.scatter(x['HDI'], x['Renewable energy consumption'])
    plt.title('HDI - Renewable energy distribution', fontsize=16)
    plt.xlabel('HDI')
    plt.ylabel('Renewable energy consumption')
    plt.savefig(filepath + 'inspection/distribution/HDI - Renewable energy distribution.jpg')
    plt.clf()

    plt.scatter(x['HDI'], x['Fossil fuel energy consumption'])
    plt.title('HDI - Fossil fuel energy distribution', fontsize=16)
    plt.xlabel('HDI')
    plt.ylabel('Fossil fuel energy consumption')
    plt.savefig(filepath + 'inspection/distribution/HDI - Fossil fuel energy distribution.jpg')
    plt.clf()

    plt.scatter(x['HDI'], x['Renewable energy consumption'])
    plt.scatter(x['HDI'], x['Fossil fuel energy consumption'])
    plt.xlabel('HDI')
    plt.ylabel('Renewable & Fossil fuel energy consumption')
    plt.savefig(filepath + 'inspection/distribution/HDI - Energy distribution.jpg')
    plt.clf()


# Use K-means model to cluster dataset.
def k_means(df):
    from sklearn.cluster import KMeans

    # To tune hyper-parameter k, use the elbow method.
    wcss = []
    for i in range(3, 10):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km.fit(df)
        wcss.append(km.inertia_)

    sns.set_style("whitegrid")
    plt.plot(range(3, 10), wcss)
    plt.title('The Elbow Method', fontsize=16)
    plt.xlabel('No. of Clusters')
    plt.ylabel('wcss')
    plt.savefig(filepath + 'kmeans/k-means the elbow method.jpg')

    # Use K-means
    best_km = KMeans(n_clusters=4, max_iter=30, random_state=42, n_init=10, init='k-means++')
    labels = best_km.fit_predict(df)

    # Draw a scatter plot to visualize the clustering result.
    plt.figure(figsize=(12, 8))
    x = df.iloc[:, [0, 1]].values
    plt.scatter(x[labels == 0, 0], x[labels == 0, 1], s=10, c='pink', label='1')
    plt.scatter(x[labels == 1, 0], x[labels == 1, 1], s=10, c='yellow', label='2')
    plt.scatter(x[labels == 2, 0], x[labels == 2, 1], s=10, c='cyan', label='3')
    plt.scatter(x[labels == 3, 0], x[labels == 3, 1], s=10, c='magenta', label='4')
    plt.scatter(best_km.cluster_centers_[:, 0], best_km.cluster_centers_[:, 1], s=10, c='black', label='centeroid')

    plt.style.use('fivethirtyeight')
    title = 'K-Means Clustering with n_clusters 4'
    plt.title(title, fontsize=14)
    plt.xlabel('HDI')
    plt.ylabel('Renewable energy consumption')
    plt.legend()
    plt.grid()
    plt.savefig((filepath + 'kmeans/' + 'k-means.png'))
    plt.clf()

    n_clusters_ = len(set(labels))

    print('\n========== K-means ==========')
    if len(np.unique(labels)) != 1:
        print(title)
        print('Estimated number of clusters: %d' % n_clusters_)
        print("Silhouette Coefficient: %0.3f\n" % silhouette_score(df, labels))

    target = np.where(labels == 1)

    return target


# Use DBSCAN model to cluster dataset.
def dbscan(df):
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    # We can calculate the distance from each point to its closest neighbour using the NearestNeighbors.
    # The point itself is included in n_neighbors.
    # The k-neighbors method returns two arrays,
    # one which contains the distance to the closest n_neighbors points and the other
    # which contains the index for each of those points.

    # The optimal value for epsilon will be found at the point of maximum curvature.
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.title('Distance from each point to its closest neighbor', fontsize=12)
    plt.savefig(filepath + 'dbscan/distance.jpg')
    plt.clf()

    # hyper-parameter
    params = {
        'eps': [0.02, 0.03, 0.04, 0.05, 0.06],
        'min_samples': [10, 15, 20, 25, 30]
    }

    # Try various combinations of parameters
    # DBSCAN will output an array of -1’s and 0’s, where -1 indicates an outlier.
    print('\n========== DBSCAN ==========')
    plt.figure(figsize=(12, 8))
    for eps in params['eps']:
        for min_sample in params['min_samples']:
            dbs = DBSCAN(eps=eps, min_samples=min_sample).fit(df)
            labels = dbs.labels_

            plt.style.use('fivethirtyeight')
            sns.scatterplot(x=df['HDI'], y=df['Renewable energy consumption'], hue=labels,
                            palette=sns.color_palette('hls', np.unique(labels).shape[0]))
            title = 'DBSCAN with epsilon {}, min samples {}'.format(eps, min_sample)
            plt.title(title)
            plt.savefig(filepath + 'dbscan/' + title + '.jpg')
            plt.clf()

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            if len(np.unique(labels)) != 1:
                print(title)
                print('Estimated number of clusters: %d' % n_clusters_)
                print('Estimated number of noise points: %d' % n_noise_)
                print("Silhouette Coefficient: %0.3f\n" % silhouette_score(df, labels))

    dbs = DBSCAN(eps=0.03, min_samples=15).fit(df)
    labels = dbs.labels_

    target = np.where(labels == 1)

    return target


# Use gaussian mixture model to cluster dataset.
def em(df):
    from sklearn.mixture import GaussianMixture

    # hyper-parameter
    params = {
        'n_components': [2, 3, 4, 5, 6, 7],
        'max_iter': [50, 100, 200, 500]
    }

    em = GaussianMixture(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    em_gscv = GridSearchCV(em, params, cv=cv)
    em_gscv.fit(df)
    best_params = em_gscv.best_params_

    best_em = GaussianMixture(n_components=best_params['n_components'], max_iter=best_params['max_iter'],
                              random_state=42)
    labels = best_em.fit_predict(df)

    plt.style.use('fivethirtyeight')

    sns.scatterplot(x=df['HDI'], y=df['Renewable energy consumption'], hue=labels,
                    palette=sns.color_palette('hls', np.unique(labels).shape[0]))
    title = 'EM with n-components {}, max iteration {}'.format(best_params['n_components'], best_params['max_iter'])
    plt.title(title)
    plt.savefig(filepath + 'em/' + title + '.jpg')
    plt.clf()

    n_clusters_ = len(set(labels))

    print('\n========== EM ==========')
    if len(np.unique(labels)) != 1:
        print(title)
        print('Estimated number of clusters: %d' % n_clusters_)
        print("Silhouette Coefficient: %0.3f\n" % silhouette_score(df, labels))

    target = np.where(labels == 0)

    return target


# Check original dataset
df = make_dataset()
df.to_csv('original_dataset.csv')
df = type_convert(df)
print('============ Before cleaning dataset ============')
feature_info(df)
statistical_info(df, 'before statistic.csv')
verify_missing_value(df, 'before cleaning')

# Check cleaned dataset
cluster_df = cleaning_data(df)
print(cluster_df)
cluster_df.to_csv('cluster_dataset.csv')
hdi_distribution(cluster_df)
check_outlier(cluster_df)

# cluster_df = pd.read_csv('cluster_dataset.csv')
# cluster_df = cluster_df.drop(columns='Unnamed: 0')
country = cluster_df['ShortName']

# Scaling
train_df = cluster_df.drop(columns=['ShortName', 'Year'])
scaled_train_df = minmax_scaling(train_df)
# scaled_train_df = standard_scaling(train_df)
# scaled_train_df = normalize_scaling(train_df)

# Analysis algorithm
target_country = list()
target = k_means(scaled_train_df.drop(columns='Fossil fuel energy consumption'))
# for i in target:
#     target_country.append(country[i])
#     print(np.unique(target_country))

target = dbscan(scaled_train_df.drop(columns='Fossil fuel energy consumption'))

target = em(scaled_train_df.drop(columns='Fossil fuel energy consumption'))
for i in target:
    target_country.append(country[i])
    print(np.unique(target_country))
