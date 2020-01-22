import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jinja2
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

np.random.seed(123)

data = pd.read_csv('./dataset.csv')

def booleanTransform(labels):
    Labels_binary = labels.copy()
    for i in range(len(Labels_binary)):
        if Labels_binary[i] == 'NOTHING':
            Labels_binary[i] = 0
        else:
            Labels_binary[i] = 1
    Labels_binary = np.array(Labels_binary.astype(int))

    return Labels_binary

def k_means(n_clust, data_frame, true_labels):
    k_means = KMeans(n_clusters=n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print("\n")
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (k_means.inertia_,
             homogeneity_score(true_labels, y_clust),
             completeness_score(true_labels, y_clust),
             v_measure_score(true_labels, y_clust),
             adjusted_rand_score(true_labels, y_clust),
             adjusted_mutual_info_score(true_labels, y_clust),
             silhouette_score(data_frame, y_clust, metric='euclidean')))
    print("\n")

def transformedData(n_comp, dF):
    pca = PCA(n_components=n_comp, random_state=123)
    Data_reduced = pca.fit_transform(dF)
    print('\nРазмерность нового набора данных: ' + str(Data_reduced.shape))

    return Data_reduced

def pcaWorker(dF):
    # найдем оптимальный набор признаков
    pca = PCA(random_state=123)
    pca.fit(dF)
    features = range(pca.n_components_)

    plt.figure(figsize=(12, 8))
    plt.bar(features[:24], pca.explained_variance_[:24], color='lightskyblue')
    plt.xlabel('PCA feature')
    plt.ylabel('Variance')
    plt.xticks(features[:24])
    plt.show()

def dataWorker(dF):
    # нормализация
    scaler = StandardScaler()
    dataNorm = scaler.fit_transform(dF)

    # найдем оптимальное количество кластеров k
    ks = range(1, 10)
    inertias = []

    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(dataNorm)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.style.use('bmh')
    plt.plot(ks, inertias, '-o')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Inertia')
    plt.xticks(ks)
    plt.show()

    return dataNorm

if __name__ == '__main__':
    # немного анализа
    print('\nРазмерность набора данных: ' + str(data.shape))

    plt.figure(figsize=(18, 9))
    sns.countplot(data['ACTIVITY'], label="Count")
    plt.savefig("density.png")

    data.corr().style.background_gradient(cmap='coolwarm')
    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(data.corr(), annot=True)
    plt.savefig("correlation.png")

    Labels = data['ACTIVITY']
    dataFrame = data.drop(['ACTIVITY', 'LIGHT', 'PROXIMITY',
                           'LOCATION_Altitude-google', 'Satellites_in_range',
                           'LOCATION_ORIENTATION', 'Time', 'Date',
                           'LOCATION_Longitude', 'LOCATION_Latitude', 'LOCATION_Altitude'], axis=1)
    Labels_keys = Labels.unique().tolist()
    Labels = np.array(Labels)

    print('\nДействия из набора данных: ' + str(Labels_keys))

    Temp = pd.DataFrame(dataFrame.isnull().sum())
    Temp.columns = ['Sum']
    print('\nКоличество строк с пустыми ячейками: ' + str(len(Temp.index[Temp['Sum'] > 0])))

    dataFrame.info()
    dataFrame.describe()

    # посмотрим с точки зрения нашей задачи
    finalData = dataWorker(dF=dataFrame)
    
    # сама работа по кластеризации
    k_means(n_clust=2, data_frame=finalData, true_labels=Labels)

    k_means(n_clust=5, data_frame=finalData, true_labels=Labels)

    # поменяем наш подход в сторону булевого (в движении = 1, без движения = 0)
    Labels_binary = booleanTransform(labels=Labels)

    k_means(n_clust=2, data_frame=finalData, true_labels=Labels_binary)

    ## PCA
    pcaWorker(dF=finalData)
    k_means(n_clust=2, data_frame=transformedData(1, finalData), true_labels=Labels_binary)
    k_means(n_clust=2, data_frame=transformedData(2, finalData), true_labels=Labels_binary)



