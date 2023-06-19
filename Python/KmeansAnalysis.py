from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from ReadFiles import *

df = read_data()
categorias = load(BASE_DIR / 'data' / 'categories_dict.pkl')
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)
ckli = [x for x in categorias.keys()]
vectorizer.fit(ckli)
X_tfidf = vectorizer.transform(ckli)

inertias = []
for i in range(1,2000):
    kmeans = KMeans(n_clusters=i,n_init='auto')
    kmeans.fit(X_tfidf)
    inertias.append(kmeans.inertia_)

if __name__ == '__main__':
    plt.plot(range(0,2000,500), inertias[::500], marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(BASE_DIR / 'plots' / 'kmeans.png')