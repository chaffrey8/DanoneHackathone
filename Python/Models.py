from joblib import dump,load
from numpy import round
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from .ReadFiles import generate_categories

BASE_DIR = Path(__file__).resolve().parent.parent

def build_category_models():
    if not Path.exists(BASE_DIR / 'data' / 'categories_dict.pkl'):
        generate_categories()
    categorias = load(BASE_DIR / 'data' / 'categories_dict.pkl')
    vectorizer = TfidfVectorizer(
        max_df=0.5,
        min_df=5,
        stop_words="english",
    )
    ckli = [x for x in categorias.keys()]
    vectorizer.fit(ckli)
    with BASE_DIR / 'models' / 'CategoryVectorizer.pkl' as file:
        dump(vectorizer,file)
    X_tfidf = vectorizer.transform(ckli)
    kmeans = KMeans(n_clusters=500,n_init='auto')
    kmeans.fit(X_tfidf)
    with BASE_DIR / 'models' / 'CategoryClustering.pkl' as file:
        dump(kmeans,file)
    pipeline = Pipeline([('vectorizer', vectorizer), ('clustering', kmeans)])
    with BASE_DIR / 'models' / 'CategoryClassifier.pkl' as file:
        dump(pipeline,file)

def CovariablesTransformer(df):
    categorical = ['Brand','SubBrand','nutrition_grade']
    t = [('bin',
            OneHotEncoder(drop='first',sparse=True,dtype=int,handle_unknown='ignore'),
            categorical)
        ]
    ct = ColumnTransformer(transformers=t,remainder='passthrough')
    ct.fit(df)
    with open(BASE_DIR / 'models' / 'CategoryTransformer.pkl','wb') as file:
        dump(ct,file)

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, round(y_pred))
    return 'f1_err', err

def ecoscore_dt(X,y):
    dtc = XGBClassifier(verbosity = 1,
            n_estimators=20,
                        disable_default_eval_metric=True,
            n_jobs=-1,
                        #Tree Booster
            booster='dart',
                        learning_rate = 0.1,
                        max_depth = 10,
                        subsample = .8,
                        colsample_bytree = 0.3,
                        alpha = 10,
                        predictor = 'cpu_predictor',
                        num_parallel_tree=32,
                        #Learning Task
                        objective ='binary:logistic',
                        eval_metric=['logloss','auc'],
                        use_label_encoder=False
                        )
    dtc.set_params(eval_metric=f1_score)
    dtc.fit(X,y)
    with open(BASE_DIR / 'models' / 'EcoScoreClassifier.pkl') as file:
            dump(dtc,file)
