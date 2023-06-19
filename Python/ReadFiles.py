from joblib import load,dump
import re
from pathlib import Path
from pandas import DataFrame,read_json,concat
from unidecode import unidecode
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from numpy import argsort,nan,unique,where,zeros
from functools import reduce

BASE_DIR = Path(__file__).resolve().parent.parent

ps = PorterStemmer()

def read_data(train=True):
    name = f'{"train" if train else "test"}_products.json'
    with open(BASE_DIR / 'data' / name) as file:
        df = read_json(file,orient='index')
    return df

def stemming(sentence):
    uni = unidecode(sentence)
    low = uni.lower()
    non = re.sub('\W+',' ',low)
    san = ' '.join(non.split())
    ste = reduce(lambda x,y: x + ' ' + ps.stem(y),word_tokenize(san),'')
    return ste.strip()

def generate_ranks():
    df = read_data()
    brands = dict()
    for v in df.brand:
        for x in re.split(',',v):
            px = stemming(x)
            if brands.get(px):
                brands[px] += 1
            else:
                brands[px] = 1
    perc = DataFrame(brands.items(),columns=['Brand','Number'])
    perc.sort_values(by='Number',inplace=True,ignore_index=True,ascending=False)
    ranks = {row['Brand']:index for index,row in perc.iterrows()}
    with BASE_DIR / 'data' / 'ranks_dict.pkl' as file:
        dump(ranks,file)

def get_brands(brand):
    li = re.split(',',brand)
    pr = unique([stemming(x) for x in li])
    if not Path.exists(BASE_DIR / 'data' / 'ranks_dict.pkl'):
        generate_ranks()
    radi = load(BASE_DIR / 'data' / 'ranks_dict.pkl')
    ra = [radi.get(x) for x in pr]
    so = argsort(ra)
    br = pr[so[0]]
    if len(so) > 1:
        sb = pr[so[1]]
    else:
        sb = br
    return br,sb

def BrandCategorizer(brand):
    brands = ['carrefour','marqu reper','nestl','eroski','unilev','mondelez','kraft',
              'pepsico','lipton','knorr r','nestl r']
    return [1 if x == brand else 0 for x in brands]

def brands_matrix(df):
    new_df = DataFrame([[x for x in get_brands(row['brand'])] for _,row in df.iterrows()],columns=['Brand','SubBrand'])
    brands = ['carrefour','marqu reper','nestl','eroski','unilev','mondelez','kraft',
                'pepsico','lipton','knorr_r','nestl_r']
    dfbra = DataFrame([BrandCategorizer(brand) for brand in new_df['Brand']],columns=brands)
    return dfbra

def generate_categories():
    df = read_data()
    categorias = dict()
    idiomas = dict()
    for producto in df.categories_hierarchy:
        for elemento in producto:
            la,ca = re.split(':',elemento)
            la,ca = stemming(la),stemming(ca)
            if not idiomas.get(la):
                idiomas[la] = 1
            if categorias.get(ca):
                categorias[ca] += 1
            else:
                categorias[ca] = 1
    with BASE_DIR / 'data' / 'categories_dict.pkl' as file:
        dump(categorias,file)
    with BASE_DIR / 'data' / 'languages_list.pkl' as file:
        dump(sorted(unique([key for key in idiomas.keys()])),file)

def get_languages(categories):
    ladi = dict()
    for category in categories:
        ca = re.split(':',category)[0]
        if not ladi.get(ca):
            ladi[ca] = 1
    if not Path.exists(BASE_DIR / 'data' / 'languages_dict.pkl'):
        generate_categories()
    idiomas = load(BASE_DIR / 'data' / 'languages_list.pkl')
    return [1 if idioma in ladi.keys() else 0 for idioma in idiomas]

def language_matrix(df):
    if not Path.exists(BASE_DIR / 'data' / 'languages_list.pkl'):
        generate_categories()
    idiomas = load(BASE_DIR / 'data' / 'languages_list.pkl')
    df = DataFrame([[1 if idioma in [re.split(':',category)[0] for category in row['categories_hierarchy']] else 0 for idioma in idiomas] for _,row in df.iterrows()],columns=idiomas)
    return df

def get_hierarchy(categories):
    pipeline = load(BASE_DIR / 'models' / 'CategoryClassifier.pkl')
    arr = pipeline.predict([re.split(':',x)[1] for x in categories])
    res = []
    [res.append(x) for x in arr if x not in res]
    hierarchies = zeros(500,dtype=int)
    for i,v in enumerate(res):
        hierarchies += where([x == v for x in range(500)],i+1,0)
    hierarchies += where([x == 0 for x in hierarchies],30,0)
    return hierarchies

def hierarchy_matrix(df):
    categorias = [i for i in range(500)]
    df = DataFrame([get_hierarchy(row['categories_hierarchy']) for _,row in df.iterrows()],columns=categorias)
    return df

def countries_dict():
    codi = {'en:andorra-francia-espana':'other',
            'ar:espagne':'spain','en:espagne':'spain','en:spagna':'spain','en:spain':'spain',
            'es:espagne':'spain','es:espanha':'spain','fa:espagne':'spain',
            'fr:espanya':'spain','fr:spagna':'spain','fr:spanien':'spain',
            'en:deutschland':'germany','en:east-germany':'germany',
            'fr:deutschland':'germany','fr:niemcy':'germany','en:germany':'germany',
            'en:tyskland':'germany','fr:alemania':'germany',            
            'en:france':'france','en:francia':'france','en:frankreich':'france',
            'en:frankrike':'france','es:franca':'france','fr:franța':'france',
            'fr:francia':'france','fr:francja':'france','fr:frankreich':'france',
            'en:greece':'greece','fr:grecia':'greece',
            'en:nederland':'holland','en:nederlanderna':'holland','en:netherlands':'holland',
            'en:austria':'austria','en:autriche':'austria',
            'en:marruecos':'morocco','en:morocco':'morocco','fr:marruecos':'morocco',
            'en:poland':'poland','en:pologne':'poland','fr:polonia':'poland','fr:polska':'poland',
            'en:romania':'romania','en:roumanie':'romania',
            'en:schweiz':'switzerland','en:svizzera':'switzerland',
            'en:switzerland':'switzerland','fr:suiza':'switzerland',
            'en:sverige':'sweden','en:sweden':'sweden',
            'en:royaume-uni':'united-kingdom','en:united-kingdom':'united-kingdom',
            'en:etats-unis':'united-states','en:united-states':'united-states','fr:estados-unidos':'united-states',
            'fr:belgia':'belgium','fr:belgica':'belgium','fr:belgie':'belgium',
            'nl:belgie':'belgium',
            'en:guadeloupe':'guadeloupe','fr:guadalupe':'guadeloupe',
            'en:martinique':'martinique','fr:martinica':'martinique',
            'en:afghanistan':'afghanistan',
            'en:aland-islands':'aland',
            'en:albania':'albania',
            'en:algeria':'algeria',
            'en:andorra':'andorra',
            'en:angola':'angola',
            'en:argentina':'argentina',
            'en:australia':'australia',
            'en:bahrain':'bahrain',
            'en:bangladesh':'bangladesh',
            'en:belarus':'belarus',
            'en:belgica':'belgium',
            'en:belgien':'belgium',
            'en:belgique':'belgium',
            'en:belgium':'belgium',
            'en:bolivia':'bolivia',
            'en:bosnia-and-herzegovina':'bosnia-herzegovina',
            'en:brazil':'brazil',
            'en:bulgaria':'bulgaria',
            'en:burkina-faso':'burkina-faso',
            'en:cameroon':'cameroon',
            'en:canada':'canada',
            'en:chile':'chile',
            'en:china':'china',
            'en:colombia':'colombia',
            'en:costa-rica':'costa-rica',
            'en:cote-d-ivoire':'ivory-coast',
            'en:croatia':'croatia',
            'en:czech-republic':'czech-republic',
            'en:denmark':'denmark',            
            'en:dominican-republic':'dominican-republic',            
            'en:ecuador':'ecuador',
            'en:egypt':'egypt',
            'en:en':'england',
            'en:estonia':'estonia',
            'en:ethiopia':'ethiopia',
            'en:european-union':'european-union',
            'en:finland':'finland',
            'en:french-guiana':'french-guiana',
            'en:french-polynesia':'french-polynesia',
            'en:gabon':'gabon',
            'en:ghana':'ghana',
            'en:gibraltar':'gibraltar',
            'en:guatemala':'guatemala',
            'en:guinee':'guinea',
            'en:honduras':'honduras',
            'en:hong-kong':'hong-kong',
            'en:hungary':'hungary',
            'en:india':'india',
            'en:indonesia':'indonesia',
            'en:iraq':'iraq',
            'en:ireland':'ireland',
            'en:israel':'israel',
            'en:italy':'italy',
            'en:jamaica':'jamaica',
            'en:japan':'japan',
            'en:jersey':'jersey',
            'en:jordan':'jordan',
            'en:kazakhstan':'kazakhstan',
            'en:kuwait':'kuwait',
            'en:latvia':'latvia',
            'en:lebanon':'lebanon',
            'en:libya':'libya',
            'en:lithuania':'lithuania',
            'en:luxembourg':'luxembourg',
            'en:malaysia':'malaysia',
            'en:mali':'mali',
            'en:malta':'malta',
            'en:mauritius':'mauritius',
            'en:mexico':'mexico',
            'en:monaco':'monaco',
            'en:montenegro':'montenegro',
            'en:nepal':'nepal',            
            'en:new-caledonia':'new-caledonia',
            'en:new-zealand':'new-zealand',
            'en:nigeria':'nigeria',
            'en:norge':'norway','en:norway':'norway',
            'en:north-macedonia':'north-macedonia',
            'en:pakistan':'pakistan',
            'en:panama':'panama',
            'en:paraguay':'paraguay',
            'en:peru':'peru',
            'en:philippines':'philippines',
            'en:portugal':'portugal',
            'en:puerto-rico':'puerto-rico',
            'en:qatar':'qatar',
            'en:reunion':'reunion',            
            'en:russia':'russia',
            'en:saint-kitts-and-nevis':'saint-kitts-and-nevis',
            'en:saint-pierre-and-miquelon':'saint-pierre-and-miquelon',
            'en:saudi-arabia':'saudi-arabia',
            'en:senegal':'senegal',
            'en:serbia':'serbia',
            'en:singapore':'singapore',
            'en:sint-maarten':'sint-maarten',
            'en:slovakia':'slovakia',
            'en:slovenia':'slovenia',
            'en:somalia':'somalia',
            'en:south-africa':'south-africa',            
            'en:taiwan':'taiwan',
            'en:thailand':'thailand','fr:tailandia':'thailand',
            'en:togo':'togo',
            'en:trinidad-and-tobago':'trinidad-and-tobago',
            'en:tunisia':'tunisia',
            'en:turkey':'turkey',            
            'en:ukraine':'ukraine',
            'en:united-arab-emirates':'united-arab-emirates',
            'en:uruguay':'uruguay',
            'en:uzbekistan':'uzbekistan',
            'en:venezuela':'venezuela',
            'en:yugoslavia':'yugoslavia',
            'en:србија':'serbia',
            'fr:polinesia-francesa':'french-polynesia'
            }
    with BASE_DIR / 'data' / 'countries_dict.pkl' as file:
        dump(codi,file)

def selling_matrix(df):
    codi = load(BASE_DIR / 'data' / 'countries_dict.pkl')
    soco = unique(sorted(codi.values()))
    df = DataFrame([[1 if x in [codi.get(y,'other') for y in countries] else 0 for x in soco] for countries in df['selling_countries']],columns=soco)
    return df

def get_proportions(origins):
    codi = load(BASE_DIR / 'data' / 'countries_dict.pkl')
    soco = unique(sorted(codi.values()))
    proportions = zeros(len(soco),dtype=float)
    for key,value in origins.items():
        proportions[where(soco==codi.get(key,'other'))] += float(value)
    return proportions

def origins_matrix(df):
    codi = load(BASE_DIR / 'data' / 'countries_dict.pkl')
    soco = unique(sorted(codi.values()))
    df = DataFrame([get_proportions(origins) for origins in df.ingredient_origins],columns=soco)
    return df

def generate_materials():
    df = read_data()
    materials = set()
    _ = [[materials.add(x) for x in row['packaging_materials']] for _,row in df.iterrows()]
    lista = unique([stemming(re.split(':',material)[1]) for material in materials])
    with BASE_DIR / 'data' / 'materials_list.pkl' as file:
        dump(lista,file)
    
def materials_matrix(df):
    mali = load(BASE_DIR / 'data' / 'materials_list.pkl')
    df = DataFrame([[1 if x in [stemming(re.split(':',y)[1]) for y in materials] else 0 for x in mali] for materials in df.packaging_materials],columns=mali)
    return df

def nutrition_matrix(df):
    grades = ['a','b','c','d','e']
    df = DataFrame([[1 if x == row['nutrition_grade'] else 0 for x in grades] for _,row in df.iterrows()],columns=grades)
    return df

def covariables_matrix(df):
    dfcova = df[['is_beverage','additives_count','calcium_100g',
                 'carbohydrates_100g','energy_kcal_100g','fat_100g','fiber_100g',
                 'proteins_100g','salt_100g','sodium_100g','sugars_100g',
                 'non_recyclable_and_non_biodegradable_materials_count',
                 'est_co2_agriculture','est_co2_consumption','est_co2_distribution',
                 'est_co2_packaging','est_co2_processing','est_co2_transportation'
                 ]]
    for col in ['calcium_100g','carbohydrates_100g','energy_kcal_100g','fat_100g','fiber_100g',
                'proteins_100g','salt_100g','sodium_100g','sugars_100g']:
        dfcova[col] = dfcova[col].replace('[A-Za-z]*',0,regex=True)
        dfcova[col] = dfcova[col].astype('float')    
    dfcova['additives_count'].replace('unknown',-1,inplace=True)
    dfcova['additives_count'] = dfcova['additives_count'].astype('int')
    dfcova['additives_count'].replace(-1,nan,inplace=True)
    return dfcova

def build_Xy(df):
    dfran = brands_matrix(df)
    dfran.columns = [f'B_{col}' for col in dfran.columns]
    dflan = language_matrix(df)
    dflan.columns = [f'L_{col}' for col in dflan.columns]
    dfhie = hierarchy_matrix(df)
    dfhie.columns = [f'H_{col}' for col in dfhie.columns]
    dfsec = selling_matrix(df)
    dfsec.columns = [f'S_{col}' for col in dfsec.columns]
    dfopr = origins_matrix(df)
    dfopr.columns = [f'O_{col}' for col in dfopr.columns]
    dfpma = materials_matrix(df)
    dfpma.columns = [f'M_{col}' for col in dfpma.columns]
    dfnut = nutrition_matrix(df)
    dfnut.columns = [f'N_{col}' for col in dfnut.columns]
    dfcova = covariables_matrix(df)
    dfcova.columns = [f'C_{col}' for col in dfcova.columns]
    X = concat([dfran,dflan,dfhie,dfsec,dfopr,dfpma,dfnut,dfcova],axis=1)
    y = df[['ecoscore_grade']].astype(int)
    return X,y

