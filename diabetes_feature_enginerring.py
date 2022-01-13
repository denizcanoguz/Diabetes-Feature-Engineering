import numpy as np
import pandas as pd
import seaborn as sns; sns.set_theme()
from _plotly_utils import png
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
# console-setup
import helpers.data_prep

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Veri Setini Getirelim.
def load_diabetes():
    data = pd.read_csv("datasets/diabetes.csv")
    return data

# Feature Engineering
# ----------------------------------------------------------------------------------------------------------------------
# İş Problemi
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
# edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
# geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
# gerçekleştirmeniz beklenmektedir.
# ----------------------------------------------------------------------------------------------------------------------
# Veri Seti Hikayesi
# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
# Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
# yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu,
# 0 ise negatif oluşunu belirtmektedir.
# diabetes_variables.png !!



# Proje Görevleri
# ----------------------------------------------------------------------------------------------------------------------

# Görev 1 : Keşifçi Veri Analizi
# ----------------------------------------------------------------------------------------------------------------------

# Adım 1: Genel resmi inceleyiniz.
# ----------------------------------------------------------------------------------------------------------------------
df_ = load_diabetes()
df = df_.copy()
df.columns = [col.upper() for col in df.columns]


def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)




# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# ----------------------------------------------------------------------------------------------------------------------

def grab_col_names(dataframe, cat_th=3, car_th=10):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
# ----------------------------------------------------------------------------------------------------------------------

# Numerik değişken analizi
num_cols
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# örnek
df["PREGNANCIES"].max()
# ['PREGNANCIES','GLUCOSE','BLOODPRESSURE','SKINTHICKNESS','INSULIN','BMI','DIABETESPEDIGREEFUNCTION','AGE']
df.groupby(['PREGNANCIES','AGE']).agg({"OUTCOME":"mean"}).sort_values("OUTCOME", ascending=False).head(20)



# Kategorik değişken analizi
cat_cols
# ['OUTCOME']

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
cat_summary(df,"OUTCOME")
df.isnull().sum().sum()
# 0

# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
# Hedef değişkenizmiz "OUTCOME"
# ----------------------------------------------------------------------------------------------------------------------
df.groupby("OUTCOME").agg({"PREGNANCIES":"mean","GLUCOSE":"mean",
                           "BLOODPRESSURE":"mean","SKINTHICKNESS":"mean",
                           "INSULIN":"mean","BMI":"mean",
                           "DIABETESPEDIGREEFUNCTION":"mean","AGE":"mean"}).sort_values("OUTCOME")


# Adım 5: Aykırı gözlem analizi yapınız.
# ----------------------------------------------------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
sns.boxplot(data=df, orient="h", palette="Set2")
plt.show()

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
for col in num_cols:
    print(col, outlier_thresholds(df, col))

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 6: Eksik gözlem analizi yapınız.
# ----------------------------------------------------------------------------------------------------------------------
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df)


# Adım 7: Korelasyon analizi yapınız.
# ----------------------------------------------------------------------------------------------------------------------
cor = df.corr(method='pearson')
cor
sns.heatmap(cor, annot=True)
plt.show()


# Görev 2 : Feature Engineering
# ----------------------------------------------------------------------------------------------------------------------
    
# Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız.
# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır.
# Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp
# sonrasında eksik değerlere işlemleri uygulayabilirsiniz.
# ----------------------------------------------------------------------------------------------------------------------

# 0 olan değerleri NaN atama işlemleri
df.loc[df["GLUCOSE"] == 0, "GLUCOSE"].count()
df.loc[df["INSULIN"] == 0, "INSULIN"].count()
df.loc[((df["GLUCOSE"] == 0) & (df["INSULIN"] == 0))]
df["GLUCOSE"].replace({0:np.nan},inplace=True)
df["BLOODPRESSURE"].replace({0:np.nan},inplace=True)
df["SKINTHICKNESS"].replace({0:np.nan},inplace=True)
df["INSULIN"].replace({0:np.nan},inplace=True)
df["BMI"].replace({0:np.nan},inplace=True)
df["AGE"].replace({0:np.nan},inplace=True)

# ısı haritası
msno.heatmap(df)
plt.show()

# eksik değerleri görelim
missing_values_table(df)
# missing_values_table(df)
#                n_miss  ratio
# INSULIN           374 48.700
# SKINTHICKNESS     227 29.560
# BLOODPRESSURE      35  4.560
# BMI                11  1.430
# GLUCOSE             5  0.650

def quick_missing_imp(data, num_method="median", cat_length=20, target="OUTCOME"):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)
missing_values_table(df)

# one hot encoding yapıyoruz çünkü tahmine dayalı bir doldurma işlemini yapabilmek için gerekiyor.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()


# değişkenlerin standartlatırılması
scaler = RobustScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=33)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()
# tahmine göre atama işlemlerini main df de görelim
df["INSULIN"] = dff[["INSULIN"]]
df["SKINTHICKNESS"] = dff[["SKINTHICKNESS"]]
df["BLOODPRESSURE"] = dff[["BLOODPRESSURE"]]
df["BMI"] = dff[["BMI"]]
df["GLUCOSE"] = dff[["GLUCOSE"]]

# eksik değerlerimizi tahmine göre atama işlemi ile doldurduk
missing_values_table(df)

# komşuluk ilkesine bağlı olarak aykırı değerleri bulmak ve silmek
clf = LocalOutlierFactor(n_neighbors=5)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:30]
# df_scores = -df_scores
np.sort(df_scores)[0:30]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[3]
df[df_scores < th]
df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
clf_index = df[df_scores < th].index
df.drop(index=clf_index, inplace=True)

# ----------------------------------------------------------------------------------------------------------------------

# Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    elif dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] < 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    else:
        print("Nothing")

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(col, grab_outliers(df,col))

# aykırı değerler için baskılama yöntemi kullanımı
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# aykırı değer varmıydı ?
for col in num_cols:
    print(col, check_outlier(df, col))


# baskılama yöntemi uygulayaklım
for col in num_cols:
    replace_with_thresholds(df, col)

# tekrar kontrol
for col in num_cols:
    print(col, check_outlier(df, col))




# Adım 2: Yeni değişkenler oluşturunuz.
# ----------------------------------------------------------------------------------------------------------------------

#df.AGE.min() # 21

# df[df["AGE"] < 22]
# df.loc[(df["AGE"] >= 8) & (df["AGE"] < 19), 'NEW_AGE_CAT'] = 'YOUNG-(8,19)'
# df.loc[(df["AGE"] >= 19) & (df["AGE"] < 30), 'NEW_AGE_CAT'] = 'ADULT-(19,30)'
# df.loc[(df["AGE"] >= 30) & (df["AGE"] < 46), 'NEW_AGE_CAT'] = 'MIDDLE-AGE-(30,46)'
# df.loc[(df["AGE"] >= 46) & (df["AGE"] < 60), 'NEW_AGE_CAT'] = 'OLD-(46,60)'
# df.loc[(df["AGE"] >= 60), 'NEW_AGE_CAT'] = 'SENIOR-(60+)'
# df.groupby("NEW_AGE_CAT")["OUTCOME"].mean()
# yaş değişkeni için sınıflar güzel oldu nice :D

df["INSULIN/AGE"]=df["INSULIN"]/df["AGE"]
df["BMI/AGE"]=df["BMI"]/df["AGE"]
df["PREGNANCIES/AGE"]=df["PREGNANCIES"]/df["AGE"]
df["INS*GLU"]=df["INSULIN"]* df["GLUCOSE"]
df.drop(["AGE"],axis = 1, inplace = True)

df['New_BMI'] = pd.cut(x = df['BMI'], bins = [0,18.5, 24.9, 29.9, 100], labels = ["Underweight",
                                                                                  "NormalWeight",
                                                                                   "Overweight",
                                                                                   "Obes"])
df['New_BloodPressure'] = pd.cut(x = df['BLOODPRESSURE'], bins = [0,80, 90, 120, 122], labels = ["Normal",
                                                                                                 "Hyper_St1",
                                                                                                 "Hyper_St2",
                                                                                                 "Hyper_Emer"])
df["New_Glucose"] = pd.cut(x = df["GLUCOSE"], bins = [0,140,200,300], labels = ["Normal",
                                                                                 "Prediabetes",
                                                                                 "Diabetes"])

# insülin değişkeni oluşturalım
def set_insulin(row):
     if row["INSULIN"] >= 100 and row["INSULIN"] <= 126:
         return "Normal"
     else:
         return "Abnormal"
df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))
# insülin değişkenin grafik ile incelenmesi
sns.countplot(data=df, x = 'NewInsulinScore', label='Count')

AB, NB = df['NewInsulinScore'].value_counts()
print('Number of patients Having Abnormal Insulin Levels: ',AB)
print('Number of patients Having Normal Insulin Levels: ',NB)

# BMI Vücut kitle endeksi için sınıflar
# df.loc[(df["BMI"] < 18.5), 'NEW_BMI_CAT'] = 'UNDER WEIGHT'
# df.loc[(df["BMI"] >= 18.5) & (df["BMI"] < 25), 'NEW_BMI_CAT'] = 'NORMALY WEIGHT'
# df.loc[(df["BMI"] >= 25) & (df["BMI"] < 30), 'NEW_BMI_CAT'] = 'OVER WEIGHT'
# df.loc[(df["BMI"] >= 30), 'NEW_BMI_CAT'] = 'OBESE'
# df.groupby("NEW_BMI_CAT")["OUTCOME"].mean()
#
# df.loc[(df["BLOODPRESSURE"] < 80), 'NEW_BP_CAT'] = 'OPTIMAL PRESSURE'
# df.loc[(df["BLOODPRESSURE"] >= 80) & (df["BLOODPRESSURE"] <= 84), 'NEW_BP_CAT'] = 'NORMAL PRESSURE'
# df.loc[(df["BLOODPRESSURE"] >= 85) & (df["BLOODPRESSURE"] < 90), 'NEW_BP_CAT'] = 'HIGH-NORMAL PRESSURE'
# df.loc[(df["BLOODPRESSURE"] >= 90), 'NEW_BP_CAT'] = 'HIGH PRESSURE'
# df.groupby("NEW_BP_CAT")["OUTCOME"].mean()

df.head()
df.shape



# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
# ----------------------------------------------------------------------------------------------------------------------
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# RARE  analizimizi yapalım
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "OUTCOME", cat_cols)

# One Hot encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() >= 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
# ----------------------------------------------------------------------------------------------------------------------
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()




# Adım 5: Model oluşturunuz.
# ----------------------------------------------------------------------------------------------------------------------
y = df["OUTCOME_1"]
X = df.drop(["OUTCOME_1"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test) # 0.8347


# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)