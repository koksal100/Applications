import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, classification_report, precision_score, \
    recall_score, f1_score, mean_absolute_error
import xgboost as xgb
from tornado.options import options

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #2E2E2E;  /* Ana ekran için koyu gri renk */
    color: #800000;  /* Yazıların rengi koyu bordo */
    font-size: 18px;  /* Yazı fontunu büyüt */
}

[data-testid="stHeader"] {
    background-color: #1C1C1C;  /* Başlık kısmı için daha koyu renk */
}

[data-testid="stSidebar"] {
    background-color: #1C1C1C;  /* Kenar çubuğu için daha koyu renk */
    color: #800000;  /* Kenar çubuğundaki yazılar da koyu bordo */
    font-size: 20000px;  /* Kenar çubuğundaki yazı fontunu büyüt */
    font-weight: 600;
}

[data-testid="stToolbar"] {
    background-color: #1C1C1C;  /* Araç çubuğu için daha koyu renk */
}

h1, h2, h3, h4, h5, h6, p, div, span, label, input, textarea, select {
    color: #800000;  /* Tüm başlıklar, paragraflar, etiketler, giriş alanları vb. koyu bordo */
    font-size: 18px;  /* Tüm yazıların fontunu büyüt */
    font-weight: 600;  /* Yazıları biraz daha kalın yap */
}
</style>
"""

# CSS kodunu Streamlit sayfasına uygulayın
st.markdown(page_bg_img, unsafe_allow_html=True)



# Bilgi sağlayan fonksiyon
def get_column_info(col, df):
    if pd.api.types.is_numeric_dtype(df[col]):
        min_val = df[col].min()
        max_val = df[col].max()
        return f"Sayısal Sütun - Aralık: {min_val} - {max_val}"
    else:
        unique_values = df[col].nunique()
        return f"Kategorik Sütun - Farklı Değer Sayısı: {unique_values}"


# Eksik veri analizi
def missing_data_analysis(df):
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    return missing_data


# Özellik seçimi
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def feature_selection(df, target_column):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Kategorik sütunlar için eksik verileri mod ile doldur
    cat_cols = df.select_dtypes(include=[object]).columns
    for col in cat_cols:
        mode_value = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_value)

    # Hedef sütunu ve özellikleri ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Kategorik sütunları sayısal verilere dönüştür
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Hedef değişkeni sayısal veriye dönüştür
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    # En iyi özelliklerin seçilmesi
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)

    scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values(by='Score', ascending=False)

    return scores.reset_index()


# Grafiği PNG olarak kaydetme ve indirme fonksiyonu
def download_plot(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return st.download_button(
        label="Grafiği İndir",
        data=buf,
        file_name=f"{filename}.png",
        mime="image/png"
    )


# Sidebar'da CSV dosyasını yüklet
st.sidebar.header("CSV Dosyası Yükleme ve Sütun Seçimi")
uploaded_file = st.sidebar.file_uploader("Bir CSV dosyası yükleyin", type="csv")

if uploaded_file is not None:
    # CSV dosyasını oku
    df = pd.read_csv(uploaded_file)

    # Sütunları göster ve kategorik/ordinal olanları seçmelerini sağla
    columns = df.columns.tolist()

    # Target sütun seçimi
    target_column = st.sidebar.selectbox(
        "Target (Hedef) sütunu seçin:",
        columns,
        index=len(columns) - 1,  # Default olarak son sütunu seç
        format_func=lambda col: f"{col} ({get_column_info(col, df)})"
    )
    is_target_categorical = df[target_column].dtype == 'object'

    # Target sütun seçimi
    date_column = st.sidebar.selectbox(
        "Date (tarih) sütunu seçin (eğer varsa):",
        ["Seçilmedi"] + columns,
        index=0,  # Varsayılan olarak hiçbir şey seçilmesin
        format_func=lambda col: f"{col} ({get_column_info(col, df)})" if col != "Seçilmedi" else "Seçilmedi"
    )

    # Grafik türü seçimi
    plot_type_cat = st.sidebar.multiselect(
        "Kategorik sütunlar için grafik türlerini seçin:",
        ["Bar Plot", "Violin Plot", "Point Plot", "Strip Plot"]
    )

    num_columns_for_numeric_target=["Scatter Plot",
    "Line Plot",
    "Hexbin Plot",
    "Pair Plot",
    "Residual Plot",
    "Joint Plot"]

    plot_types_for_categoric_target = [
        "Box Plot",
        "Violin Plot",
        "Strip Plot",
        "Swarm Plot",
        "Point Plot",
        "Bar Plot"
    ]

    if not is_target_categorical:
        plot_type_num = st.sidebar.multiselect(
            "Sayısal sütunlar için grafik türlerini seçin:",
            num_columns_for_numeric_target
        )
    else:
        plot_type_num = st.sidebar.multiselect(
            "Sayısal sütunlar için grafik türlerini seçin:",
            plot_types_for_categoric_target
        )


    plot_type_bin=st.sidebar.multiselect(
        "Binary sütunlar için grafik türlerini seçin:",
        ["Box Plot","Violin Plot"]
    )

    # Geçerli renk paletlerini sunalım
    color_palette = st.sidebar.selectbox(
        "Renk paletini seçin:",
        ["deep", "muted", "pastel", "bright", "dark", "colorblind"]
    )

    # Plot özellikleri
    plot_size = st.sidebar.slider("Plot boyutunu seçin:", 5, 15, 10)
    plot_orientation = st.sidebar.radio("Plot yönünü seçin:", ["Dikey", "Yatay"])

    # "Başla/Tekrar Çalıştır" butonu
    if st.sidebar.button("Başla" if "plots_started" not in st.session_state else "Tekrar Çalıştır"):
        st.session_state.plots_started = True

    # EDA başlığı ve eksik veri analizi
    if "plots_started" in st.session_state:
        st.header("EDA (Exploratory Data Analysis)")
        st.write(df.head(10))

        # Sayfaya yönlendirme düğmeleri
        st.sidebar.markdown("<a href='#section1'>Eksik Veriler</a>", unsafe_allow_html=True)
        st.sidebar.markdown("<a href='#section2'>Korelasyon Matrisi</a>", unsafe_allow_html=True)
        st.sidebar.markdown("<a href='#section3'>Özellik Seçimi</a>", unsafe_allow_html=True)
        st.sidebar.markdown("<a href='#section4'>Kategorik Veriler</a>", unsafe_allow_html=True)
        st.sidebar.markdown("<a href='#section5'>Sayısal Veriler</a>", unsafe_allow_html=True)

        # Eksik verilerin sayısal gösterimi
        st.markdown("<a name='section1'></a>", unsafe_allow_html=True)
        st.write("## Eksik Veriler", unsafe_allow_html=True)
        missing_data = missing_data_analysis(df)
        if not missing_data.empty:
            st.write(missing_data)
            fig_missing, ax_missing = plt.subplots(figsize=(10, 4))
            missing_data.plot(kind="bar", ax=ax_missing)
            ax_missing.set_title("Eksik Veri Sayısı")
            ax_missing.set_xlabel("Sütunlar")
            ax_missing.set_ylabel("Eksik Veri Sayısı")
            st.pyplot(fig_missing)
            download_plot(fig_missing, "missing_data")

        # Korelasyon matrisini görselleştirme
        st.markdown("<a name='section2'></a>", unsafe_allow_html=True)
        st.subheader("Korelasyon Matrisi")
        corr_matrix = df.corr(numeric_only=True)
        fig_corr, ax_corr = plt.subplots(figsize=(15, 12))
        sns.heatmap(corr_matrix, fmt=".2f", cmap="coolwarm", ax=ax_corr)
        ax_corr.set_title("Korelasyon Matrisi")
        st.pyplot(fig_corr)
        download_plot(fig_corr, "correlation_matrix")

        # Özellik seçimi
        st.markdown("<a name='section3'></a>", unsafe_allow_html=True)
        st.subheader("Özellik Seçimi")
        scores = feature_selection(df, target_column)
        if not scores.empty:
            st.write("En iyi özelliklerin skorları:")
            st.write(scores)

            # Özelliklerin skorlarının grafiği
            fig_features, ax_features = plt.subplots(figsize=(15, 9))
            sns.barplot(data=scores, x='Feature', y='Score', ax=ax_features, palette=color_palette)
            ax_features.set_title("Özellik Skorları")
            ax_features.set_xlabel("Özellikler")
            ax_features.set_ylabel("Skor")
            ax_features.tick_params(axis='x', rotation=90)
            st.pyplot(fig_features)
            download_plot(fig_features, "feature_scores")

        # Kategorik ve sayısal sütunların ayrılması
        binary_columns=[col for col in columns if df[col].nunique()==2]
        categorical_columns = [col for col in columns if df[col].dtype == 'object' and not col in binary_columns]
        numerical_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col]) and not col in binary_columns]


        # Tarih ve hedef sütunları seçildiğinde grafikler oluşturulması
        if date_column and target_column and date_column != "Seçilmedi":
            # Tarih sütununu datetime formatına çevir
            df[date_column] = pd.to_datetime(df[date_column])


            # Zaman serisi grafiği
            if pd.api.types.is_numeric_dtype(df[target_column]):
                st.header("Zaman Serisi Grafiği")
                fig_time_series, ax_time_series = plt.subplots(figsize=(18, 12))
                sns.lineplot(data=df, x=date_column, y=target_column, ax=ax_time_series)
                ax_time_series.set_title(f"{date_column} ile {target_column} Zaman Serisi")
                ax_time_series.set_xlabel(date_column)
                ax_time_series.set_ylabel(target_column)
                st.pyplot(fig_time_series)
                download_plot(fig_time_series, "time_series_plot")

            if df[target_column].dtype == 'object':
                # Tarihe göre hedef sütununun dağılımı
                st.header("Tarihe Göre Hedef Dağılımı")
                fig_target_distribution, ax_target_distribution = plt.subplots(figsize=(18,12))
                sns.histplot(data=df, x=date_column, hue=target_column, multiple="stack", ax=ax_target_distribution)
                ax_target_distribution.set_title(f"{date_column} ve {target_column} Dağılımı")
                ax_target_distribution.set_xlabel(date_column)
                ax_target_distribution.set_ylabel("Frekans")
                st.pyplot(fig_target_distribution)
                download_plot(fig_target_distribution, "target_distribution_plot")


        st.markdown("<a name='section4'></a>", unsafe_allow_html=True)
        if binary_columns and plot_type_bin:
            st.write(f"Seçilen binary sütunlar için grafikler:{len(binary_columns)} tane")
            for col in binary_columns:
                if col == "id" or col == target_column:
                    continue
                for plot_type in plot_type_bin:
                    fig, ax = plt.subplots(figsize=(18,12))
                    if plot_type == "Box Plot":
                        sns.boxplot(data=df, x=col, y=target_column, palette=color_palette, ax=ax)
                    elif plot_type == "Violin Plot":
                        sns.violinplot(data=df, x=col, y=target_column, palette=color_palette, ax=ax)
                    ax.set_title(f"'{col}' Sütunu ve '{target_column}' Arasındaki İlişki")
                    st.pyplot(fig)
                    download_plot(fig, f"{col}_{plot_type}_plot")

        if categorical_columns and plot_type_cat:
            st.write("Seçilen kategorik sütunlar için grafikler ve value_counts:")
            for col in categorical_columns:
                if col in ["id", "date", target_column]:
                    continue

                # Kategorik sütundaki değer sayıları
                value_counts = df[col].value_counts()

                if len(value_counts) > 30:
                    # En çok bulunan 30 değeri seç
                    top_values = value_counts.head(30).index
                    df_filtered = df[df[col].isin(top_values)]
                    st.write(f"'{col}' sütununun en çok bulunan 30 değerinin sayıları:")
                    st.write(df_filtered[col].value_counts())
                else:
                    df_filtered = df
                    st.write(f"'{col}' sütununun değer sayıları:")
                    st.write(df[col].value_counts())

                for plot_type in plot_type_cat:
                    fig, ax = plt.subplots(figsize=(18, 15))
                    if plot_type == "Bar Plot":
                        sns.barplot(data=df_filtered, x=col, y=target_column, palette=color_palette, ax=ax)
                    elif plot_type == "Violin Plot":
                        sns.violinplot(data=df_filtered, x=col, y=target_column, palette=color_palette, ax=ax)
                    elif plot_type == "Point Plot":
                        sns.pointplot(data=df_filtered, x=col, y=target_column, palette=color_palette, ax=ax)
                    elif plot_type == "Strip Plot":
                        sns.stripplot(data=df_filtered, x=col, y=target_column, palette=color_palette, ax=ax)

                    ax.set_title(f"'{col}' Sütunu ve '{target_column}' Arasındaki İlişki")
                    if len(value_counts) > 20:
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

                    st.pyplot(fig)
                    download_plot(fig, f"{col}_{plot_type}_plot")
                    plt.clf()  # Plotları temizle

        # Sayısal sütunlar için grafik ve korelasyon analizi
        st.markdown("<a name='section5'></a>", unsafe_allow_html=True)
        import matplotlib.ticker as ticker

        if numerical_columns and plot_type_num:
            st.write("Seçilen sayısal sütunlar için grafikler:")
            for col in numerical_columns:
                for plot_type in plot_type_num:
                    fig, ax = plt.subplots(figsize=(plot_size, plot_size / 2))

                    if plot_type == "Scatter Plot":
                        sns.scatterplot(data=df, x=col, y='target_column', ax=ax)
                    elif plot_type == "Line Plot":
                        sns.lineplot(data=df, x=col, y='target_column', ax=ax)
                    elif plot_type == "Hexbin Plot":
                        sns.hexbin(data=df, x=col, y='target_column', ax=ax, cmap='Blues')
                    elif plot_type == "Pair Plot":
                        sns.pairplot(df, vars=[col, 'target_column'], ax=ax)
                    elif plot_type == "Residual Plot":
                        sns.residplot(data=df, x=col, y='target_column', ax=ax)
                    elif plot_type == "Joint Plot":
                        sns.jointplot(data=df, x=col, y='target_column', ax=ax)
                    elif plot_type == "Box Plot":
                        sns.boxplot(data=df, x=col, y='target_column', ax=ax)
                    elif plot_type == "Violin Plot":
                        sns.violinplot(data=df, x=col, y='target_column', ax=ax)
                    elif plot_type == "Strip Plot":
                        sns.stripplot(data=df, x=col, y='target_column', ax=ax)
                    elif plot_type == "Swarm Plot":
                        sns.swarmplot(data=df, x=col, y='target_column', ax=ax)
                    elif plot_type == "Point Plot":
                        sns.pointplot(data=df, x=col, y='target_column', ax=ax)
                    elif plot_type == "Bar Plot":
                        sns.barplot(data=df, x=col, y='target_column', ax=ax)

                    ax.set_title(f"'{col}' Sütunu ve 'target_column' Arasındaki İlişki")

                    st.pyplot(fig)
                    download_plot(fig, f"{col}_{plot_type}")

    # Sidebar
    st.sidebar.markdown("---")

    # Null değerlere sahip sütunların doldurulması
    st.sidebar.header("Null Değerleri Doldurma")
    null_columns = df.columns[df.isnull().any()]
    fill_methods = {}
    for col in null_columns:
        if df[col].dtype == 'object':
            fill_methods[col] = st.sidebar.selectbox(
                f"'{col}' sütunundaki null değerleri nasıl doldurmak istersiniz?",
                options=["Drop Column", "Mode", "Fill with Specific Value"],
                index=0,
                key=f"fill_methods_{col}"
            )
        else:
            fill_methods[col] = st.sidebar.selectbox(
                f"'{col}' sütunundaki null değerleri nasıl doldurmak istersiniz?",
                options=["Drop Column", "Mean", "Median", "Mode", "Fill with Specific Value"],
                index=0,
                key=f"fill_methods_{col}"
            )

    for col, method in fill_methods.items():
        if method == "Fill with Specific Value":
            specific_value = st.sidebar.text_input(
                f"'{col}' sütunu için doldurma değerini girin:",
                key=f"specific_value_{col}"
            )
            fill_methods[col] = specific_value

    # Kategorik sütunların dönüşüm seçenekleri
    st.sidebar.header("String Türündeki Sütunları Dönüştürme")
    object_columns = df.select_dtypes(include=['object']).columns
    encoding_options = {}
    for col in object_columns:
        if col==target_column:
            continue
        options = ["Drop", "Target Encoding", "One-Hot Encoding", "Label Encoding"]
        ##if df[target_column].dtype == 'object':
        ##    options=["Drop", "One-Hot Encoding", "Label Encoding"]
        encoding_options[col] = st.sidebar.selectbox(
            f"'{col}' sütunu için dönüşüm seçin:",
            options=options,
            index=0,
            key=f"encoding_options_{col}"
        )


    # Ölçekleme seçenekleri
    st.sidebar.header("Ölçekleme Seçenekleri")
    scale_options = st.sidebar.multiselect(
        "Scale",
        options=["Normalize", "Standardize"],
        key="scale_options"
    )

    # Boyut azaltma seçenekleri
    st.sidebar.header("Boyut Azaltma Seçenekleri")
    reduce_dim_options = st.sidebar.selectbox(
        "Reduce Dimension",
        options=["None", "PCA", "SelectKBest"],
        index=0,
        key="reduce_dim_options"
    )

    if reduce_dim_options in ["PCA", "SelectKBest"]:
        percentage = st.sidebar.slider(
            "Percentage of Dimensions",
            min_value=0,
            max_value=100,
            value=10,  # Varsayılan olarak %10
            key="percentage"
        )
    # Başlangıçta df_selected'i tanımla
    if 'df_selected' not in st.session_state:
        st.session_state.df_selected = None

    # Dönüşümleri Uygula butonu
    if st.sidebar.button("Dönüşümleri Uygula"):

        # Null değerleri doldurma
        for col, method in fill_methods.items():
            if method == "Drop Column":
                df = df.drop(columns=[col])
            elif method == "Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "Median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "Mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = method

        for col, option in encoding_options.items():
            if option == "Drop":
                df = df.drop(columns=[col])
            elif option == "Target Encoding":
                if is_target_categorical:
                    # Kategorik hedef sütun için target encoding (her kategoriyi sayısal temsil ile kodlama)
                    target_classes = df[target_column].unique()
                    # Her feature kategorisi için target oranlarını hesapla
                    encoding_map = {}
                    for feature_value in df[col].unique():
                        feature_df = df[df[col] == feature_value]
                        encoding_map[feature_value] = {
                            target_class: len(feature_df[feature_df[target_column] == target_class]) / len(feature_df)
                            for target_class in target_classes
                        }

                    # Encoding sonuçlarını dataframe'e ekle
                    for target_class in target_classes:
                        col_name = f"{col}_{target_class}_encoded"
                        df[col_name] = df[col].map(lambda x: encoding_map[x][target_class])

                else:
                    # Sürekli hedef sütun için target encoding
                    mean_target = df.groupby(col)[target_column].mean()
                    df[col] = df[col].map(mean_target)

                df.drop(columns=[col], axis=1, inplace=True)

            elif option == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=[col])
            elif option == "Label Encoding":
                df[col] = df[col].astype('category').cat.codes


        # Ölçeklendirme
        if "Normalize" in scale_options:
            scaler = MinMaxScaler()
            df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
                df.select_dtypes(include=[np.number]))

        if "Standardize" in scale_options:
            scaler = StandardScaler()
            df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
                df.select_dtypes(include=[np.number]))
        st.session_state.df_selected=df

        # Boyut azaltma
        dim_number = int(percentage * len(df.columns))
        if reduce_dim_options == "PCA":
            pca = PCA(n_components=dim_number)
            st.session_state.df_selected = pd.DataFrame(pca.fit_transform(df.select_dtypes(include=[np.number])))
        elif reduce_dim_options == "SelectKBest":
            selector = SelectKBest(score_func=f_classif, k=dim_number)
            X = df.drop(columns=[target_column, date_column])
            y = df[target_column]
            X_selected = selector.fit_transform(X, y)
            selected_columns = X.columns[selector.get_support(indices=True)]
            st.session_state.df_selected = pd.DataFrame(X_selected, columns=selected_columns)
            st.session_state.df_selected[target_column] = y
            st.session_state.df_selected[date_column] = df[date_column]

        st.success("Dönüşümler başarıyla uygulandı!")
        st.write(st.session_state.df_selected.head(5))


        model_type = None
        # Sütunun tipini kontrol et
        if pd.api.types.is_numeric_dtype(df[target_column]):
            model_type = 'Regression'
        else:
            model_type = 'Classification'


        def select_random_sample(df, size_in_mb=80):
            # 1 MB = 1e6 bytes
            bytes_per_row = df.memory_usage(deep=True).sum() / len(df)
            n_rows = int((size_in_mb * 1e6) / bytes_per_row)
            # DataFrame'in rastgele bir örneğini al
            df_sampled = df.sample(n=n_rows, random_state=42)
            return df_sampled


        # Veri setini küçült
        st.session_state.df_selected = select_random_sample(st.session_state.df_selected)

        # Model Eğitmeye Başla butonuna tıklanınca çalışacak kod
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.df_selected.drop(columns=[target_column, date_column], axis=1),
            st.session_state.df_selected[target_column], test_size=0.2, random_state=42)

        try:
            if model_type == "Regression":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Gradient Boosting Regressor": GradientBoostingRegressor(),
                    "Support Vector Regressor": SVR(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "ElasticNet": ElasticNet(),
                    "XGBoost Regressor": xgb.XGBRegressor()
                }

                param_grid = {
                    "Linear Regression": {},
                    "Random Forest Regressor": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
                    "Gradient Boosting Regressor": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
                    "Support Vector Regressor": {"C": [0.1, 1], "gamma": [0.001, 0.01]},
                    "K-Neighbors Regressor": {"n_neighbors": [3, 5, 7]},
                    "Decision Tree Regressor": {"max_depth": [None, 10, 20]},
                    "ElasticNet": {"alpha": [0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
                    "XGBoost Regressor": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1],
                                          "max_depth": [3, 5, 7]}
                }

                st.write("### Regression Model Results")
                results = []
                ensemble_predictions = np.zeros_like(y_test, dtype=float)

                # Modelleri çalıştır ve sonuçları listeye kaydet
                for name, model in models.items():
                    grid_search = GridSearchCV(model, param_grid[name], cv=5)
                    grid_search.fit(X_train, y_train)
                    y_pred = grid_search.predict(X_test)

                    # Sonuçları sözlük formatında kaydet
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    results.append({
                        "Model": name,
                        "Best Parameters": grid_search.best_params_,
                        "R^2 Score": r2,
                        "MSE": mse,
                        "MAE": mae
                    })

                    # Ensemble için tahminleri ağırlıklı olarak ekle (MAE'yi ters çevirerek kullan)
                    ensemble_predictions += (1 / mae) * y_pred

                # Ensemble sonucu normalize et
                ensemble_predictions /= sum(1 / result['MAE'] for result in results)
                ensemble_r2 = r2_score(y_test, ensemble_predictions)
                ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
                ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)

                # Sonuçları DataFrame'e dönüştür
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)

                st.write("### Ensemble Regression Model")
                st.write(f"Ensemble R^2 Score: {ensemble_r2:.3f}")
                st.write(f"Ensemble MSE: {ensemble_mse:.3f}")
                st.write(f"Ensemble MAE: {ensemble_mae:.3f}")

            elif model_type == "Classification":
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "Gradient Boosting Classifier": GradientBoostingClassifier(),
                    "Support Vector Classifier": SVC(),
                    "K-Neighbors Classifier": KNeighborsClassifier(),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                    "XGBoost Classifier": xgb.XGBClassifier()
                }

                param_grid = {
                    "Logistic Regression": {"C": [0.1, 1], "solver": ["liblinear"]},
                    "Random Forest Classifier": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
                    "Gradient Boosting Classifier": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
                    "Support Vector Classifier": {"C": [0.1, 1], "gamma": [0.001, 0.01]},
                    "K-Neighbors Classifier": {"n_neighbors": [3, 5, 7]},
                    "Decision Tree Classifier": {"max_depth": [None, 10, 20]},
                    "AdaBoost Classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                    "XGBoost Classifier": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1],
                                           "max_depth": [3, 5, 7]}
                }

                st.write("### Classification Model Results")
                ensemble_predictions = np.zeros_like(y_test, dtype=float)

                for name, model in models.items():
                    grid_search = GridSearchCV(model, param_grid[name], cv=5)
                    grid_search.fit(X_train, y_train)
                    y_pred = grid_search.predict(X_test)
                    st.write(f"**{name}**")
                    st.write(f"Best Parameters: {grid_search.best_params_}")
                    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
                    st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
                    st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.3f}")
                    st.write(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")


        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

