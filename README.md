# PycharmRep

#code for ML Streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App', layout='wide')

# ---------------------------------#
st.write(""" # The Machine Learning App """)

# ---------------------------------#
# Sidebar - Collects user input features into dataframe
#st.sidebar.header('Upload your CSV/XLSX data')

#uploaded_file = st.sidebar.file_uploader("Upload your input CSV/XLSX file", type=["csv","xlsx"])
#st.sidebar.markdown("""
#[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)""")

selectbox = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Gradient Boost"])
st.sidebar.write(f"You selected {selectbox}")

# ---------------------------------#
# Main panel
# Displays the dataset
st.subheader('Dataset')

# ---------------------------------#
# Model building

def file_download(a):
    csv = a.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href


def gradient_boost(df1):

    #Data pre-processing
    df1 = df1.head(100)
    df1['obligor_name'] = df1['obligor_name'].astype(str)
    df1['nomura_rating'] = df1['nomura_rating'].astype(str)

    def clean_text(df1, df1_column_name):

        # Converting all messages to lowercase
        df1[df1_column_name] = df1[df1_column_name].str.lower()

        # Replace 10 digit numbers (formats include paranthesis, spaces, no spaces, dashes)
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumber')

        # Replace numbers with 'numbr'
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'\d+(\.\d+)?', 'numbr')

        # Remove punctuation
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'[^\w\d\s]', ' ')

        # Replace whitespace between terms with a single space
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'\s+', ' ')

        # Remove leading and trailing whitespace
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'^\s+|\s+?$', '')

        # Remove stopwords
        stop_words = set(stopwords.words('english') + ["ltd", "llc", "inc", "co", "limited", "corp"])
        df1[df1_column_name] = df1[df1_column_name].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

    # Calling the class
    clean_text(df1, 'obligor_name')

    # Tokenizing the data using RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    df1['obligor_name'] = df1['obligor_name'].apply(lambda x: tokenizer.tokenize(x.lower()))

    # Lemmatizing and then Stemming with Snowball to get root words and further reducing characters
    stemmer = SnowballStemmer("english")

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    # Tokenize and Lemmatize
    def preprocess(text):
        result = []
        for token in text:
            if len(token) >= 3:
                result.append(lemmatize_stemming(token))
        return result

    #Processing review with above Function
    processed_long = []
    for doc in df1.obligor_name:
        processed_long.append(preprocess(doc))

    df1['new_obligor_name'] = processed_long
    df1['obligor_name'] = df1['new_obligor_name'].apply(lambda x: ' '.join(y for y in x))

    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'species'.
    df1['nomura_rating'] = label_encoder.fit_transform(df1['nomura_rating'])

    Y = df1['nomura_rating']

    #Converting the features into number vectors
    tf_vec = TfidfVectorizer()
    features = tf_vec.fit_transform(df1['obligor_name'])
    X = features

    st.write("""In this implementation, the *Gradient Boost()* function is used in this app for build a classification model using the **Gradient Boost** algorithm.""")
    st.header('Gradient Boost')
    st.sidebar.header('Set Parameters')
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.sidebar.subheader('Learning Parameters')
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10, 50), 50)
    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
    st.write('---')
    parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1, 3), 1)
    st.sidebar.number_input('Step size for max_features', 1)
    st.sidebar.write('---')
    parameter_min_samples_split = st.sidebar.slider(
        'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider(
        'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.sidebar.subheader('General Parameters')
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)

    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1] + parameter_n_estimators_step,
                                   parameter_n_estimators_step)
    max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1] + 1, 1)
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

    #X = df.iloc[:, :-1]  # Using all column except for the last column as X
    #Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    # X_train.shape, Y_train.shape
    # X_test.shape, Y_test.shape

    gb = GradientBoostingClassifier(n_estimators=parameter_n_estimators,
                                   random_state=parameter_random_state,
                                   max_features=parameter_max_features,
                                   min_samples_split=parameter_min_samples_split,
                                   min_samples_leaf=parameter_min_samples_leaf)

    grid = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5)
    grid.fit(X_train, Y_train)

    st.subheader('Model Performance')

    Y_pred_test = grid.predict(X_test)
    st.write('Accuracy score:')
    st.info(accuracy_score(Y_test, Y_pred_test))

    st.write('Confusion Matrix:')
    st.info(confusion_matrix(Y_test, Y_pred_test))

    st.write('Precision:')
    st.info(precision_score(Y_test, Y_pred_test, average='micro'))

    st.write('Recall:')
    st.info(recall_score(Y_test, Y_pred_test, average='weighted'))

    st.write('F1-score:')
    st.info(f1_score(Y_test, Y_pred_test, average='weighted'))

    st.write("The best parameters are %s with a score of %0.2f"
             % (grid.best_params_, grid.best_score_))

    # ActualVsPredicted
    #gb.fit(X_train,Y_train)
    #predicted = gb.predict(X)
    #newdf = pd.DataFrame({'Actual data': Y, 'Predicted data': predicted})
    #st.write(newdf.head(10))

    st.subheader('Model Parameters')
    st.write(grid.get_params())

    # -----Process grid data-----#
    grid_results = pd.concat(
        [pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["accuracy"])],
        axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features', 'n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'accuracy']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # -----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='n_estimators')
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='max_features')
        ))
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter tuning',
                      scene=dict(
                          xaxis_title='n_estimators',
                          yaxis_title='max_features',
                          zaxis_title='accuracy'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    # -----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x, y, z], axis=1)

    st.write(grid_results)


    st.subheader('Graphs')
    st.header('Histogram')
    fig2 = px.histogram(grid_results)
    st.plotly_chart(fig2)
    st.header('Actual Vs Predicted values')
    fig3 = px.line(grid_results)
    st.plotly_chart(fig3)

    #ActualVsPredicted graph
    #gb.fit(X_train, Y_train)
    #predicted = gb.predict(X)
    #fig4 = px.line(x=Y, y=predicted)
    #st.plotly_chart(fig4)


    st.markdown(file_download(grid_results), unsafe_allow_html=True)


def build_model(df1):

    #Data pre-processing
    df1 = df1.head(100)
    df1['obligor_name'] = df1['obligor_name'].astype(str)
    df1['nomura_rating'] = df1['nomura_rating'].astype(str)

    def clean_text(df1, df1_column_name):

        # Converting all messages to lowercase
        df1[df1_column_name] = df1[df1_column_name].str.lower()

        # Replace 10 digit numbers (formats include paranthesis, spaces, no spaces, dashes)
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumber')

        # Replace numbers with 'numbr'
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'\d+(\.\d+)?', 'numbr')

        # Remove punctuation
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'[^\w\d\s]', ' ')

        # Replace whitespace between terms with a single space
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'\s+', ' ')

        # Remove leading and trailing whitespace
        df1[df1_column_name] = df1[df1_column_name].str.replace(r'^\s+|\s+?$', '')

        # Remove stopwords
        stop_words = set(stopwords.words('english') + ["ltd", "llc", "inc", "co", "limited", "corp"])
        df1[df1_column_name] = df1[df1_column_name].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

    # Calling the class
    clean_text(df1, 'obligor_name')

    # Tokenizing the data using RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    df1['obligor_name'] = df1['obligor_name'].apply(lambda x: tokenizer.tokenize(x.lower()))

    # Lemmatizing and then Stemming with Snowball to get root words and further reducing characters
    stemmer = SnowballStemmer("english")

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    # Tokenize and Lemmatize
    def preprocess(text):
        result = []
        for token in text:
            if len(token) >= 3:
                result.append(lemmatize_stemming(token))
        return result

    #Processing review with above Function
    processed_long = []
    for doc in df1.obligor_name:
        processed_long.append(preprocess(doc))

    df1['new_obligor_name'] = processed_long
    df1['obligor_name'] = df1['new_obligor_name'].apply(lambda x: ' '.join(y for y in x))

    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'species'.
    df1['nomura_rating'] = label_encoder.fit_transform(df1['nomura_rating'])

    Y = df1['nomura_rating']

    #Converting the features into number vectors
    tf_vec = TfidfVectorizer()
    features = tf_vec.fit_transform(df1['obligor_name'])
    X = features


    st.write("""In this implementation, the *Random Forest()* function is used in this app for build a classification model using the **Random Forest** algorithm.""")
    st.header('Set Parameters')
    split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.sidebar.header('Learning Parameters')
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10, 50), 50)
    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
    st.sidebar.write('---')
    parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1, 3), 1)
    st.sidebar.number_input('Step size for max_features', 1)
    st.sidebar.write('---')
    parameter_min_samples_split = st.sidebar.slider(
        'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider(
        'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.sidebar.subheader('General Parameters')
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)',
                                                   options=[True, False])
    parameter_oob_score = st.sidebar.select_slider(
        'Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1] + parameter_n_estimators_step,
                                   parameter_n_estimators_step)
    max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1] + 1, 1)
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

    #X = df.iloc[:, :-1]  # Using all column except for the last column as X
    #Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown('A model is being built to predict the following **Y** variable:')
    st.info(Y.name)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
    # X_train.shape, Y_train.shape
    # X_test.shape, Y_test.shape

    rf = RandomForestClassifier(n_estimators=parameter_n_estimators,
                               random_state=parameter_random_state,
                               max_features=parameter_max_features,
                               min_samples_split=parameter_min_samples_split,
                               min_samples_leaf=parameter_min_samples_leaf,
                               bootstrap=parameter_bootstrap,
                               oob_score=parameter_oob_score,
                               n_jobs=parameter_n_jobs)

    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid.fit(X_train, Y_train)

    st.subheader('Model Performance')

    Y_pred_test = grid.predict(X_test)
    st.write('Accuracy score:')
    st.info(accuracy_score(Y_test, Y_pred_test))

    st.write('Confusion Matrix:')
    st.info(confusion_matrix(Y_test, Y_pred_test))

    st.write('Precision:')
    st.info(precision_score(Y_test, Y_pred_test, average='micro'))

    st.write('Recall:')
    st.info(recall_score(Y_test, Y_pred_test, average='weighted'))

    st.write('F1-score:')
    st.info(f1_score(Y_test, Y_pred_test, average='weighted'))

    st.write("The best parameters are %s with a score of %0.2f"
             % (grid.best_params_, grid.best_score_))

    #ActualVsPredicted
    #rf.fit(X_train, Y_train)
    #predicted=rf.predict(X)
    #newdf = pd.DataFrame({'Actual data': Y,'Predicted data': predicted})
    #st.write(newdf.head(10))

    st.subheader('Model Parameters')
    st.write(grid.get_params())

    # -----Process grid data-----#
    grid_results = pd.concat(
        [pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["accuracy"])],
        axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features', 'n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'accuracy']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # -----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='n_estimators')
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='max_features')
        ))
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter tuning',
                      scene=dict(
                          xaxis_title='n_estimators',
                          yaxis_title='max_features',
                          zaxis_title='accuracy'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    # -----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)

    df = pd.concat([x, y, z], axis=1)

    st.write(grid_results)
    st.subheader('Graphs')
    st.header('Histogram')
    fig2 = px.histogram(grid_results)
    st.plotly_chart(fig2)
    st.header('Actual Vs Predicted values')
    fig3 = px.line(grid_results)
    st.plotly_chart(fig3)

    #ActualVsPredicted graph
    #rf.fit(X_train, Y_train)
    #predicted=rf.predict(X)
    #fig4= px.line(x=Y, y=predicted)
    #st.plotly_chart(fig4)

    st.markdown(file_download(grid_results), unsafe_allow_html=True)

# ---------------------------------#
df= pd.read_csv(r'dataset.csv')
st.write(df.head(5))
if selectbox == 'Random Forest':
    build_model(df)
else:
    gradient_boost(df)
