import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
import statsmodels.api as sm
import statsmodels
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Matplotlib setup
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 8})

# Custom CSS for styling title and subheader
st.markdown("""
    <style>
        .dataframe {
            border: 2px solid blue;
            background-color: #2c2c2c;  /* Lighter black */
            color: #32CD32;  /* Bright green */
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            margin: 10px 0;
        }
        .dataframe td, .dataframe th {
            padding: 8px;
            text-align: left;
        }
        .dataframe th {
            background-color: #444;
        }
        h1 {
            text-align: center;
            font-size: 36px;
            background-image: linear-gradient(to right, #32CD32, #00BFFF);  /* Green to Sky Blue */
            -webkit-background-clip: text;
            color: transparent;
            text-decoration: underline;
        }
        h3 {
            color: #FF6347;  /* Light Red color */
        }
    </style>
""", unsafe_allow_html=True)


def render_dataframe(df):
    """Render a pandas dataframe as HTML with the custom styles."""
    return df.to_html(classes='dataframe', index=False)


# Global function to display content inside a styled box
def display_in_box2(content):
    """
    This function displays content inside a custom HTML box with a light yellow border and a transparent or no background.
    """
    html_content = f"""
    <div style="border: 2px solid #f1c232; padding: 10px; border-radius: 5px;">
        <p style="color: #333; font-weight: bold;">{content}</p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


# Define CSS styling for boxes
st.markdown("""
    <style>
        .data-box {
            border: 2px solid #FFA07A;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            background-color: #F5F5DC;
        }
        .metric-title {
            color: #4682B4;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .metric-value {
            color: #333;
            font-size: 14px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Data processing
# Load your dataset
df = pd.read_csv('crime data-set.csv')

# Define crime levels based on TOTAL IPC CRIMES values
min_val = df['TOTAL IPC CRIMES'].min()
max_val = df['TOTAL IPC CRIMES'].max()
range_val = (max_val - min_val) / 4
low = min_val + range_val
medium = low + range_val
high = medium + range_val

def get_crime_level(crime_count):
    if crime_count <= low:
        return 1
    elif crime_count <= medium:
        return 2
    elif crime_count <= high:
        return 3
    else:
        return 4

df['CRIME_LEVEL'] = df['TOTAL IPC CRIMES'].apply(get_crime_level)

# Encode categorical variables
le = LabelEncoder()
df["STATE/UT_encoded"] = le.fit_transform(df["STATE/UT"])
df["DISTRICT_encoded"] = le.fit_transform(df["DISTRICT"])

# Function to show linear regression results
def show_linear_regression():
    X = df.drop(['CRIME_LEVEL', 'STATE/UT', 'DISTRICT', 'TOTAL IPC CRIMES'], axis=1)
    y = df['TOTAL IPC CRIMES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_score = lr.score(X_test, y_test)

    # Display Linear Regression metrics
    st.subheader("Linear Regression Results")
    st.markdown("<div class='data-box'>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-title'>Linear Regression Score:</div> <div class='metric-value'>{lr_score:.4f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-title'>Mean Absolute Error:</div> <div class='metric-value'>{mean_absolute_error(lr_pred, y_test):.4f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-title'>Mean Squared Error:</div> <div class='metric-value'>{mean_squared_error(lr_pred, y_test):.4f}</div>", unsafe_allow_html=True)

    # R2 and Adjusted R2 calculation
    R2 = r2_score(lr_pred, y_test)
    adj_R2 = 1 - ((1 - R2) * (len(y) - 1) / (len(y) - X.shape[1] - 1))
    st.markdown(f"<div class='metric-title'>R2 Score:</div> <div class='metric-value'>{R2:.4f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-title'>Adjusted R2:</div> <div class='metric-value'>{adj_R2:.4f}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Function to show Breusch-Pagan test results
def show_breusch_pagan_test():
    X = df.drop(['CRIME_LEVEL', 'STATE/UT', 'DISTRICT', 'TOTAL IPC CRIMES'], axis=1)
    y = df['TOTAL IPC CRIMES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Add constant to exog
    X_train = sm.add_constant(X_train)
    
    # Fit model and perform Breusch-Pagan test
    results = sm.OLS(y_train, X_train).fit()
    bp = statsmodels.stats.diagnostic.het_breuschpagan(results.resid, results.model.exog)
    bp_test = pd.DataFrame(list(zip(['Lagrange Multiplier', 'p-value', 'f-value', 'f p-value'], bp)), columns=['Metric', 'Value'])
    
    # Display Breusch-Pagan test results in a styled box
    st.subheader("Breusch-Pagan Test Results")
    st.markdown("<div class='data-box'>", unsafe_allow_html=True)
    for index, row in bp_test.iterrows():
        st.markdown(f"<div class='metric-title'>{row['Metric']}:</div> <div class='metric-value'>{row['Value']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Function to show logistic regression results
def show_logistic_regression():
    X = df.drop(['CRIME_LEVEL', 'STATE/UT', 'DISTRICT'], axis=1)
    y = df['CRIME_LEVEL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log = LogisticRegression()
    log.fit(X_train, y_train)
    log_pred = log.predict(X_test)
    log_score = log.score(X_test, y_test)

    # Display Logistic Regression metrics
    st.subheader("Logistic Regression Results")
    st.markdown("<div class='data-box'>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-title'>Logistic Regression Score:</div> <div class='metric-value'>{log_score:.4f}</div>", unsafe_allow_html=True)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, log_pred)
    st.markdown("<div class='metric-title'>Confusion Matrix:</div>", unsafe_allow_html=True)
    st.write(conf_matrix)

    # Classification Report
    class_report = classification_report(y_test, log_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    st.markdown("<div class='metric-title'>Classification Report:</div>", unsafe_allow_html=True)
    st.write(class_report_df)
    st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("# ML model Implementation  🎈")

st.title("Different Machine Learning Models Implemented and their Outputs:-")

# Checkbox controls to display each section
st.subheader("--1. Analyse Linear Regression  Results")
if st.checkbox("Show Linear Regression Results"):
    show_linear_regression()

st.subheader("--2. Analyse Breusch-Pagan Test Results")
if st.checkbox("Show Breusch-Pagan Test Results"):
    show_breusch_pagan_test()

st.subheader("--3. Analyse Logistic Regression Results")
if st.checkbox("Show Logistic Regression Results"):
    show_logistic_regression()



# Support Vector Machine
def svm_model(X_train, X_test, y_train, y_test):
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_score = svm.score(X_test, y_test)
    st.write(f"Support Vector Machine Score: {svm_score:.4f}")
    st.write("Confusion Matrix:", confusion_matrix(y_test, svm_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, svm_pred))

# Decision Tree
def decision_tree_model(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_score = dt.score(X_test, y_test)
    st.write(f"Decision Tree Score: {dt_score:.4f}")
    st.write("Confusion Matrix:", confusion_matrix(y_test, dt_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, dt_pred))

# Random Forest
def random_forest_model(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    st.write(f"Random Forest Score: {rf_score:.4f}")
    st.write("Confusion Matrix:", confusion_matrix(y_test, rf_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, rf_pred))

# Naive Bayes
def naive_bayes_model(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_score = nb.score(X_test, y_test)
    st.write(f"Naive Bayes Score: {nb_score:.4f}")
    st.write("Confusion Matrix:", confusion_matrix(y_test, nb_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, nb_pred))

# K-means Clustering
def kmeans_clustering(df, X):
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    fig, ax = plt.subplots()
    ax.scatter(df['THEFT'], df['MURDER'], c=kmeans.labels_)
    ax.scatter(kmeans.cluster_centers_[:, 4], kmeans.cluster_centers_[:, 0], marker='*', c='red')
    ax.set_xlabel('Number of Thefts')
    ax.set_ylabel('Number of Murders')
    ax.set_title("K-means Clustering of Crimes")
    st.pyplot(fig)

# Hierarchical Clustering
def hierarchical_clustering(X):
    Z = linkage(X, method='ward')
    fig, ax = plt.subplots(figsize=(20, 8))
    dendrogram(Z, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Data point")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

# Agglomerative Clustering
def agglomerative_clustering(df, X):
    hier_clust = AgglomerativeClustering(n_clusters=3)
    labels = hier_clust.fit_predict(X)
    fig, ax = plt.subplots()
    ax.scatter(df['THEFT'], df['MURDER'], c=labels)
    ax.set_xlabel("Number of Thefts")
    ax.set_ylabel("Number of Murders")
    ax.set_title("Agglomerative Clustering of Crimes")
    st.pyplot(fig)

# DBSCAN Clustering
def dbscan_clustering(df, X):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X)
    fig, ax = plt.subplots()
    ax.scatter(df['THEFT'], df['MURDER'], c=dbscan.labels_)
    ax.set_xlabel("Number of Thefts")
    ax.set_ylabel("Number of Murders")
    ax.set_title("DBSCAN Clustering of Crimes")
    st.pyplot(fig)

# Main function to run the Streamlit app
def display_model_results(df):
    # Preprocess data
    X = df.drop(['CRIME_LEVEL', 'STATE/UT', 'DISTRICT', 'TOTAL IPC CRIMES'], axis=1)
    y = df['CRIME_LEVEL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Display model options with checkboxes
    st.subheader("--4. Analyse Support Vector Machine Results")
    if st.checkbox("Support Vector Machine"):
        svm_model(X_train, X_test, y_train, y_test)
        
    st.subheader("--5. Analyse Decision Tree Results")
    if st.checkbox("Decision Tree"):
        decision_tree_model(X_train, X_test, y_train, y_test)
        
    st.subheader("--6. Analyse Random Forest Results")
    if st.checkbox("Random Forest"):
        random_forest_model(X_train, X_test, y_train, y_test)
        
    st.subheader("--7. Analyse Naive Bayes Results")
    if st.checkbox("Naive Bayes"):
        naive_bayes_model(X_train, X_test, y_train, y_test)

    # Prepare data for clustering
    clustering_X = df[['MURDER', 'RAPE', 'ROBBERY', 'BURGLARY', 'THEFT', 'AUTO THEFT']]
    clustering_X = scaler.fit_transform(clustering_X)

    st.subheader("--8. Analyse K-means Clustering Results")
    if st.checkbox("K-means Clustering"):
        kmeans_clustering(df, clustering_X)
        
    st.subheader("--9. Analyse Hierarchical Clustering Results")
    if st.checkbox("Hierarchical Clustering"):
        hierarchical_clustering(clustering_X)
        
    st.subheader("--10. Analyse Agglomerative Clustering Results")
    if st.checkbox("Agglomerative Clustering"):
        agglomerative_clustering(df, clustering_X)
        
    st.subheader("--11. Analyse DBSCAN Clustering Results")
    if st.checkbox("DBSCAN Clustering"):
        dbscan_clustering(df, clustering_X)


display_model_results(df)
