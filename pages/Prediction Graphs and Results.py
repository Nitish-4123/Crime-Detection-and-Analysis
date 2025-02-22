import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib
import pandas as pd
import seaborn as sns
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# Load your dataset
# df = pd.read_csv('crime data-set.csv')

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

def time_series_analysis(crime_df):

    # crime_yearly = crime_df.resample('Y').sum()

    # Resample data to yearly frequency
    st.subheader("Analyze the Total IPC Crimes over Time and Decomposition:-")
    crime_yearly = crime_df.resample('Y').sum()

    # Plot the time series of Total IPC Crimes over Time
    st.subheader("---1. Analyze Time Series Plot:-")
    if st.checkbox("Show Time Series Plot of Total IPC Crimes"):
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(crime_yearly.index, crime_yearly['TOTAL IPC CRIMES'])
        ax.set_title('Total IPC Crimes over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Cases')
        st.pyplot(fig)

    # Decompose the time series into trend, seasonality, and residuals
    st.subheader("---2. Analyze the Decomposition of Time Series:-")
    if st.checkbox("Show Decomposition of Time Series"):
        decomposition = sm.tsa.seasonal_decompose(crime_yearly['TOTAL IPC CRIMES'], model='additive')

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 7))
        ax1.plot(crime_yearly.index, decomposition.observed)
        ax1.set_ylabel('Observed')
        ax2.plot(crime_yearly.index, decomposition.trend)
        ax2.set_ylabel('Trend')
        ax3.plot(crime_yearly.index, decomposition.seasonal)
        ax3.set_ylabel('Seasonal')
        ax4.plot(crime_yearly.index, decomposition.resid)
        ax4.set_ylabel('Residual')
        plt.tight_layout()
        st.pyplot(fig)

    # Augmented Dickey-Fuller test for stationarity
    st.subheader("---3. Analyze the ADF Test Results")
    if st.checkbox("Show ADF Test Results"):
        adf_result = sm.tsa.stattools.adfuller(crime_yearly['TOTAL IPC CRIMES'])
        result_text = f"""
        <p><strong>ADF Statistic:</strong> {adf_result[0]:.4f}</p>
        <p><strong>p-value:</strong> {adf_result[1]:.4f}</p>
        <p><strong>Critical Values:</strong> {adf_result[4]}</p>
        """
        if adf_result[1] < 0.05:
            result_text += f"\n\n**Hypothesis**: Reject null hypothesis - The time series is stationary."
        else:
            result_text += f"\n\n**Hypothesis**: Accept null hypothesis - The time series is not stationary."
        
        display_in_box2(result_text)

    # ACF plot for the Total IPC Crimes
    st.subheader("Analyze the Autocorrelation Function (ACF):")
    if st.checkbox("Show ACF Plot of Total IPC Crimes"):
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_acf(crime_yearly['TOTAL IPC CRIMES'], ax=ax)
        ax.set_title('Autocorrelation Function (ACF) of Total IPC Crimes')
        st.pyplot(fig)

    # PACF plot for the Total IPC Crimes
    st.subheader("Analyze the Partial Autocorrelation Function (PACF):")
    if st.checkbox("Show PACF Plot of Total IPC Crimes"):
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_pacf(crime_yearly['TOTAL IPC CRIMES'], lags=5, ax=ax)
        ax.set_title('Partial Autocorrelation Function (PACF) of Total IPC Crimes')
        ax.set_xlabel('Lags')
        ax.set_ylabel('Partial Autocorrelation')
        st.pyplot(fig)

    # Plot the actual values and the predicted values
    # st.subheader("Analyze the Variation Graph between Actual and Prediction values:")
    # column = st.selectbox("Select the column for Actual vs Predicted Comparison", crime_yearly.columns)
    # model = sm.tsa.ARIMA(crime_yearly['TOTAL IPC CRIMES'], order=(1,1,1))
    # results = model.fit()
    # # print(results.summary())
    
    # # Check if the user has selected a column
    # if column:
    #     if st.checkbox(f"Show Actual vs Predicted {column} Plot"):
    #         fig, ax = plt.subplots(figsize=(15, 8))
            
    #         # Plot actual vs predicted for the selected column
    #         ax.plot(crime_yearly.index, crime_yearly[column], label='Actual')
    #         ax.plot(crime_yearly.index, results.predict(start=0, end=len(crime_yearly)-1), label='Predicted')
    #         ax.set_title(f'Actual vs Predicted {column}')
    #         ax.set_xlabel('Year')
    #         ax.set_ylabel('Number of Cases')
    #         ax.legend()
    #         st.pyplot(fig)

    # # forecasted Allow user to select the column for comparison
    # st.subheader("Analyze the Variation Graph between Actual and Forecated values:")
    # column = st.selectbox("Select the column for Actual vs Forecasted Comparison", crime_yearly.columns)
    
    # # Check if the user has selected a column
    # if column:
    #     if st.checkbox(f"Show Actual vs Forecasted {column} Plot"):
    #         fig, ax = plt.subplots(figsize=(15, 8))
            
    #         # Generate forecast for the selected column
    #         forecast = results.predict(start=len(crime_yearly), end=len(crime_yearly)+4)
            
    #         # Plot actual vs forecasted for the selected column
    #         ax.plot(crime_yearly.index, crime_yearly[column], label='Actual')
    #         ax.plot(forecast.index, forecast, label='Forecast')
    #         ax.set_title(f'Actual vs Forecasted {column}')
    #         ax.set_xlabel('Year')
    #         ax.set_ylabel('Number of Cases')
    #         ax.legend()
    #         st.pyplot(fig)
    
    st.subheader("Analyze the Variation Graph between Actual and Prediction values:")
    column = st.selectbox("Select the column for Actual vs Predicted Comparison", crime_yearly.columns)

    # Check if the user has selected a column
    if column:
        if st.checkbox(f"Show Actual vs Predicted {column} Plot"):
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Fit ARIMA model on the selected column
            model = sm.tsa.ARIMA(crime_yearly[column], order=(1,1,1))  # Fit ARIMA on the selected column
            results = model.fit()  # Fit the model

            # Plot actual vs predicted for the selected column
            ax.plot(crime_yearly.index, crime_yearly[column], label='Actual')
            ax.plot(crime_yearly.index, results.predict(start=0, end=len(crime_yearly)-1), label='Predicted')
            ax.set_title(f'Actual vs Predicted {column}')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Cases')
            ax.legend()
            st.pyplot(fig)

    # Forecasted Allow user to select the column for comparison
    st.subheader("Analyze the Variation Graph between Actual and Forecasted values:")
    column = st.selectbox("Select the column for Actual vs Forecasted Comparison", crime_yearly.columns)

    # Check if the user has selected a column
    if column:
        if st.checkbox(f"Show Actual vs Forecasted {column} Plot"):
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Fit ARIMA model on the selected column
            model = sm.tsa.ARIMA(crime_yearly[column], order=(1,1,1))  # Fit ARIMA on the selected column
            results = model.fit()  # Fit the model
            
            # Generate forecast for the selected column
            forecast = results.predict(start=len(crime_yearly), end=len(crime_yearly)+4)
            
            # Plot actual vs forecasted for the selected column
            ax.plot(crime_yearly.index, crime_yearly[column], label='Actual')
            ax.plot(forecast.index, forecast, label='Forecast')
            ax.set_title(f'Actual vs Forecasted {column}')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Cases')
            ax.legend()
            st.pyplot(fig)


def display_in_box(content):
    # Display content inside the styled box with light yellow border
    st.markdown(
        f"""
        <div style="border: 2px solid #FFFF99; background-color: white; padding: 10px; border-radius: 5px;">
            <h4>Analysis Results</h4>
            <pre>{content}</pre>
        </div>
        """, 
        unsafe_allow_html=True
    )

def show():
    # Load the dataset
    crime_df = pd.read_csv('crime data-set.csv', parse_dates=['YEAR'], index_col=['YEAR'])
    st.sidebar.markdown("# Time Series Analysis  🎈")


    st.title("Crime Data Time Series Analysis")
    # st.subheader("Analyze the Total IPC Crimes over Time and Decomposition")

    # Perform Time Series Analysis
    time_series_analysis(crime_df)

if __name__ == "__main__":
    show()




# min_val = df['TOTAL IPC CRIMES'].min()
# max_val = df['TOTAL IPC CRIMES'].max()
# range_val = (max_val - min_val) / 4
# low = min_val + range_val
# medium = low + range_val
# high = medium + range_val
# def get_crime_level(crime_count):
#     if crime_count <= low:
#         return 1
#     elif crime_count <= medium:
#         return 2
#     elif crime_count <= high:
#         return 3
#     else:
#         return 4

# df['CRIME_LEVEL'] = df['TOTAL IPC CRIMES'].apply(get_crime_level)
# crime_level_count = df['CRIME_LEVEL'].value_counts()

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()

# # fit and transform the STATE/UT column using the LabelEncoder
# df["STATE/UT_encoded"] = le.fit_transform(df["STATE/UT"])

# # fit and transform the DISTRICT column using the LabelEncoder
# df["DISTRICT_encoded"] = le.fit_transform(df["DISTRICT"])
# grouped_state = df[["STATE/UT", "STATE/UT_encoded"]].groupby("STATE/UT").first()

# grouped_district = df[["DISTRICT", "DISTRICT_encoded"]].groupby("DISTRICT").first()

# # here linear regression starts
# X = df.drop(['CRIME_LEVEL','STATE/UT', 'DISTRICT','TOTAL IPC CRIMES'], axis=1)
# y = df['TOTAL IPC CRIMES']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
# lr = LinearRegression()
# lr.fit(X_train,y_train)
# lr_pred = lr.predict(X_test)
# lr_score = lr.score(X_test, y_test)
# print('Linear Regression score : ',lr_score)
# print()
# print('Mean_absolute_error  = ',metrics.mean_absolute_error(lr_pred,y_test))
# print()
# print('Mean_squared_error   = ',metrics.mean_squared_error(lr_pred,y_test))
# print()
# print('R2_score             = ',metrics.r2_score(lr_pred,y_test))
# print()
# R2 = metrics.r2_score(lr_pred,y_test)
# adj_R2 = 1-((1-R2)*(len(y)-1)/(len(y)-X.shape[1]-1))
# print('Adjusted_R2         = ',adj_R2)

# import statsmodels.api as sm
# results = sm.OLS(y_train,X_train).fit()
# print(results.summary())

# import statsmodels
# name = ['Lagrange multiplier statistic', 'p-value', 
#         'f-value', 'f p-value']

# bp = statsmodels.stats.diagnostic.het_breuschpagan(results.resid, results.model.exog)
# bp

# pd.DataFrame(name, bp)
# #ends

# # here logistic regression starts
# X = df.drop(['CRIME_LEVEL','STATE/UT', 'DISTRICT'], axis=1)
# y = df['CRIME_LEVEL']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# from sklearn.metrics import confusion_matrix, classification_report

# from sklearn.linear_model import LogisticRegression
# log = LogisticRegression()
# log.fit(X_train, y_train)
# log_pred = log.predict(X_test)
# log_score = log.score(X_test, y_test)
# log_prob = log.predict_proba(X_test)[:, 1]
# print('Logistic regression score : ',log_score)
# print()
# print(confusion_matrix(y_test, log_pred))
# print()
# print(classification_report(y_test, log_pred))
# # ends

# #svc starts here
# from sklearn.svm import SVC
# svm = SVC(probability=True)
# svm.fit(X_train, y_train)
# svm_pred = svm.predict(X_test)
# svm_prob = svm.predict_proba(X_test)[:, 1] if svm.probability else None
# svm_score = svm.score(X_test, y_test)
# print('Support Vector Machine score : ',svm_score)
# print()
# print(confusion_matrix(y_test, svm_pred))
# print()
# print(classification_report(y_test, svm_pred))


# # decision tree starts here
# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# dt_pred = dt.predict(X_test)
# dt_prob = dt.predict_proba(X_test)[:, 1]
# dt_score = dt.score(X_test, y_test)
# print('Decision tree score : ',dt_score)
# print()
# print(confusion_matrix(y_test, dt_pred))
# print()
# print(classification_report(y_test, dt_pred))

# #random forest starts here
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)
# rf_prob = rf.predict_proba(X_test)[:, 1]
# rf_score = rf.score(X_test, y_test)
# print('Random forest score : ',rf_score)
# print()
# print(confusion_matrix(y_test, rf_pred))
# print()
# print(classification_report(y_test, rf_pred))

# #naive bayes starts here
# from sklearn.naive_bayes import GaussianNB
# nb = GaussianNB()
# nb.fit(X_train, y_train)
# nb_pred = nb.predict(X_test)
# nb_prob = nb.predict_proba(X_test)[:, 1]
# nd_score = nb.score(X_test, y_test)
# print('Naive Bayes score : ',nd_score)
# print()
# print(confusion_matrix(y_test, nb_pred))
# print()
# print(classification_report(y_test, nb_pred))


# #clustering
# X = df.drop(['STATE/UT', 'DISTRICT','TOTAL IPC CRIMES','CRIME_LEVEL', 'STATE/UT_encoded', 'DISTRICT_encoded'], axis=1)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # k-means here
# from sklearn.cluster import KMeans

# X = df[['MURDER', 'RAPE', 'ROBBERY', 'BURGLARY', 'THEFT', 'AUTO THEFT']]
# X_scaled = scaler.fit_transform(X)

# # Perform k-means clustering
# kmeans = KMeans(n_clusters=4, random_state=42)
# kmeans.fit(X_scaled)

# # Visualize clusters and cluster centers
# plt.scatter(df['THEFT'], df['MURDER'], c=kmeans.labels_)
# plt.scatter(kmeans.cluster_centers_[:, 4], kmeans.cluster_centers_[:, 0], marker='*', c='red')
# plt.xlabel('Number of Thefts')
# plt.ylabel('Number of Murders')
# plt.show()

# # hierarchical clustering here
# from scipy.cluster.hierarchy import linkage, dendrogram
# Z = linkage(X_scaled, method='ward')

# # Plot dendrogram
# plt.figure(figsize=(20, 8))
# dendrogram(Z)
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Data point')
# plt.ylabel('Distance')
# plt.show()

# # agglomerative clustering here
# from sklearn.cluster import AgglomerativeClustering

# hier_clust = AgglomerativeClustering(n_clusters=3)
# labels = hier_clust.fit_predict(X_scaled)

# # Visualize clusters
# plt.scatter(crime['THEFT'], crime['MURDER'], c=labels)
# plt.xlabel('Number of Thefts')
# plt.ylabel('Number of Murders')
# plt.show()

# # DBSCAN here
# from sklearn.cluster import DBSCAN

# # Perform DBSCAN clustering
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# dbscan.fit(X_scaled)

# # Visualize clusters
# plt.scatter(crime['THEFT'], crime['MURDER'], c=dbscan.labels_)
# plt.xlabel('Number of Thefts')
# plt.ylabel('Number of Murders')
# plt.show()