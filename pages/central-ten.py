
import matplotlib
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import io

# Load your dataset
df = pd.read_csv('crime data-set.csv')

# Matplotlib setup
matplotlib.use("Agg")
fig, ax = plt.subplots()
matplotlib.rcParams.update({"font.size": 8})

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

def central_tendencies(df):

    st.subheader("Analysing Mean for Data-Set :-")
    if st.checkbox("Mean Analysis"):
        mean_df = pd.DataFrame(df.mean(numeric_only=True), columns=["Mean"]).reset_index()
        mean_df.columns = ["Column Name", "Mean Value"]
        st.markdown(render_dataframe(mean_df), unsafe_allow_html=True)
        
    st.subheader("Analysing Median for Data-Set :-")
    if st.checkbox("Median Analysis"):
        median_df = pd.DataFrame(df.median(numeric_only=True), columns=["Median"]).reset_index()
        median_df.columns = ["Column Name", "Median Value"]
        st.markdown(render_dataframe(median_df), unsafe_allow_html=True)

    st.subheader("Analysing Mode for Data-Set :-")
    if st.checkbox("Mode Analysis"):
        mode_df = df.mode(numeric_only=True).transpose().reset_index()
        mode_df.columns = ["Column Name", "Mode Value"]
        st.markdown(render_dataframe(mode_df), unsafe_allow_html=True)

    st.subheader("Analysing Variance for Data-Set :-")
    if st.checkbox("Variance Analysis"):
        variance_df = df.var(numeric_only=True).transpose().reset_index()
        variance_df.columns = ["Column Name", "Variance Value"]
        st.markdown(render_dataframe(variance_df), unsafe_allow_html=True)

    st.subheader("Analysing Standard Deviation for Data-Set :-")
    if st.checkbox("STD Analysis"):
        std_df = df.std(numeric_only=True).transpose().reset_index()
        std_df.columns = ["Column Name", "STD Value"]
        st.markdown(render_dataframe(std_df), unsafe_allow_html=True)

    st.subheader("Analysing Ranges for Numeric Data-Set :-")
    if st.checkbox("Range Analysis"):
        numeric_data = df.select_dtypes(exclude='object')
        range_data = []
        for col in numeric_data.columns:
            range_value = df[col].max() - df[col].min()
            range_data.append([col, range_value])
        range_df = pd.DataFrame(range_data, columns=["Column Name", "Range Value"])
        st.markdown(render_dataframe(range_df), unsafe_allow_html=True)

    st.subheader("Analysing Inter-Quartile Range for Numeric Data-Set :-")
    if st.checkbox("IQR Analysis"):
        numeric_data = df.select_dtypes(exclude='object')
        iqr_data = []
        for col in numeric_data.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_data.append([col, IQR])
        iqr_df = pd.DataFrame(iqr_data, columns=["Column Name", "IQR Value"])
        st.markdown(render_dataframe(iqr_df), unsafe_allow_html=True)

    st.subheader("Analysing Percentiles for Numeric Data-Set :-")
    if st.checkbox("Percentile Analysis"):
        numeric_data = df.select_dtypes(exclude='object')
        percentiles_df = df.describe(percentiles=[.01, .25, .5, .75, .99]).transpose()[["1%", "25%", "50%", "75%", "99%"]]
        st.markdown(render_dataframe(percentiles_df), unsafe_allow_html=True)

    st.subheader("Analysing MAD and Skewness for Numeric Data-Set :-")
    if st.checkbox("MAD and Skewness Analysis"):
        numeric_data = df.select_dtypes(exclude='object')

        # MAD Calculation
        mad_df = numeric_data.apply(lambda x: (x - x.median()).abs().median()).reset_index()
        mad_df.columns = ["Column Name", "MAD Value"]

        # Skewness Calculation
        skew_df = numeric_data.skew().reset_index()
        skew_df.columns = ["Column Name", "Skewness Value"]

        st.subheader("Median Absolute Deviation (MAD)")
        st.markdown(render_dataframe(mad_df), unsafe_allow_html=True)

        st.subheader("Skewness")
        st.markdown(render_dataframe(skew_df), unsafe_allow_html=True)

def show():
    st.sidebar.markdown("# Central Tendencies 🎈")
    st.title("Different Measures for Central Tendencies and Variances")

    central_tendencies(df)

if __name__ == "__main__":
    show()






# import matplotlib
# import pandas as pd
# import seaborn as sns
# import streamlit as st
# import matplotlib.pyplot as plt
# import io

# df=pd.read_csv('crime data-set.csv')

# matplotlib.use("Agg")
# fig, ax = plt.subplots()
# matplotlib.rcParams.update({"font.size": 8})

# def central_tendencies(df):

#     st.subheader("Analysising Mean for Data-Set :-")
#     if st.checkbox("Mean Analysis"):
#         mean_df = pd.DataFrame(df.mean(numeric_only=True), columns=["Mean"]).reset_index()
#         mean_df.columns = ["Column Name", "Mean Value"]

#         st.dataframe(mean_df,width=800)
#     st.subheader("Analysising Medium for Data-Set :-")
#     if st.checkbox("Median Analysis"):
#         median_df = pd.DataFrame(df.median(numeric_only=True), columns=["Median"]).reset_index()
#         median_df.columns = ["Column Name", "Median Value"]

#         st.dataframe(median_df,width=800)

#     st.subheader("Analysising Mode for Data-Set :-")
#     if st.checkbox("Mode Analysis"):
#         # mode_df = pd.DataFrame(df.mode(numeric_only=True), columns=["Mode"]).reset_index()
#         mode_df = df.mode(numeric_only=True).transpose().reset_index()

#         mode_df.columns = ["Column Name", "Mode Value"]

#         st.dataframe(mode_df,width=800)

#     st.subheader("Analysising Variance for Data-Set :-")
#     if st.checkbox("Variance Analysis"):
#         variance_df = df.var(numeric_only=True).transpose().reset_index()

#         variance_df.columns = ["Column Name", "Variance Value"]

#         st.dataframe(variance_df,width=800)

#     st.subheader("Analysising Standard Deviation for Data-Set :-")
#     if st.checkbox("STD Analysis"):
#         std_df = df.std(numeric_only=True).transpose().reset_index()

#         std_df.columns = ["Column Name", "STD Value"]

#         st.dataframe(std_df,width=800)

#     st.subheader("Analysising Ranges for Numeric Data-Set :-")
#     if st.checkbox("Range Analysis"):
#         numeric_data = df.select_dtypes(exclude='object')

#         range_data=[]
#         for col in numeric_data.columns:
#             range = df[col].max() - df[col].min()
#             range_data.append([col, range])
#             print('range of %s : %d'%(col,range))
#         range_df = pd.DataFrame(range_data, columns=["Column Name", "Range Value"])

#         st.dataframe(range_df,width=800)

#     st.subheader("Analysing Inter-Quartile Range for Numeric Data-Set :-")
#     if st.checkbox("IQR Analysis"):
#         numeric_data = df.select_dtypes(exclude='object')

#         iqr_data = []

#         for col in numeric_data.columns:
#             Q1 = df[col].quantile(0.25)
#             Q3 = df[col].quantile(0.75)
#             IQR = Q3 - Q1
#             iqr_data.append([col, IQR])

#         iqr_df = pd.DataFrame(iqr_data, columns=["Column Name", "IQR Value"])

#         st.dataframe(iqr_df, width=800)

#     st.subheader("Analysing Percentiles for Numeric Data-Set :-")
#     if st.checkbox("Percentile Analysis"):
#         # Select only numeric columns
#         numeric_data = df.select_dtypes(exclude='object')

#         # Calculate percentiles (1%, 25%, 50%, 75%, 99%) for each numeric column
#         percentiles_df = df.describe(percentiles=[.01, .25, .5, .75, .99]).transpose()[["1%", "25%", "50%", "75%", "99%"]]

#         # Display the dataframe in Streamlit
#         st.dataframe(percentiles_df, width=800)

#     st.subheader("Analysing MAD and Skewness for Numeric Data-Set :-")
#     if st.checkbox("MAD and Skewness Analysis"):
#         # Select only numeric columns
#         numeric_data = df.select_dtypes(exclude='object')

#         # Calculate Median Absolute Deviation (MAD) for each numeric column
#         mad_df = numeric_data.apply(lambda x: (x - x.median()).abs().median()).reset_index()
#         mad_df.columns = ["Column Name", "MAD Value"]

#         # Calculate Skewness for each numeric column
#         skew_df = numeric_data.skew().reset_index()
#         skew_df.columns = ["Column Name", "Skewness Value"]

#         # Display MAD and Skewness dataframes
#         st.subheader("Median Absolute Deviation (MAD)")
#         st.dataframe(mad_df, width=800)

#         st.subheader("Skewness")
#         st.dataframe(skew_df, width=800)

    




# def show():
#     st.title("Different measures for Central Tendencies and Variances-")

#     central_tendencies(df)


# if __name__ == "__main__":
#     show()
