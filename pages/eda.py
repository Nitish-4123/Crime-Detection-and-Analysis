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

# Custom CSS for styling dataframes, title, and subheader
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
        .graph {
            border: 2px solid #87CEFA;  /* Light Blue Border */
            border-radius: 5px;
            padding: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

def render_dataframe(df):
    """Render a pandas dataframe as HTML with the custom styles."""
    return df.to_html(classes='dataframe', index=False)

def categorical_column(df, max_unique_values=15):
    categorical_column_list = []
    for column in df.columns:
        if df[column].nunique() < max_unique_values:
            categorical_column_list.append(column)
    return categorical_column_list

def eda(df):
    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Numbers of rows to view", 5)
        st.dataframe(df.head(number), width=800)

    # Data set info
    if st.checkbox("See Valuable insight for Data-Set"):
        info_df = pd.DataFrame({
            "Column Name": df.columns,
            "Non-Null Count": df.notnull().sum(),
            "Data Type": df.dtypes
        }).reset_index(drop=True)
    
        st.markdown(render_dataframe(info_df), unsafe_allow_html=True)

    # Check for null values
    if st.checkbox("Check null Values"):
        info_df2 = pd.DataFrame({
            "Column Name": df.columns,
            "Null total": df.isna().sum()
        }).reset_index(drop=True)

        st.markdown(render_dataframe(info_df2), unsafe_allow_html=True)

    # Show Summary
    if st.checkbox("Show Summary"):
        st.text("Summary")
        st.write(df.describe().T)

    # Show Columns
    if st.checkbox("Columns Names"):
        columns_df = pd.DataFrame(df.columns, columns=["Column Names"]).reset_index()
        st.markdown(render_dataframe(columns_df), unsafe_allow_html=True)

    # Show Shape
    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)
        data_dim = st.radio("Show Dimension by ", ("Rows", "Columns"))
        if data_dim == "Columns":
            st.text("Numbers of Columns")
            st.write(df.shape[1])
        elif data_dim == "Rows":
            st.text("Numbers of Rows")
            st.write(df.shape[0])
        else:
            st.write(df.shape)

    # Select Columns
    if st.checkbox("Select Column to show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select Columns", all_columns)
        new_df = df[selected_columns]
        st.markdown(render_dataframe(new_df), unsafe_allow_html=True)

    # Show Value Count
    if st.checkbox("Show Value Counts"):
        all_columns = df.columns.tolist()
        selected_columns = st.selectbox("Select Column", all_columns)
        st.write(df[selected_columns].value_counts())

    # Show Datatypes
    if st.checkbox("Show Data types"):
        st.text("Data Types")
        st.write(df.dtypes)

    # Plot and visualization
    st.subheader("Data Visualization")
    all_columns_names = df.columns.tolist()

    # Correlation Seaborn Plot
    if st.checkbox("Show Correlation Plot"):
        st.success("Generating Correlation Plot ...")
        if st.checkbox("Annot the Plot"):
            corr_plot = sns.heatmap(df.corr(), annot=True)
        else:
            corr_plot = sns.heatmap(df.corr())
        st.markdown('<div class="graph">', unsafe_allow_html=True)
        st.pyplot()
        st.markdown('</div>', unsafe_allow_html=True)

    # Count Plot
    if st.checkbox("Show Value Count Plots"):
        x = st.selectbox("Select Categorical Column", all_columns_names)
        st.success("Generating Plot ...")
        if x:
            fig, ax = plt.subplots()  # Create a new figure and axis
            if st.checkbox("Select Second Categorical column"):
                hue_all_column_name = df[df.columns.difference([x])].columns
                hue = st.selectbox("Select Column for Count Plot", hue_all_column_name)
                count_plot = sns.countplot(x=x, hue=hue, data=df, palette="Set2", ax=ax)  # Use the new axis `ax`
            else:
                count_plot = sns.countplot(x=x, data=df, palette="Set2", ax=ax)  # Use the new axis `ax`
            
            st.markdown('<div class="graph">', unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            # st.markdown(f"<div class='graph'>{st.pyplot(fig).to_html()}</div>",unsafe_allow_html=True)

    if st.checkbox("Show Pie Plot"):
        all_columns = categorical_column(df)
        selected_columns = st.selectbox("Select Column", all_columns)
        if selected_columns:
            st.success("Generating Pie Chart ...")
            fig, ax = plt.subplots()  # Create a new figure and axis
            pie_chart = df[selected_columns].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            st.markdown('<div class="graph">', unsafe_allow_html=True)
            st.pyplot(fig)  # Pass the figure explicitly
            st.markdown('</div>', unsafe_allow_html=True)
            # st.markdown('<div class="graph">',st.pyplot(),'</div>',unsafe_allow_html=True)


    # Customizable Plot
    st.subheader("Customizable Plot")

    type_of_plot = st.selectbox(
        "Select type of Plot", ["area", "bar", "line", "hist", "box", "kde"]
    )
    selected_columns_names = st.multiselect("Select Columns to plot", all_columns_names)

    if st.button("Generate Plot"):
        st.success(
            "Generating Customizable Plot of {} for {}".format(
                type_of_plot, selected_columns_names
            )
        )

        custom_data = df[selected_columns_names]
        if type_of_plot == "area":
            st.area_chart(custom_data)

        elif type_of_plot == "bar":
            st.bar_chart(custom_data)

        elif type_of_plot == "line":
            st.line_chart(custom_data)

        elif type_of_plot:
            custom_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.markdown('<div class="graph">', unsafe_allow_html=True)
            st.pyplot()
            st.markdown('</div>', unsafe_allow_html=True)

def app():
    st.sidebar.markdown("# Exploratory Data Analysis of the Data-Set 🎈")
    st.title("Exploratory Data Analysis of the Data-Set")
    st.subheader("Simple Data Science Explorer")
    eda(df)

if __name__ == "__main__":
    app()
