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

def plots(df):

    # Dropdown for selecting a numeric column
    st.subheader("Dynamic Histogram Plot:")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    selected_column = st.selectbox("Select Column for Histogram", numeric_columns)
    
    # Checkbox to show the histogram only if selected
    if st.checkbox("Show Histogram for Selected Column"):
        # Create histogram based on the selected column
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df[selected_column], bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {selected_column}")
        ax.set_xlabel(f"Number of {selected_column}")
        ax.set_ylabel("Frequency")
        
        # Display the histogram
        st.pyplot(fig)


     # Dropdown for selecting a numeric column for boxplot
    st.subheader("Dynamic Boxplot:")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    selected_column_box = st.selectbox("Select Column for Boxplot", numeric_columns)
    
    # Checkbox to show the boxplot only if selected
    if st.checkbox("Show Boxplot for Selected Column"):
        # Create boxplot based on the selected column
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(df[selected_column_box])
        ax.set_title(f"Kurtosis Plot of {selected_column_box}")
        ax.set_xlabel(selected_column_box)
        
        # Display the boxplot
        st.pyplot(fig)

    # Dropdown for selecting a numeric column for the pie chart
    st.subheader("Dynamic Pie Chart:")
    numeric_columns_pie = df.select_dtypes(include=['number']).columns.tolist()
    selected_column_pie = st.selectbox("Select Column for Pie Chart", numeric_columns_pie)

    # Checkbox to show the pie chart only if selected
    if st.checkbox("Show Pie Chart for Selected Column by Year"):
        # Filter numeric columns and group by 'YEAR' before calculating mean
        state = df.groupby('YEAR')[selected_column_pie].mean()

        # Create pie chart based on the selected column
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(state, labels=state.index, startangle=90, shadow=True,
            textprops={'fontsize': 10, 'color': 'green'}, autopct='%0.2f%%')

        ax.set_title(f'Crime in India: {selected_column_pie} by Year')
        
        # Display the pie chart
        st.pyplot(fig)


    # Dropdown for selecting a numeric column for the bar chart
    st.subheader("Dynamic Bar Chart by State/Union Territory")
    numeric_columns_bar = df.select_dtypes(include=['number']).columns.tolist()
    selected_column_bar = st.selectbox("Select Column for Bar Chart", numeric_columns_bar)

    # Checkbox to show the bar chart only if selected
    if st.checkbox("Show Bar Chart for Selected Column by State/Union Territory"):
        # Group by 'STATE/UT' and calculate the sum of the selected column
        state_totals = df.groupby('STATE/UT')[selected_column_bar].sum()

        # Create bar chart
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.bar(state_totals.index, state_totals.values)
        ax.set_xticklabels(state_totals.index, rotation=90)
        ax.set_xlabel('State/Union Territory')
        ax.set_ylabel(f'Total {selected_column_bar}')
        ax.set_title(f'Total {selected_column_bar} by State/Union Territory')

        # Display the bar chart
        st.pyplot(fig)

    # Dropdowns for selecting numeric columns for the scatter plot
    st.subheader("Dynamic Scatter Plot")
    numeric_columns_scatter = df.select_dtypes(include=['number']).columns.tolist()
    x_column_scatter = st.selectbox("Select X-axis Column", numeric_columns_scatter, index=numeric_columns_scatter.index('THEFT') if 'THEFT' in numeric_columns_scatter else 0)
    y_column_scatter = st.selectbox("Select Y-axis Column", numeric_columns_scatter, index=numeric_columns_scatter.index('BURGLARY') if 'BURGLARY' in numeric_columns_scatter else 1)

    # Checkbox to show the scatter plot only if selected
    if st.checkbox("Show Scatter Plot for Selected Columns"):
        # Create scatter plot with the selected columns
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(df[x_column_scatter], df[y_column_scatter], color='blue')
        ax.set_xlabel(f'Total {x_column_scatter} Crimes')
        ax.set_ylabel(f'Total {y_column_scatter} Crimes')
        ax.set_title(f'{x_column_scatter} vs. {y_column_scatter} Crimes')

        # Display the scatter plot
        st.pyplot(fig)


    # Dropdowns for selecting columns for the heatmap correlation
    st.subheader("Dynamic Heatmap for Correlation Matrix")

    # Filter numeric columns for selection
    numeric_columns_heatmap = df.select_dtypes(include=['number']).columns.tolist()

    # Option to choose whether to visualize correlation for selected columns or the entire dataset
    heatmap_option = st.radio("Choose Heatmap Option", ["Selected Columns", "Entire Dataset"])

    if heatmap_option == "Selected Columns":
        # Allow the user to select columns for row-wise and column-wise correlation
        selected_columns_row = st.selectbox("Select Row-wise Columns for Correlation", numeric_columns_heatmap)
        selected_columns_col = st.selectbox("Select Column-wise Columns for Correlation", numeric_columns_heatmap)

        # Checkbox to show the heatmap only if columns are selected
        if st.checkbox("Show Correlation Heatmap"):
            if selected_columns_row and selected_columns_col:
                # Compute correlation matrix for the selected rows and columns
                corr_matrix = df[[selected_columns_row, selected_columns_col]].corr()

                # Create heatmap for the selected row and column correlation
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='icefire', fmt=".2f", linewidths=0.5)

                # Display the heatmap using the figure object
                st.pyplot(fig)
            else:
                st.warning("Please select both row-wise and column-wise columns to compute the correlation.")

    elif heatmap_option == "Entire Dataset":
        # Filter only numeric columns for correlation (exclude non-numeric columns)
        numeric_df = df.select_dtypes(include=['number'])

        # Compute correlation matrix for the entire dataset with numeric columns only
        corr_matrix = numeric_df.corr()

        # Create heatmap for the entire dataset's correlation
        fig, ax = plt.subplots(figsize=(25, 15))
        sns.heatmap(corr_matrix, annot=True, cmap='icefire', fmt=".2f", linewidths=0.5)

        # Display the heatmap using the figure object
        st.pyplot(fig)

def show():
    st.sidebar.markdown("# Dynamic Plotting of Graphs and Charts 🎈")

    st.title("Dynamic Plotting of Graphs and Charts")
    plots(df)

if __name__ == "__main__":
    show()
