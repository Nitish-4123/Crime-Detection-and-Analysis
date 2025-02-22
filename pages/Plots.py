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
    st.subheader("Density Plot for 'THEFT' Column:")
    if st.checkbox("Show Theft Density Plot"):
        # Create the plot with border color
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#FFE4B5')  # Light color between yellow and pink as "border"
        ax.set_facecolor('white')           # Background color of the plot area

        # Adjust margins to simulate a border
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        # Plot the density
        df['THEFT'].plot(kind='density', ax=ax, color='green', lw=2)
        ax.set_title('Density Plot of THEFT Column')
        ax.set_xlabel('THEFT')
        ax.set_ylabel('Density')

        # Display the plot in Streamlit
        st.pyplot(fig)


    st.subheader("Distribution Plot for 'YEAR' Column:")
    if st.checkbox("Show Year Distribution Plot"):
        # Create the plot with border color
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#FFE4B5')  # Light color between yellow and pink as "border"
        ax.set_facecolor('white')           # Background color of the plot area

        # Adjust margins to simulate a border
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        # Plot the distribution with Seaborn
        sns.histplot(df['YEAR'], kde=True, ax=ax, color='blue')
        ax.set_title('Distribution Plot of YEAR Column')
        ax.set_xlabel('YEAR')
        ax.set_ylabel('Density')

        # Display the plot in Streamlit
        st.pyplot(fig)

    st.subheader("Stacked KDE Plot for 'THEFT' by 'YEAR':")
    if st.checkbox("Show Stacked KDE Plot for Theft by Year"):
        # Create the displot with KDE stacking
        fig = sns.displot(df, x="THEFT", hue="YEAR", kind="kde", multiple="stack", height=6, aspect=1.5)
        
        # Set figure aesthetics
        fig.set_titles("Stacked KDE Plot of THEFT by YEAR")
        fig.set_axis_labels("THEFT", "Density")
        
        # Display plot in Streamlit
        st.pyplot(fig)

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

def show():
    st.sidebar.markdown("# Info. Plots Analysis  🎈")

    st.title("Showing Informative plots for Analysis")
    plots(df)

if __name__ == "__main__":
    show()
