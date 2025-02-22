import matplotlib
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import io
from statsmodels.stats.weightstats import ztest
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

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


# Global function to display content inside a styled box
def display_in_box(content):
    """
    This function displays content inside a custom HTML box with a light yellow border and a transparent or no background.
    """
    html_content = f"""
    <div style="border: 2px solid #f1c232; padding: 10px; border-radius: 5px;">
        <p style="color: #333; font-weight: bold;">{content}</p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def inferential_tests(df):
    # Ask the user to select two years for comparison
    st.subheader("Applying Z-Test on The Selective Columns:")
    year_1 = st.selectbox("Select the first year for comparison:", df['YEAR'].unique())
    year_2 = st.selectbox("Select the second year for comparison:", df['YEAR'].unique())

    # Filter the data for the selected years and columns
    crime_data_year_1 = df[df['YEAR'] == year_1]['TOTAL IPC CRIMES']
    crime_data_year_2 = df[df['YEAR'] == year_2]['TOTAL IPC CRIMES']


    if st.checkbox("Show Test Results"):
        # Perform Z-test
        if len(crime_data_year_1) > 1 and len(crime_data_year_2) > 1:
            z_stat, p_value = ztest(crime_data_year_1, crime_data_year_2)

            # Prepare the Z-test results content
            result_content = f"""
            <h4>Z-test Results between {year_1} and {year_2}</h4>
            <p><strong>Z-test Statistic:</strong> {z_stat}</p>
            <p><strong>P-value:</strong> {p_value}</p>
            <p><strong>Hypothesis Test Result:</strong></p>
            <p>{'Reject null hypothesis - There is a significant difference in the total IPC crimes between the selected years.' if p_value < 0.05 else 'Accept null hypothesis - There is no significant difference in the total IPC crimes between the selected years.'}</p>
            """

            # Display the Z-test results inside the styled box
            display_in_box(result_content)

        else:
            display_in_box("Insufficient data for the selected years. Please choose years with enough data.")


    # Select the states
    st.subheader("Applying One-Way Anova Test on The Data-Set:")
    states = df['STATE/UT'].unique()
    murder_rates = {}

    for state in states:
        murder_rates[state] = df[df['STATE/UT'] == state]['MURDER']

    # Perform the ANOVA test
    f_stat, p_value = f_oneway(*murder_rates.values())

    # Prepare the content for results
    result_content = f"""
    <h4>One-way ANOVA Test Results (Comparing Murder Rates Across States)</h4>
    <p><strong>F-statistic:</strong> {f_stat}</p>
    <p><strong>P-value:</strong> {p_value}</p>
    <p><strong>Hypothesis Test Result:</strong></p>
    <p>{'Reject null hypothesis - There is a significant difference in the mean number of murders across different states in India.' if p_value < 0.05 else 'Accept null hypothesis - There is no significant difference in the mean number of murders across different states in India.'}</p>
    """
    
    # Display result based on checkbox selection
    if st.checkbox("Show ANOVA Test Results"):
        # Display the ANOVA test results inside the styled box
        display_in_box(result_content)


    # Perform the post-hoc Tukey test
    st.subheader("Applying Tukey Test on The Data-Set:")
    tukey_results = pairwise_tukeyhsd(df['MURDER'], df['STATE/UT'])

    # Convert the Tukey results into a DataFrame for better display
    tukey_summary = pd.DataFrame(data=tukey_results.summary().data[1:], 
                                 columns=tukey_results.summary().data[0])

    # Prepare the content for results in DataFrame format
    # tukey_summary

    # Display result based on checkbox selection
    if st.checkbox("Show Tukey Test Results"):
        # Display the Tukey test results inside the styled box
        # display_in_box(result_content)
        st.dataframe(tukey_summary)


    st.subheader("Applying T-Test on The Data-Set:")
    # Dropdown for selecting a state
    state = st.selectbox("Select a State for T-Test Comparison", df['STATE/UT'].unique())

    # Get the data for selected state and other states
    state_thefts = df[df['STATE/UT'] == state]['THEFT']
    other_thefts = df[df['STATE/UT'] != state]['THEFT']

    # Perform the t-test
    t_stat, p_val = ttest_ind(state_thefts, other_thefts)

    # Format the results as text
    result_text = f"""
    <p><strong>state:</strong> {state}</p>
    <p><strong>T-statistics:</strong> {t_stat:.4f}</p>
    <p><strong>P-value:</strong> {p_val:.4f}</p>
    """
    
    if p_val < 0.05:
        result_text += f"\n\n**Hypothesis**: Reject null hypothesis - Significant difference found between thefts in {state} and all other states combined."
    else:
        result_text += f"\n\n**Hypothesis**: Accept null hypothesis - No significant difference found between thefts in {state} and all other states combined."

    # Display the result in a checkbox
    if st.checkbox("Show T-Test Results for Selected State"):
        display_in_box(result_text)


    # Perform Chi-Square test based on the selected columns
    st.subheader("Applying Chi-Square Test on The Data-Set:")
    cont_table = pd.crosstab(df['STATE/UT'], df['TOTAL IPC CRIMES'])

    # Perform the chi-square test
    chi2, p_val, dof, expected = chi2_contingency(cont_table)

    # Format the results as text
    result_text = f"""
    <p><strong>Chi-square statistic:</strong> {chi2:.4f}</p>
    <p><strong>p-value:</strong> {p_val:.4f}</p>
    <p><strong>Degrees of freedom:</strong> {dof}</p>
    """
    
    if p_val < 0.05:
        result_text += f"\n\n**Hypothesis**: Reject null hypothesis - There is a significant relationship between STATE/UT and TOTAL IPC CRIMES."
    else:
        result_text += f"\n\n**Hypothesis**: Accept null hypothesis - There is no significant relationship between STATE/UT and TOTAL IPC CRIMES."

    # Display the result in a checkbox
    if st.checkbox("Show Chi-Square Test Results"):
        display_in_box(result_text)

def show():
    st.sidebar.markdown("#Inferential Statistic Tests Analysis  🎈")

    st.title("Analysis using Inferential Statistic Tests")
    inferential_tests(df)
    
if __name__ == "__main__":
    show()
