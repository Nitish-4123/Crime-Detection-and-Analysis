
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom CSS for styling title, dataframes, and widgets
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

crime_data = pd.read_csv('crime data-set.csv')

# st.sidebar.markdown("# Main page 🎈")

# st.title("Indian Crime Forecasting")

# st.markdown(f"<div class='dataframe'>{crime_data().to_html(index=False)}</div>", unsafe_allow_html=True)

# st.markdown(f"<div class='dataframe'>{crime_data.info()}</div>", unsafe_allow_html=True)

# x = st.slider('Select a value')  # 👈 
# st.write(f"The square of {x} is {x * x}")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
# import warnings
# warnings.filterwarnings("ignore")
# from multipage import MultiPage
# from pages import data_overview, data_preprocessing, statistical_analysis
# st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")
st.title("Indian Crime Forecasting")
st.subheader('By Nitish and Vinay')

crime_data = pd.read_csv('crime data-set.csv')
# print(crime_data)


st.dataframe(crime_data)
# st.markdown(render_dataframe(crime_data), unsafe_allow_html=True)


st.write(crime_data.dtypes)

# st.map(crime_data)
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)