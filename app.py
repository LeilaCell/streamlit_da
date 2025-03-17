import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the title of the app
st.title("Streamlit Template for Data Visualizations")

# Create multiple divisions (blocks) using containers
# Block 1: Introduction
with st.container():
    st.header("Introduction")
    st.write("""
        This template is designed to help you get started quickly with Streamlit.
        Use the sections below to display different visualizations.
    """)

# Block 2: Line Chart Example
with st.container():
    st.header("Line Chart Example")
    # Create sample data
    df_line = pd.DataFrame({
        'x': np.arange(0, 10, 0.1),
        'y': np.sin(np.arange(0, 10, 0.1))
    })
    # Plot using matplotlib
    fig_line, ax_line = plt.subplots()
    ax_line.plot(df_line['x'], df_line['y'], label='Sine Wave')
    ax_line.set_xlabel('X Axis')
    ax_line.set_ylabel('Y Axis')
    ax_line.legend()
    st.pyplot(fig_line)

# Block 3: Bar Chart Example
with st.container():
    st.header("Bar Chart Example")
    # Create sample data
    categories = ['A', 'B', 'C', 'D']
    values = np.random.randint(10, 100, size=4)
    df_bar = pd.DataFrame({
        'Category': categories,
        'Value': values
    })
    # Plot using matplotlib
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(df_bar['Category'], df_bar['Value'], color='skyblue')
    ax_bar.set_xlabel('Category')
    ax_bar.set_ylabel('Value')
    st.pyplot(fig_bar)

# Block 4: Data Table
with st.container():
    st.header("Data Table Example")
    # Create a sample dataframe
    data = {
        "Column 1": np.random.rand(5),
        "Column 2": np.random.rand(5)
    }
    df_table = pd.DataFrame(data)
    st.write(df_table)
