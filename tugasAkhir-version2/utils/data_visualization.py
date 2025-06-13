import streamlit as st
import pandas as pd


def distribution_by_request(df, column_name):
    st.subheader(f"Distribusi dari {column_name}")
    chart_type = st.selectbox("Pilih jenis grafik", [
                              "Bar Chart", "Line Chart", "Area Chart"])

    # Get column counts and sort by index
    column_counts = df[column_name].value_counts().sort_index()

    # Display the selected chart type
    if chart_type == "Bar Chart":
        st.bar_chart(column_counts)
    elif chart_type == "Line Chart":
        st.line_chart(column_counts)
    elif chart_type == "Area Chart":
        st.area_chart(column_counts)
