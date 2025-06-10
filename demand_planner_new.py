import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Demand Planning Tool", layout="wide")
st.title("📦 AI-Powered Demand Planning Tool (with Geo Region Support)")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Sales Forecast Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Load the Excel file with proper header
    df_raw = pd.read_excel(uploaded_file, header=1)

    # Identify columns that represent months (2024-xx or 2025-xx)
    date_cols = [col for col in df_raw.columns if str(col).startswith("2024") or str(col).startswith("2025")]

    # Melt the table to long format, keeping Geo Region
    df = df_raw.melt(
        id_vars=["Part Number", "Geo Region"],
        value_vars=date_cols,
        var_name="Month",
        value_name="Sales Qty"
    )

    # Clean and format
    df = df.dropna(subset=["Part Number", "Sales Qty"])
    df.rename(columns={"Part Number": "PN"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Month"], format="%Y-%m")

    # Show preview
    st.subheader("📄 Cleaned Sales Data Preview")
    st.dataframe(df.head())

    # User input for part number and geo
    pn_options = df["PN"].unique()
    geo_options = df["Geo Region"].unique()

    selected_pn = st.selectbox("Select Part Number (PN)", pn_options)
    selected_geo = st.selectbox("Select Geo Region", geo_options)
    forecast_horizon = st.slider("Forecast Horizon (months)", 1, 12, 6)

    # Filter and prepare data
    df_filtered = df[(df["PN"] == selected_pn) & (df["Geo Region"] == selected_geo)]
    df_grouped = df_filtered.groupby("Date")["Sales Qty"].sum().reset_index()
    df_grouped.columns = ["ds", "y"]

    # Show historical demand
    st.subheader(f"📊 Historical Demand for {selected_pn} in {selected_geo}")
    fig = px.line(df_grouped, x="ds", y="y", title="Historical Demand")
    st.plotly_chart(fig, use_container_width=True)

    # Forecasting using Prophet
    st.subheader(f"🔮 Forecast: {forecast_horizon} months")
    model = Prophet()
    model.fit(df_grouped)

    future = model.make_future_dataframe(periods=forecast_horizon * 30, freq='D')
    forecast = model.predict(future)

    # Resample daily forecast into monthly average
    forecast_monthly = forecast[["ds", "yhat"]].set_index("ds").resample("MS").mean().reset_index()
    forecast_monthly = forecast_monthly.tail(forecast_horizon)

    # Plot forecast
    fig_forecast = px.line(forecast_monthly, x="ds", y="yhat", title="Forecasted Monthly Demand")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Show table and download
    st.subheader("📋 Forecast Data")
    st.dataframe(forecast_monthly)

    csv = forecast_monthly.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name=f"{selected_pn}_{selected_geo}_forecast.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload your Excel file to begin.")
