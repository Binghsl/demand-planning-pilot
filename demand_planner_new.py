import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Demand Planning Tool", layout="wide")
st.title("📦 AI-Powered Demand Planning Tool (with Geo Region Support)")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Sales Forecast Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, header=1)
    date_cols = [col for col in df_raw.columns if str(col).startswith("2024") or str(col).startswith("2025")]
    df = df_raw.melt(
        id_vars=["Part Number", "Geo Region"],
        value_vars=date_cols,
        var_name="Month",
        value_name="Sales Qty"
    )
    df = df.dropna(subset=["Part Number", "Sales Qty"])
    df.rename(columns={"Part Number": "PN"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Month"], format="%Y-%m")

    st.subheader("📄 Cleaned Sales Data Preview")
    st.dataframe(df.head())

    geo_options = df["Geo Region"].unique()
    selected_geo = st.selectbox("Select Geo Region", geo_options)
    forecast_horizon = st.slider("Forecast Horizon (months)", 1, 12, 6)

    pn_options = df[df["Geo Region"] == selected_geo]["PN"].unique()
    st.write(f"Forecasting for **all PNs** in region: **{selected_geo}**")

    all_forecast = []
    for pn in pn_options:
        df_filtered = df[(df["PN"] == pn) & (df["Geo Region"] == selected_geo)]
        df_grouped = df_filtered.groupby("Date")["Sales Qty"].sum().reset_index()
        df_grouped.columns = ["ds", "y"]

        model = Prophet()
        model.fit(df_grouped)

        future = model.make_future_dataframe(periods=forecast_horizon * 30, freq='D')
        forecast = model.predict(future)
        forecast_monthly = forecast[["ds", "yhat"]].set_index("ds").resample("MS").mean().reset_index()
        forecast_monthly = forecast_monthly.tail(forecast_horizon)
        forecast_monthly.insert(0, "PN", pn)
        forecast_monthly.insert(1, "Geo Region", selected_geo)

        # Plot for each PN
        fig_forecast = px.line(forecast_monthly, x="ds", y="yhat", title=f"Forecasted Monthly Demand for PN {pn}")
        st.plotly_chart(fig_forecast, use_container_width=True)

        all_forecast.append(forecast_monthly)

    # Combine all forecasts
    all_forecast_df = pd.concat(all_forecast, ignore_index=True)
    st.subheader("📋 Forecast Data for All PNs")
    st.dataframe(all_forecast_df)

    csv = all_forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download All PNs Forecast CSV",
        data=csv,
        file_name=f"all_pn_{selected_geo}_forecast.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload your Excel file to begin.")
