import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Demand Planning Tool", layout="wide")
st.title("📦 AI-Powered Demand Planning Tool (Aggregated Regional Forecast)")

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
    region_forecasts = []

    for pn in pn_options:
        df_filtered = df[(df["PN"] == pn) & (df["Geo Region"] == selected_geo)]
        df_grouped = df_filtered.groupby("Date")["Sales Qty"].sum().reset_index()
        df_grouped.columns = ["ds", "y"]

        if len(df_grouped) < 2:
            continue  # skip PNs without enough data

        model = Prophet()
        model.fit(df_grouped)

        future = model.make_future_dataframe(periods=forecast_horizon * 30, freq='D')
        forecast = model.predict(future)
        forecast_monthly = forecast[["ds", "yhat"]].set_index("ds").resample("MS").mean().reset_index()
        forecast_monthly = forecast_monthly.tail(forecast_horizon)
        forecast_monthly["PN"] = pn
        region_forecasts.append(forecast_monthly)

    if not region_forecasts:
        st.warning("No PNs in this region have enough data for forecasting.")
    else:
        # Aggregate all PN forecasts by date (sum)
        agg_forecast = pd.concat(region_forecasts)
        agg_monthly = agg_forecast.groupby("ds")["yhat"].sum().reset_index()

        st.subheader(f"📊 Aggregated Historical Demand for {selected_geo}")
        # Show historical aggregated demand
        df_region_hist = df[df["Geo Region"] == selected_geo].groupby("Date")["Sales Qty"].sum().reset_index()
        fig_hist = px.line(df_region_hist, x="Date", y="Sales Qty", title="Historical Demand (All PNs)")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader(f"🔮 Aggregated Forecast: {forecast_horizon} months ({selected_geo})")
        fig_forecast = px.line(agg_monthly, x="ds", y="yhat", title="Aggregated Forecasted Monthly Demand")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("📋 Aggregated Forecast Data")
        st.dataframe(agg_monthly)

        csv = agg_monthly.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Aggregated Forecast CSV",
            data=csv,
            file_name=f"aggregated_{selected_geo}_forecast.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload your Excel file to begin.")
