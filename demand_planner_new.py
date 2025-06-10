import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Demand Planning Tool", layout="wide")
st.title("📦 AI-Powered Demand Planning Tool (Aggregated Regional Forecast with PN/Geo Filtering)")

MIN_DATA_POINTS = 3  # Minimum required data points per (PN, Geo Region)

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
    valid_groups = []
    for (pn, geo), group in df.groupby(["PN", "Geo Region"]):
        if geo != selected_geo:
            continue
        if group["Date"].nunique() >= MIN_DATA_POINTS:
            valid_groups.append((pn, geo))

    if not valid_groups:
        st.warning("No PN/Geo Region combinations have enough data for forecasting in this region.")
    else:
        region_forecasts = []
        skipped_pns = []
        for pn, geo in valid_groups:
            group = df[(df["PN"] == pn) & (df["Geo Region"] == geo)]
            df_grouped = group.groupby("Date")["Sales Qty"].sum().reset_index()
            df_grouped.columns = ["ds", "y"]

            if len(df_grouped) < MIN_DATA_POINTS or df_grouped['y'].nunique() < 2:
                skipped_pns.append(pn)
                continue  # skip PNs without enough data or with constant values

            try:
                model = Prophet()
                model.fit(df_grouped)
                future = model.make_future_dataframe(periods=forecast_horizon * 30, freq='D')
                forecast = model.predict(future)
                forecast_monthly = forecast[["ds", "yhat"]].set_index("ds").resample("MS").mean().reset_index()
                forecast_monthly = forecast_monthly.tail(forecast_horizon)
                forecast_monthly["PN"] = pn
                forecast_monthly["Geo Region"] = geo
                region_forecasts.append(forecast_monthly)
            except Exception as e:
                skipped_pns.append(pn)
                continue

        if not region_forecasts:
            st.warning("No PNs in this region have enough data or a valid forecast.")
        else:
            # Aggregate all PN forecasts by date (sum)
            all_forecast_df = pd.concat(region_forecasts, ignore_index=True)
            agg_monthly = all_forecast_df.groupby("ds")["yhat"].sum().reset_index()

            st.subheader(f"📊 Aggregated Historical Demand for {selected_geo}")
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

            # Optionally show which PNs were skipped
            if skipped_pns:
                st.info(f"Skipped {len(skipped_pns)} PN(s) due to insufficient or invalid data: {', '.join(map(str, skipped_pns))}")

else:
    st.info("Please upload your Excel file to begin.")
