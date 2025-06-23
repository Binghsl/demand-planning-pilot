import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Demand Planning Tool", layout="wide")
st.title("📦 AI-Powered Demand Planning Tool (PN Forecast + Inventory Risk)")

MIN_DATA_POINTS = 3

uploaded_file = st.file_uploader("Upload Sales Forecast Excel File (.xlsx)", type=["xlsx"])
inv_file = st.file_uploader("Upload Inventory Data (.xlsx, columns: PN, On Hand, In Transit)", type=["xlsx"])
bo_file = st.file_uploader("Upload Backorder Data (.xlsx, columns: PN, Backorder)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, header=1)
    date_cols = [col for col in df_raw.columns if str(col).startswith("2024") or str(col).startswith("2025")]

    df = df_raw.melt(
        id_vars=["Part Number"],
        value_vars=date_cols,
        var_name="Month",
        value_name="Sales Qty"
    )
    df = df.dropna(subset=["Part Number", "Sales Qty"])
    df.rename(columns={"Part Number": "PN"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Month"], format="%Y-%m")

    st.subheader("📄 Cleaned Sales Data Preview")
    st.dataframe(df.head())

    valid_pns = [
        pn for pn, group in df.groupby("PN")
        if group["Date"].nunique() >= MIN_DATA_POINTS
    ]

    forecast_horizon = st.slider("Forecast Horizon (months)", 1, 12, 6)

    if not valid_pns:
        st.warning("No PNs have enough data for forecasting.")
    else:
        forecasts = []
        skipped_pns = []

        for pn in valid_pns:
            group = df[df["PN"] == pn]
            df_grouped = group.groupby("Date")["Sales Qty"].sum().reset_index()
            df_grouped.columns = ["ds", "y"]

            if len(df_grouped) < MIN_DATA_POINTS or df_grouped['y'].nunique() < 2:
                skipped_pns.append(pn)
                continue

            try:
                model = Prophet()
                model.fit(df_grouped)
                future = model.make_future_dataframe(periods=forecast_horizon * 30, freq='D')
                forecast = model.predict(future)

                forecast_monthly = (
                    forecast[["ds", "yhat"]]
                    .set_index("ds")
                    .resample("MS").mean()
                    .reset_index()
                    .tail(forecast_horizon)
                )
                forecast_monthly["PN"] = pn
                forecasts.append(forecast_monthly)
            except Exception:
                skipped_pns.append(pn)

        if not forecasts:
            st.warning("No PNs could be forecasted.")
        else:
            all_forecast_df = pd.concat(forecasts, ignore_index=True)
            all_forecast_df = all_forecast_df.rename(columns={"ds": "Month", "yhat": "Forecast Qty"})

            st.subheader("📋 Forecast by PN and Month")
            st.dataframe(all_forecast_df)

            csv_pn = all_forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download PN-level Forecast CSV",
                data=csv_pn,
                file_name="pn_level_forecast.csv",
                mime="text/csv"
            )

            # 📈 Aggregated Forecast
            agg_monthly = all_forecast_df.groupby("Month")["Forecast Qty"].sum().reset_index()

            st.subheader("📊 Aggregated Historical Demand")
            df_hist = df.groupby("Date")["Sales Qty"].sum().reset_index()
            fig_hist = px.line(df_hist, x="Date", y="Sales Qty", title="Historical Demand (All PNs)")
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader(f"🔮 Aggregated Forecast: {forecast_horizon} months")
            fig_forecast = px.line(agg_monthly, x="Month", y="Forecast Qty", title="Forecasted Monthly Demand")
            st.plotly_chart(fig_forecast, use_container_width=True)

            # 📦 Inventory & Backorder
            if inv_file is not None and bo_file is not None:
                inv_df = pd.read_excel(inv_file)
                bo_df = pd.read_excel(bo_file)

                st.subheader("📦 Inventory Data Preview")
                st.dataframe(inv_df.head())
                st.subheader("📝 Backorder Data Preview")
                st.dataframe(bo_df.head())

                latest_forecast = (
                    all_forecast_df.groupby("PN").apply(lambda x: x.sort_values("Month").iloc[-1])
                    .reset_index(drop=True)
                )

                merged = pd.merge(latest_forecast, inv_df, how="left", on="PN")
                merged = pd.merge(merged, bo_df, how="left", on="PN")

                # Fill missing values
                merged["Backorder"] = merged["Backorder"].fillna(0)
                merged["On Hand"] = merged["On Hand"].fillna(0)
                merged["In Transit"] = merged["In Transit"].fillna(0)

                # Risk logic
                merged["Overstock Risk"] = (merged["On Hand"] + merged["In Transit"]) - (merged["Backorder"] + merged["Forecast Qty"])
                merged["Overstock Flag"] = merged["Overstock Risk"] > 0

                st.subheader("🛑 Overstock Risk Analysis")
                st.dataframe(merged[[
                    "PN", "Forecast Qty", "On Hand", "In Transit", "Backorder", "Overstock Risk", "Overstock Flag"
                ]])

                csv_risk = merged.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Overstock Risk CSV",
                    data=csv_risk,
                    file_name="overstock_risk_analysis.csv",
                    mime="text/csv"
                )
            elif inv_file or bo_file:
                st.info("Please upload both Inventory and Backorder files for risk analysis.")

            if skipped_pns:
                st.info(f"Skipped {len(skipped_pns)} PN(s) due to insufficient data: {', '.join(map(str, skipped_pns))}")
else:
    st.info("Please upload your Excel sales forecast to begin.")
