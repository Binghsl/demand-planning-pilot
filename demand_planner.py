import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Demand Planning Tool", layout="wide")

st.title("📈 AI-Powered Demand Planning Tool")

# Upload CSV
uploaded_file = st.file_uploader("Upload Sales History CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    
    st.subheader("🔍 Uploaded Data Preview")
    st.write(df_raw.head())

    # Data validation
    if not {"Date", "PN", "Sales Qty"}.issubset(df_raw.columns):
        st.error("CSV must have columns: Date, PN, Sales Qty")
    else:
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        pn_list = df_raw["PN"].unique()
        selected_pn = st.selectbox("Select Part Number (PN)", pn_list)
        horizon = st.slider("Forecast Horizon (days)", 7, 180, 30)

        # Filter data
        df_pn = df_raw[df_raw["PN"] == selected_pn]
        df_daily = df_pn.groupby("Date").agg({"Sales Qty": "sum"}).reset_index()
        df_daily.columns = ["ds", "y"]

        st.subheader(f"📊 Historical Sales for {selected_pn}")
        fig_hist = px.line(df_daily, x="ds", y="y", title="Sales Over Time")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Forecasting
        st.subheader(f"🔮 Forecast for {selected_pn} ({horizon} days)")

        m = Prophet(daily_seasonality=True)
        m.fit(df_daily)

        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)

        fig_forecast = px.line(forecast, x="ds", y="yhat", title="Forecasted Demand")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("📋 Forecast Table")
        st.write(forecast[["ds", "yhat"]].tail(horizon))

        csv_export = forecast[["ds", "yhat"]].tail(horizon)
        csv = csv_export.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", data=csv, file_name=f"{selected_pn}_forecast.csv", mime="text/csv")
else:
    st.info("Upload a sales history CSV to begin.")
