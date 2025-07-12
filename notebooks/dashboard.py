import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# For Sales Forecasting
import pmdarima as pm
from pmdarima import model_selection
import numpy as np

# For Customer Segmentation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# For Market Basket Analysis
from mlxtend.frequent_patterns import apriori, association_rules

# --- Page setup ---
st.set_page_config(page_title="Supermarket Sales Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom Title
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🛒 AI-Powered Supermarket Sales Analytics</h1>
    <hr style='border:1px solid #ccc'>
""", unsafe_allow_html=True)

# --- Load Data ---
# Ensure your 'data' folder and 'supermarket_sales.csv' are in the same directory as your app.py
try:
    df = pd.read_csv("data/supermarket_sales.csv")
    
    df["Date"] = pd.to_datetime(df["Date"])
    # Assuming 'Time' column has mixed HH:MM and HH:MM:SS formats,
    # or just HH:MM:SS (most common). errors='coerce' handles inconsistencies.
    df["Time"] = pd.to_datetime(df["Time"], errors='coerce').dt.time
    
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    
    # Extract hour from the converted 'Time' column
    df["Hour"] = pd.to_datetime(df["Time"], errors='coerce').dt.hour 
    
    # Add day of week
    df["DayOfWeek"] = df["Date"].dt.day_name()

except FileNotFoundError:
    st.error("Error: supermarket_sales.csv not found. Please ensure it's in a 'data/' directory.")
    st.stop() # Stop the app if data isn't found

# --- Sidebar Filters ---
st.sidebar.header("📊 Filter the Data")
all_cities = df["City"].unique()
city = st.sidebar.multiselect("Select City:", options=all_cities, default=all_cities)

all_product_lines = df["Product line"].unique()
product_line = st.sidebar.multiselect("Select Product Line:", options=all_product_lines, default=all_product_lines)

min_date, max_date = df["Date"].min(), df["Date"].max()
date_range = st.sidebar.date_input("Date Range:", [min_date, max_date])

# Apply Filters
df_filtered = df[
    (df["City"].isin(city)) &
    (df["Product line"].isin(product_line)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Handle empty filtered dataframe
if df_filtered.empty:
    st.warning("No data found for the selected filters. Please adjust your selections.")
    st.stop() # Stop execution if no data

# --- Key Metrics ---
st.markdown("### 📌 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4) # Added one more column
col1.metric("Total Sales", f"${df_filtered['Total'].sum():,.2f}")
col2.metric("Avg. Rating", f"{df_filtered['Rating'].mean():.2f} ⭐")
col3.metric("Avg. Gross Income", f"${df_filtered['gross income'].mean():.2f}")
col4.metric("Total Customers", f"{df_filtered['Invoice ID'].nunique():,}")


# --- Download CSV ---
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name='filtered_supermarket_sales.csv',
    mime='text/csv'
)

# --- Display Table ---
st.markdown("### 📄 Filtered Data Preview")
st.dataframe(df_filtered.head()) # Show only head for brevity

# --- Visualizations (Using Plotly Express for Interactivity) ---
st.markdown("### 📈 Interactive Sales & Performance Trends")

# Sales by Product Line (Plotly)
st.markdown("#### Sales by Product Line")
fig_prod_line = px.bar(
    df_filtered.groupby("Product line")["Total"].sum().reset_index(),
    x="Total", y="Product line", orientation='h',
    title="Total Sales by Product Line",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    template="plotly_white"
)
fig_prod_line.update_layout(showlegend=False)
st.plotly_chart(fig_prod_line, use_container_width=True)

# Gross Income by City (Plotly)
st.markdown("#### Gross Income by City")
fig_city_income = px.bar(
    df_filtered.groupby("City")["gross income"].sum().reset_index(),
    x="City", y="gross income",
    title="Total Gross Income by City",
    color_discrete_sequence=px.colors.qualitative.Dark2,
    template="plotly_white"
)
fig_city_income.update_layout(showlegend=False)
st.plotly_chart(fig_city_income, use_container_width=True)

# Payment Method Distribution (Plotly)
st.markdown("#### Payment Method Distribution")
payment_counts = df_filtered["Payment"].value_counts().reset_index()
payment_counts.columns = ['Payment Method', 'Count']
fig_payment = px.pie(
    payment_counts,
    values='Count',
    names='Payment Method',
    title='Payment Method Distribution',
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.D3
)
st.plotly_chart(fig_payment, use_container_width=True)

# Monthly Sales Trend (Plotly)
st.markdown("#### Monthly Sales Trend")
# Convert 'Month' back to datetime for proper sorting and plotting
monthly_sales_plotly = df_filtered.groupby(pd.to_datetime(df_filtered["Month"]))["Total"].sum().reset_index()
monthly_sales_plotly.columns = ["Month", "Total Sales"]
fig_monthly_trend = px.line(
    monthly_sales_plotly,
    x="Month", y="Total Sales",
    title="Monthly Sales Trend",
    markers=True,
    line_shape="linear",
    color_discrete_sequence=["#1f77b4"], # A nice blue
    template="plotly_white"
)
fig_monthly_trend.update_xaxes(tickformat="%b %Y") # Format x-axis for months
st.plotly_chart(fig_monthly_trend, use_container_width=True)


# Sales by Gender (Plotly)
st.markdown("#### Total Sales by Gender")
if not df_filtered.empty and "Gender" in df_filtered.columns:
    gender_sales = df_filtered.groupby("Gender", as_index=False)["Total"].sum()
    fig_gender = px.bar(
        gender_sales,
        x="Gender", y="Total",
        title="Total Sales by Gender",
        color="Gender", # Color bars by gender
        color_discrete_map={"Male": "cornflowerblue", "Female": "lightcoral"},
        template="plotly_white"
    )
    st.plotly_chart(fig_gender, use_container_width=True)
else:
    st.warning("⚠️ 'Gender' column missing or no filtered data available. Adjust your filters.")

# Sales by Day of Week
st.markdown("#### Sales by Day of Week")
day_of_week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sales_by_day = df_filtered.groupby("DayOfWeek")["Total"].sum().reindex(day_of_week_order).reset_index()
fig_day_week = px.bar(
    sales_by_day,
    x="DayOfWeek", y="Total",
    title="Total Sales by Day of Week",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    template="plotly_white"
)
st.plotly_chart(fig_day_week, use_container_width=True)

# Sales by Hour of Day
st.markdown("#### Sales by Hour of Day")
sales_by_hour = df_filtered.groupby("Hour")["Total"].sum().reset_index()
fig_hour = px.line(
    sales_by_hour,
    x="Hour", y="Total",
    title="Total Sales by Hour of Day",
    markers=True,
    color_discrete_sequence=["#6a0572"], # Purple
    template="plotly_white"
)
st.plotly_chart(fig_hour, use_container_width=True)


# --- Advanced AI/ML Insights ---
st.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True)
st.markdown("## 🧠 Advanced AI & Machine Learning Insights")

# --- 1. Sales Forecasting ---
st.markdown("### 🔮 Sales Forecasting")
st.info("Predicting future sales trends based on historical data. Note: More data yields better forecasts.")

forecast_period = st.slider("Select forecast period (months):", min_value=1, max_value=6, value=2)

# Aggregate data for forecasting (monthly totals)
sales_ts = df_filtered.set_index("Date").resample("MS")["Total"].sum()

if len(sales_ts) > 12: # Need enough data points for a robust forecast
    try:
        # Fit auto_arima model
        with st.spinner('Training forecasting model...'):
            model = pm.auto_arima(sales_ts, seasonal=True, m=12,
                                  D=1, # seasonal differencing
                                  trace=False,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)

        # Make predictions
        forecast, conf_int = model.predict(n_periods=forecast_period, return_conf_int=True)
        forecast_index = pd.date_range(start=sales_ts.index[-1] + pd.DateOffset(months=1), periods=forecast_period, freq='MS')

        forecast_series = pd.Series(forecast, index=forecast_index)
        conf_int_df = pd.DataFrame(conf_int, index=forecast_index, columns=['lower_bound', 'upper_bound'])

        # Combine historical and forecasted data for plotting
        plot_df = sales_ts.reset_index()
        plot_df.columns = ['Date', 'Sales']
        plot_df['Type'] = 'Historical'

        forecast_plot_df = forecast_series.reset_index()
        forecast_plot_df.columns = ['Date', 'Sales']
        forecast_plot_df['Type'] = 'Forecast'

        combined_df = pd.concat([plot_df, forecast_plot_df])

        # Plot with Plotly
        fig_forecast = px.line(
            combined_df, x='Date', y='Sales', color='Type',
            title='Supermarket Sales Forecast',
            markers=True,
            color_discrete_map={'Historical': 'blue', 'Forecast': 'red'},
            template="plotly_white"
        )
        fig_forecast.add_trace(
            px.scatter(conf_int_df.reset_index(), x='index', y='lower_bound', color_discrete_sequence=['gray']).data[0]
        )
        fig_forecast.add_trace(
            px.scatter(conf_int_df.reset_index(), x='index', y='upper_bound', color_discrete_sequence=['gray']).data[0]
        )
        fig_forecast.update_traces(showlegend=False) # Hide scatter legend
        fig_forecast.add_annotation( # Add text for confidence interval
            xref="paper", yref="paper", x=1.05, y=0.5, showarrow=False,
            text=f"Confidence Interval: {model.confidence_interval * 100:.0f}%",
            font=dict(size=10, color="gray")
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("Forecasted Sales:")
        st.dataframe(forecast_series.rename("Forecasted Sales").apply(lambda x: f"${x:,.2f}"))

    except Exception as e:
        st.error(f"Could not generate forecast. Error: {e}. Try selecting a larger date range for more data.")
        st.warning("Forecasting requires sufficient historical data (at least 12 months for seasonal models).")
else:
    st.warning("Not enough historical data for a robust sales forecast (at least 12 months recommended). Please broaden your date range.")


# --- 2. Customer Segmentation ---
st.markdown("### 👥 Customer Segmentation (K-Means Clustering)")
st.info("Groups customers based on their purchasing behavior to identify distinct customer segments.")

# Prepare data for clustering
customer_data = df_filtered.groupby('Invoice ID').agg(
    Total_Spend=('Total', 'sum'),
    Avg_Rating=('Rating', 'mean'),
    Num_Purchases=('Invoice ID', 'count')
).reset_index()

if len(customer_data) >= 3: # Need at least k customers to form clusters
    features = ['Total_Spend', 'Avg_Rating', 'Num_Purchases']
    X = customer_data[features]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal number of clusters (Elbow Method - for internal use, not user interactive here)
    # inertia = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    #     kmeans.fit(X_scaled)
    #     inertia.append(kmeans.inertia_)
    # fig_elbow = px.line(x=range(1, 11), y=inertia, title="Elbow Method for K-Means")
    # st.plotly_chart(fig_elbow)

    num_clusters = st.slider("Select number of customer segments (K):", min_value=2, max_value=5, value=3)

    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

        st.subheader(f"Customer Segments (K={num_clusters}):")
        # Visualize clusters
        fig_cluster = px.scatter_3d(
            customer_data,
            x='Total_Spend', y='Avg_Rating', z='Num_Purchases',
            color='Cluster',
            title='Customer Segments based on Spend, Rating, and Purchases',
            hover_data=['Invoice ID'],
            color_continuous_scale=px.colors.qualitative.Bold,
            template="plotly_white"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown("#### Segment Characteristics:")
        # Display characteristics of each cluster
        cluster_summary = customer_data.groupby('Cluster')[features].mean().reset_index()
        st.dataframe(cluster_summary.round(2))

        st.info("""
        **Interpreting Clusters:**
        - **Cluster 0:** Customers with X characteristics.
        - **Cluster 1:** Customers with Y characteristics.
        - ...and so on.
        Look at the 'Segment Characteristics' table to understand what differentiates each cluster (e.g., high spenders, frequent buyers, high raters).
        """)

    except Exception as e:
        st.error(f"Could not perform customer segmentation. Error: {e}. Check if you have enough unique customers (min 3) after filtering.")
else:
    st.warning("Not enough unique customers to perform segmentation. Please ensure at least 3 unique 'Invoice ID' values are present after filtering.")


# --- 3. Market Basket Analysis ---
st.markdown("### 🛍️ Market Basket Analysis (Association Rules)")
st.info("Identifies products that are frequently purchased together. Useful for product placement and cross-selling strategies.")

# Prepare data for Apriori
# Group by Invoice ID and list products in each transaction
basket = (df_filtered.groupby(['Invoice ID', 'Product line'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Invoice ID'))

# Convert quantities to boolean (presence/absence)
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.map(encode_units) # Use map instead of applymap
if not basket_sets.empty and basket_sets.sum().sum() > 0: # Check if there's any data to analyze
    st.subheader("Top Association Rules:")

    min_support = st.slider("Minimum Support:", min_value=0.01, max_value=0.2, value=0.05, step=0.01,
                            help="Minimum frequency of an itemset in transactions.")
    min_confidence = st.slider("Minimum Confidence:", min_value=0.1, max_value=1.0, value=0.7, step=0.05,
                                help="Likelihood that item Y is purchased when item X is purchased.")

    try:
        with st.spinner('Running Apriori algorithm...'):
            frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        if not rules.empty:
            # Sort by lift for interesting rules
            rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

            st.info("""
            **Understanding Association Rules:**
            - **Antecedents:** Products bought.
            - **Consequents:** Products likely to be bought *with* antecedents.
            - **Support:** How frequently the itemset (antecedents + consequents) appears.
            - **Confidence:** Probability of buying consequents given antecedents.
            - **Lift:** How much more likely consequents are bought given antecedents, relative to their independent probability. (Lift > 1 indicates a positive association).
            """)
        else:
            st.warning("No association rules found with the current minimum support and confidence. Try lowering them.")
    except Exception as e:
        st.error(f"Error performing Market Basket Analysis: {e}. This might happen if there are too few transactions or products after filtering.")
        st.warning("Try broadening your filters if you're not seeing any rules.")
else:
    st.warning("No transactions available for Market Basket Analysis with current filters. Adjust your product line or date range.")


# --- Insights & Recommendations ---
st.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True)
st.markdown("## 🎯 Business Insights & Sales Strategies")
st.success("""
- **Dynamic Product Placement:** Use **Market Basket Analysis** insights to place frequently bought together items closer to each other to encourage impulse purchases and increase average transaction value.
- **Targeted Promotions:** Leverage **Customer Segmentation** to create personalized marketing campaigns. For example, offer discounts on high-margin products to high-spending customers, or loyalty bonuses to frequent but low-spending customers.
- **Proactive Inventory Management:** Utilize **Sales Forecasting** to optimize stock levels, especially for popular or seasonal items, reducing the risk of stockouts for high-demand products and minimizing waste for perishable goods.
- **Optimized Staffing:** Use **Sales by Hour/Day of Week** data to predict peak hours and days, ensuring adequate staff on the floor to improve customer service and reduce wait times.
- **Strategic Expansion:** Analyze **Gross Income by City/Product Line** to identify underperforming areas or categories that need special attention (e.g., promotional pushes, new product introductions).
- **Payment Method Incentives:** Based on the **Payment Method Distribution**, consider offering incentives for less popular but more efficient payment methods (e.g., credit card cashback).
""")

st.markdown("""
<style>
.stButton>button {
    background-color: #2E86C1;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
}
.stDownloadButton>button {
    background-color: #2E86C1;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
}
.metric-row {
    display: flex;
    justify-content: space-around;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)