import streamlit as st
import pandas as pd
import plotly.express as px


# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="E-Commerce Customer Analytics",
    page_icon="🛒",
    layout="wide"
)


# ---------------------------------------
# LOAD DATA
# ---------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("output/customer_segments.csv")
    return data


df = load_data()


# ---------------------------------------
# TITLE
# ---------------------------------------

st.title("🛒 E-Commerce Customer Segmentation & Customer Behavior Analytics")

st.write(
    "Analyze customer purchasing behavior and customer segments using Machine Learning."
)


# ---------------------------------------
# SIDEBAR
# ---------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Customer Segmentation",
        "Customer Behavior",
        "Customer Search"
    ]
)


# =======================================
# DASHBOARD PAGE
# =======================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    total_customers = df["CustomerID"].nunique()

    total_revenue = df["Monetary"].sum()

    vip_customers = len(
        df[df["Customer_Segment"] == "VIP Customers"]
    )

    total_orders = df["Total_Orders"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        total_customers
    )

    col2.metric(
        "Total Revenue",
        f"£{total_revenue:,.2f}"
    )

    col3.metric(
        "VIP Customers",
        vip_customers
    )

    col4.metric(
        "Total Orders",
        int(total_orders)
    )

    st.divider()

    # ---------------------------------------
    # SEGMENT CHART
    # ---------------------------------------

    segment_count = (
        df["Customer_Segment"]
        .value_counts()
        .reset_index()
    )

    segment_count.columns = [
        "Customer Segment",
        "Number of Customers"
    ]

    fig = px.bar(
        segment_count,
        x="Customer Segment",
        y="Number of Customers",
        title="Customer Segment Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =======================================
# CUSTOMER SEGMENTATION PAGE
# =======================================

elif page == "Customer Segmentation":

    st.header("👥 Customer Segmentation")

    st.write(
        "Customers are grouped based on Recency, Frequency and Monetary value."
    )

    segment = st.selectbox(
        "Select Customer Segment",
        ["All"] + list(
            df["Customer_Segment"].dropna().unique()
        )
    )

    if segment == "All":
        filtered_data = df
    else:
        filtered_data = df[
            df["Customer_Segment"] == segment
        ]

    st.dataframe(
        filtered_data,
        use_container_width=True
    )

    # ---------------------------------------
    # SEGMENT SUMMARY
    # ---------------------------------------

    cluster_summary = (
        filtered_data
        .groupby("Customer_Segment")[
            ["Recency", "Frequency", "Monetary"]
        ]
        .mean()
        .reset_index()
    )

    st.subheader("Segment Summary")

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )


# =======================================
# CUSTOMER BEHAVIOR PAGE
# =======================================

elif page == "Customer Behavior":

    st.header("🛍️ Customer Purchase Behavior")

    fig = px.scatter(
        df,
        x="Frequency",
        y="Monetary",
        color="Customer_Segment",
        size="Total_Orders",
        hover_data=[
            "CustomerID",
            "Recency",
            "Average_Order_Value"
        ],
        title="Customer Frequency vs Spending"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------
    # TOP CUSTOMERS
    # ---------------------------------------

    st.subheader("🏆 Top 10 Customers by Spending")

    top_customers = (
        df
        .sort_values(
            "Monetary",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        top_customers,
        use_container_width=True
    )


# =======================================
# CUSTOMER SEARCH PAGE
# =======================================

elif page == "Customer Search":

    st.header("🔍 Customer Search")

    customer_id = st.selectbox(
        "Select Customer ID",
        df["CustomerID"]
        .dropna()
        .sort_values()
        .unique()
    )

    customer_data = df[
        df["CustomerID"] == customer_id
    ]

    st.subheader("Customer Details")

    st.dataframe(
        customer_data,
        use_container_width=True
    )

    if not customer_data.empty:

        row = customer_data.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Customer Segment",
            row["Customer_Segment"]
        )

        col2.metric(
            "Total Spending",
            f"£{row['Monetary']:,.2f}"
        )

        col3.metric(
            "Total Orders",
            int(row["Total_Orders"])
        )