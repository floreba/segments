import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Setup ---
st.set_page_config(layout="wide", page_title="User Segmentation Dashboard")

# --- Load Data ---
df = pd.read_csv("clustered_users.csv")

# --- Segment Labels & Descriptions ---
segment_labels = {
    0: "Passive Buyers",
    1: "Active Buyers",
    2: "Passive Visitors",
    3: "Active Visitors"
}

segment_descriptions = {
    "Passive Buyers": "Selective buyers. Low frequency and variety. Likely respond to promotions or specific product needs. "
                      "Focus on subscription models and discounts to further activate.",
    "Active Buyers": "High purchase activity and product variety. Key for upselling and loyalty programs. "
                     "Focus on loyalty programs.",
    "Passive Visitors": "Low activity and no conversions. Likely just browsing or waiting. "
                        "Consider improved homepage entry, more engaging landing pages, CTAs, and content.",
    "Active Visitors": "Highly engaged users, but haven’t purchased. May be stuck in the funnel or still evaluating. "
                       "Study interested products and consider re-engagement strategies (e.g. checkout reminder alerts)."
}

df["segment_name"] = df["segment"].map(segment_labels)

# --- Sidebar ---
st.sidebar.header("🔧 Filter Options")
segments = sorted(df["segment_name"].unique())
selected_segments = st.sidebar.multiselect("Select Segments", segments, default=segments)

# Optional category filter
#category_filter = None
#if "dominant_category" in df.columns:
#    category_list = sorted(df["dominant_category"].dropna().unique())
#    category_filter = st.sidebar.multiselect("Filter by Dominant Category", category_list)

# Segment descriptions
st.sidebar.markdown("### 🧠 Segment Descriptions")
for seg in selected_segments:
    st.sidebar.markdown(f"**{seg}**: {segment_descriptions.get(seg, '')}")

# --- Filter Data ---
filtered_df = df[df["segment_name"].isin(selected_segments)]
if category_filter:
    filtered_df = filtered_df[filtered_df["dominant_category"].isin(category_filter)]

# --- Main Title ---
st.title("📊 User Segmentation Dashboard")

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["Segment Overview", "📈 Feature Explorer", "🏷️ Top Categories per Segment"])

# --- Tab 1: Segment Overview ---
# --- Tab 1: Segment Overview ---
with tab1:
    st.subheader("Segment Sizes")
    seg_counts = filtered_df["segment_name"].value_counts().sort_index()
    total_users = seg_counts.sum()
    seg_percent = (seg_counts / total_users * 100).round(1)

    # FIX: Build clean DataFrame with explicit column names
    summary_df = pd.DataFrame({
        "Segment": seg_counts.index,
        "User Count": seg_counts.values,
        "Share (%)": seg_percent.values
    })

    # -- Horizontal Bar Chart with Labels --
    fig_height = 0.7 * len(summary_df) + 2  # Adjust height dynamically
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(summary_df["Segment"], summary_df["User Count"], color="steelblue")

    for i, (count, pct) in enumerate(zip(summary_df["User Count"], summary_df["Share (%)"])):
        ax.text(count + total_users * 0.01, i, f"{pct}%", va='center', fontsize=8)

    ax.set_xlabel("Users")
    ax.set_xlim(0, summary_df["User Count"].max() * 1.2)
    ax.set_title("User Distribution by Segment")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # -- Feature Summary Table BELOW the bar chart --
    st.subheader("Segment Feature Averages")
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in ["segment"]
    ]
    feature_summary = (
        filtered_df.groupby("segment_name")[numeric_cols]
        .mean()
        .round(2)
        .reset_index()
    )
    st.dataframe(feature_summary)


# --- Tab 2: Feature Distribution ---
with tab2:
    st.subheader("Feature Distribution by Segment")

    feature_cols = [
        col for col in df.columns
        if col not in ["user_id", "segment", "segment_name", "dominant_category", "dominant_device"]
        and df[col].dtype in [np.float64, np.int64]
    ]

    selected_feature = st.radio(
        "Choose a feature to visualize",
        options=feature_cols,
        horizontal=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    all_values = filtered_df[selected_feature].dropna()
    bin_edges = np.histogram_bin_edges(all_values, bins=30)

    for seg_name in sorted(filtered_df["segment_name"].unique()):
        subset = filtered_df[filtered_df["segment_name"] == seg_name][selected_feature].dropna()
        weights = np.ones(len(subset)) / len(subset) * 100
        ax.hist(subset, bins=bin_edges, weights=weights, alpha=0.6, label=seg_name)

    ax.set_title(f"{selected_feature} – Histogram by Segment")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("% of users")
    ax.legend()
    st.pyplot(fig)

# --- Tab 3: Top Categories ---
with tab3:
    st.subheader("Top engaged Product Categories per Segment [event count]")

    if "dominant_category" in filtered_df.columns:
        top_cats = (
            filtered_df.groupby(["segment_name", "dominant_category"])
            .size()
            .reset_index(name="count")
            .sort_values(["segment_name", "count"], ascending=[True, False])
        )

        for seg_name in sorted(filtered_df["segment_name"].unique()):
            st.markdown(f"### {seg_name}")
            top_n = top_cats[top_cats["segment_name"] == seg_name].head(5)
            st.dataframe(top_n)
    else:
        st.info("🛈 'dominant_category' column is missing in your dataset.")
