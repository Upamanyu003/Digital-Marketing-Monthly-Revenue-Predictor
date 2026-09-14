from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


APP_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = APP_DIRECTORY / "model.pkl"
METADATA_PATH = APP_DIRECTORY / "feature_metadata.json"


st.set_page_config(
    page_title="Digital Marketing Monthly Revenue Predictor",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem;}
    h1, h2, h3 {color: #16324F;}
    div[data-testid="stMetric"] {
        background: #F4F7FA;
        border: 1px solid #D9E2EC;
        border-radius: 10px;
        padding: 1rem;
    }
    /* Keep the result readable when Streamlit is using its dark theme. */
    div[data-testid="stMetric"] * {
        color: #16324F !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path):
    """Load the fitted pipeline once and reuse it across Streamlit reruns."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"The fitted model file was not found at {model_path}."
        )

    loaded_model = joblib.load(model_path)
    if not hasattr(loaded_model, "predict"):
        raise TypeError("The loaded object does not provide a predict method.")
    return loaded_model


def load_metadata(metadata_path):
    """Load training-derived feature names, categories, and input ranges."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Feature metadata was not found at {metadata_path}."
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_keys = {
        "feature_names",
        "categorical_features",
        "numeric_features",
    }
    missing_keys = required_keys - set(metadata)
    if missing_keys:
        raise ValueError(
            f"Feature metadata is missing required keys: {sorted(missing_keys)}"
        )
    return metadata


st.title("Digital Marketing Monthly Revenue Predictor")
st.caption(
    "Estimate monthly campaign-attributed revenue from planned campaign inputs. "
    "All monetary values are in Indian rupees."
)

try:
    model = load_model(MODEL_PATH)
    metadata = load_metadata(METADATA_PATH)
except Exception as error:
    st.error(
        "The prediction resources could not be loaded. Run the completed "
        "model_training.ipynb to recreate model.pkl and feature_metadata.json."
    )
    st.exception(error)
    st.stop()


category_metadata = metadata["categorical_features"]
numeric_metadata = metadata["numeric_features"]

st.subheader("Campaign profile")
category_col1, category_col2 = st.columns(2)

with category_col1:
    campaign_channel = st.selectbox(
        "Campaign channel",
        options=category_metadata["Campaign_Channel"]["values"],
        help="Primary channel used to run the campaign.",
    )

with category_col2:
    target_segment = st.selectbox(
        "Target segment",
        options=category_metadata["Target_Segment"]["values"],
        help="Customer group the campaign is intended to reach.",
    )

st.subheader("Campaign inputs")
input_col1, input_col2, input_col3 = st.columns(3)


def numeric_limits(column):
    """Return the minimum, maximum, and median observed in training data."""
    values = numeric_metadata[column]
    return values["minimum"], values["maximum"], values["median"]


ad_min, ad_max, ad_default = numeric_limits("Ad_Spend")
impression_min, impression_max, impression_default = numeric_limits("Impressions")
click_min, click_max, click_default = numeric_limits("Clicks")
conversion_min, conversion_max, conversion_default = numeric_limits("Conversion_Rate")
aov_min, aov_max, aov_default = numeric_limits("Average_Order_Value")
duration_min, duration_max, duration_default = numeric_limits("Campaign_Duration_Days")
open_min, open_max, open_default = numeric_limits("Email_Open_Rate")

with input_col1:
    ad_spend = st.number_input(
        "Ad spend (₹)",
        min_value=float(max(0, ad_min)),
        max_value=float(ad_max),
        value=float(ad_default),
        step=1_000.0,
        format="%.2f",
    )
    impressions = st.number_input(
        "Impressions",
        min_value=max(1, int(impression_min)),
        max_value=int(impression_max),
        value=int(impression_default),
        step=1_000,
    )
    clicks = st.number_input(
        "Clicks",
        min_value=max(1, int(click_min)),
        max_value=min(int(click_max), int(impressions)),
        value=min(int(click_default), int(impressions)),
        step=100,
    )

with input_col2:
    conversion_rate = st.number_input(
        "Conversion rate (%)",
        min_value=float(max(0, conversion_min)),
        max_value=float(min(100, conversion_max)),
        value=float(conversion_default),
        step=0.10,
        format="%.2f",
    )
    average_order_value = st.number_input(
        "Average order value (₹)",
        min_value=float(max(0, aov_min)),
        max_value=float(aov_max),
        value=float(aov_default),
        step=100.0,
        format="%.2f",
    )
    campaign_duration_days = st.number_input(
        "Campaign duration (days)",
        min_value=max(1, int(duration_min)),
        max_value=int(duration_max),
        value=int(duration_default),
        step=1,
    )

with input_col3:
    email_open_rate = st.number_input(
        "Email open rate (%)",
        min_value=float(max(0, open_min)),
        max_value=float(min(100, open_max)),
        value=float(open_default),
        step=0.10,
        format="%.2f",
    )
    st.info(
        "Input limits reflect the training-data range. Predictions outside these "
        "ranges would be extrapolations and are intentionally unavailable."
    )

validation_errors = []

if ad_spend < 0 or average_order_value < 0:
    validation_errors.append("Monetary values cannot be negative.")
if impressions <= 0 or clicks <= 0:
    validation_errors.append("Impressions and clicks must both be positive.")
if clicks > impressions:
    validation_errors.append("Clicks cannot exceed impressions.")
if not 0 <= conversion_rate <= 100:
    validation_errors.append("Conversion rate must be between 0 and 100 percent.")
if not 0 <= email_open_rate <= 100:
    validation_errors.append("Email open rate must be between 0 and 100 percent.")
if campaign_duration_days <= 0:
    validation_errors.append("Campaign duration must be positive.")

for validation_error in validation_errors:
    st.warning(validation_error)

predict_selected = st.button(
    "Predict monthly revenue",
    type="primary",
    disabled=bool(validation_errors),
    use_container_width=True,
)

if predict_selected:
    input_record = pd.DataFrame(
        [
            {
                "Campaign_Channel": str(campaign_channel),
                "Target_Segment": str(target_segment),
                "Ad_Spend": float(ad_spend),
                "Impressions": int(impressions),
                "Clicks": int(clicks),
                "Conversion_Rate": float(conversion_rate),
                "Average_Order_Value": float(average_order_value),
                "Campaign_Duration_Days": int(campaign_duration_days),
                "Email_Open_Rate": float(email_open_rate),
            }
        ],
        columns=metadata["feature_names"],
    )

    try:
        predicted_revenue = float(model.predict(input_record)[0])
    except Exception as error:
        st.error(
            "A prediction could not be generated. The saved model may be "
            "incompatible with the current application or input schema."
        )
        st.exception(error)
    else:
        st.subheader("Estimated result")
        st.metric(
            "Estimated monthly revenue",
            f"₹{predicted_revenue:,.2f}",
        )

        revenue_to_spend = predicted_revenue / ad_spend if ad_spend > 0 else None
        if revenue_to_spend is not None:
            st.write(
                f"The estimate is approximately **{revenue_to_spend:.2f} times** "
                "the selected advertising spend. This is a revenue-to-spend "
                "comparison, not a profit margin or causal ROI calculation."
            )

with st.expander("How to interpret this estimate"):
    st.write(
        "The result is a statistical estimate based on patterns in the synthetic "
        "training dataset. It is not guaranteed revenue and does not prove that "
        "changing one input will cause revenue to change by the modelled amount. "
        "Forecast uncertainty, market conditions, attribution quality, costs, and "
        "campaign execution should also inform decisions."
    )
