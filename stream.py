import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Intelligence",
    page_icon="⚙️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #07111f, #0d1b2a);
    color: white;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.hero-title {
    font-size: 52px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 15px;
}

.hero-gradient {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-text {
    font-size: 18px;
    color: #aab6c5;
    line-height: 1.7;
    max-width: 700px;
}

.feature-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 18px;
    margin-top: 15px;
    min-height: 150px;
}

.feature-card p {
    color: #aab6c5;
}

.safe-result {
    background: rgba(0, 200, 120, 0.12);
    border: 1px solid rgba(0, 200, 120, 0.4);
    padding: 25px;
    border-radius: 18px;
}

.failure-result {
    background: rgba(255, 70, 70, 0.12);
    border: 1px solid rgba(255, 70, 70, 0.4);
    padding: 25px;
    border-radius: 18px;
}

.risk-low {
    background: rgba(0, 200, 120, 0.12);
    border: 1px solid rgba(0, 200, 120, 0.35);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.risk-high {
    background: rgba(255, 80, 80, 0.12);
    border: 1px solid rgba(255, 80, 80, 0.35);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.small-text {
    color: #8796a8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "Maintenance Preditive Intelligence.pkl"
    )

try:
    model_pipeline = load_model()

except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

left, right = st.columns([1.3, 1])

with left:

    st.markdown("""
    <div class="hero-title">
        Predict Machine Failures
        <span class="hero-gradient">
        Before They Happen
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-text">
        Predictive Maintenance Intelligence uses machine learning
        to analyze machine operating conditions and estimate the
        probability of failure.
        <br><br>
        Enter the machine parameters below to receive an instant
        predictive maintenance assessment.
    </div>
    """, unsafe_allow_html=True)

with right:

    st.markdown("""
    <div class="feature-card">

    <h2>⚙️ AI Maintenance Engine</h2>

    <p>
        The system analyzes machine temperature,
        speed, torque, tool wear and engineered
        maintenance indicators.
    </p>

    <br>

    <b>Instant Failure Analysis</b>

    </div>
    """, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

st.subheader("Predictive Maintenance Features")

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown("""
    <div class="feature-card">

    <h3>🔍 Early Detection</h3>

    <p>
        Detect potential machine failure before
        expensive breakdowns occur.
    </p>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="feature-card">

    <h3>📊 ML Analysis</h3>

    <p>
        Uses machine learning and engineered
        operational features for prediction.
    </p>

    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="feature-card">

    <h3>⚡ Instant Result</h3>

    <p>
        Receive failure probability and risk
        indicators instantly.
    </p>

    </div>
    """, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# MACHINE INPUT FORM
# --------------------------------------------------

st.header("Machine Failure Prediction")

st.write(
    "Enter the current operating conditions of the machine."
)

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        UDI = st.number_input(
            "UDI",
            min_value=1,
            value=1,
            step=1
        )

        Product_ID = st.text_input(
            "Product ID",
            value="M14860"
        )

        Type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        Air_temperature_K = st.number_input(
            "Air Temperature (K)",
            value=298.1,
            step=0.1
        )

    with col2:

        Process_temperature_K = st.number_input(
            "Process Temperature (K)",
            value=308.6,
            step=0.1
        )

        Rotational_speed_rpm = st.number_input(
            "Rotational Speed (RPM)",
            value=1551.0,
            step=1.0
        )

        Torque_Nm = st.number_input(
            "Torque (Nm)",
            value=42.8,
            step=0.1
        )

        Tool_wear_min = st.number_input(
            "Tool Wear (Minutes)",
            value=0.0,
            step=1.0
        )

    submit_button = st.form_submit_button(
        "Analyze Machine"
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submit_button:

    try:

        # --------------------------------------------------
        # RAW DATA
        # --------------------------------------------------

        raw_data = {

            "UDI": [UDI],

            "Product ID": [Product_ID],

            "Type": [Type],

            "Air temperature [K]": [
                Air_temperature_K
            ],

            "Process temperature [K]": [
                Process_temperature_K
            ],

            "Rotational speed [rpm]": [
                Rotational_speed_rpm
            ],

            "Torque [Nm]": [
                Torque_Nm
            ],

            "Tool wear [min]": [
                Tool_wear_min
            ],

            "HDF_Risk": [0],

            "Failure_Risk": [0]
        }

        df = pd.DataFrame(raw_data)

        # --------------------------------------------------
        # FEATURE ENGINEERING
        # --------------------------------------------------

        df["temperature_difference"] = (
            df["Process temperature [K]"]
            -
            df["Air temperature [K]"]
        )

        df["Power"] = (
            df["Rotational speed [rpm]"]
            *
            df["Torque [Nm]"]
            *
            2
            *
            (3.12 / 60)
        )

        df["Energy Loss"] = (
            df["Rotational speed [rpm]"]
            *
            df["Tool wear [min]"]
        )

        df["Overstrain_Index"] = (
            df["Torque [Nm]"]
            *
            df["Tool wear [min]"]
        )

        # --------------------------------------------------
        # CALCULATE FAILURE RISK
        # --------------------------------------------------

        df["Failure_Risk"] = 0

        df.loc[
            (
                df["Power"] < 3500
            )
            |
            (
                df["Power"] > 9000
            ),
            "Failure_Risk"
        ] = 1

        # --------------------------------------------------
        # CALCULATE HDF RISK
        # --------------------------------------------------

        df["HDF_Risk"] = (
            (
                df["temperature_difference"] < 8.6
            )
            &
            (
                df["Rotational speed [rpm]"] < 1380
            )
        ).astype(int)

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        prediction = model_pipeline.predict(df)[0]

        probabilities = (
            model_pipeline.predict_proba(df)[0]
        )

        no_failure_probability = (
            float(probabilities[0]) * 100
        )

        failure_probability = (
            float(probabilities[1]) * 100
        )

        failure_risk = int(
            df["Failure_Risk"].iloc[0]
        )

        hdf_risk = int(
            df["HDF_Risk"].iloc[0]
        )

        # --------------------------------------------------
        # FINAL PREDICTION RESULT
        # --------------------------------------------------

        st.divider()

        st.header("Prediction Result")

        if prediction == 1:

            st.markdown("""
            <div class="failure-result">

            <h2>⚠️ Machine Failure Detected</h2>

            <p>
                The machine learning model has detected
                machine conditions associated with failure.
            </p>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="safe-result">

            <h2>✅ No Failure Detected</h2>

            <p>
                Current machine conditions appear normal
                according to the prediction model.
            </p>

            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # --------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------

        p1, p2 = st.columns(2)

        with p1:

            st.metric(
                "No Failure Probability",
                f"{no_failure_probability:.2f}%"
            )

        with p2:

            st.metric(
                "Failure Probability",
                f"{failure_probability:.2f}%"
            )

        st.subheader("Failure Probability")

        st.progress(
            min(
                max(
                    failure_probability / 100,
                    0
                ),
                1
            )
        )

        # --------------------------------------------------
        # RISK INDICATORS
        # --------------------------------------------------

        st.divider()

        st.header("Risk Indicators")

        risk1, risk2 = st.columns(2)

        with risk1:

            if failure_risk == 1:

                st.markdown("""
                <div class="risk-high">

                <h3>⚠️ Failure Risk</h3>

                <h2>HIGH</h2>

                <p>
                    Machine power is outside
                    the expected operating range.
                </p>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="risk-low">

                <h3>✅ Failure Risk</h3>

                <h2>LOW</h2>

                <p>
                    Machine power is inside
                    the expected operating range.
                </p>

                </div>
                """, unsafe_allow_html=True)

        with risk2:

            if hdf_risk == 1:

                st.markdown("""
                <div class="risk-high">

                <h3>🔥 HDF Risk</h3>

                <h2>HIGH</h2>

                <p>
                    Heat dissipation failure
                    conditions were detected.
                </p>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="risk-low">

                <h3>✅ HDF Risk</h3>

                <h2>LOW</h2>

                <p>
                    Heat dissipation conditions
                    appear normal.
                </p>

                </div>
                """, unsafe_allow_html=True)

        # --------------------------------------------------
        # MACHINE ANALYSIS
        # --------------------------------------------------

        st.divider()

        st.subheader("Machine Analysis")

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Temperature Difference",
                f"{df['temperature_difference'].iloc[0]:.2f} K"
            )

        with m2:

            st.metric(
                "Power",
                f"{df['Power'].iloc[0]:.2f}"
            )

        with m3:

            st.metric(
                "Energy Loss",
                f"{df['Energy Loss'].iloc[0]:.2f}"
            )

        with m4:

            st.metric(
                "Overstrain Index",
                f"{df['Overstrain_Index'].iloc[0]:.2f}"
            )

        # --------------------------------------------------
        # FULL DATA
        # --------------------------------------------------

        with st.expander(
            "View Complete Processed Machine Data"
        ):

            st.dataframe(
                df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div style="
text-align:center;
color:#657386;
padding:25px;
">

Predictive Maintenance Intelligence
<br>
Machine Learning Powered Predictive Maintenance System

</div>
""", unsafe_allow_html=True)