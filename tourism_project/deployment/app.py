
import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Package Purchase Prediction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "Tourism Package Purchase Prediction"
)

st.write(
    "Enter customer information to predict whether "
    "the customer is likely to purchase the tourism package."
)


# ============================================================
# MODEL PATHS
#
# The deployment package contains:
#
# deployment/
# ├── app.py
# ├── final_model.pkl
# └── preprocessor.pkl
#
# Therefore the files are loaded relative to app.py.
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "final_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "preprocessor.pkl"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return model, preprocessor


try:

    model, preprocessor = load_artifacts()

except Exception as e:

    st.error(
        "Unable to load the trained model or preprocessor."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header(
    "Customer Information"
)

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    DurationOfPitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        max_value=200,
        value=10,
        step=1
    )

    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    NumberOfFollowups = st.number_input(
        "Number of Followups",
        min_value=0,
        max_value=50,
        value=3,
        step=1
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    NumberOfTrips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=100,
        value=3,
        step=1
    )

    Passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    PitchSatisfactionScore = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    OwnCar = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=25000.0,
        step=1000.0
    )

    TypeofContact = st.selectbox(
        "Type of Contact",
        [
            "Self Enquiry",
            "Company Invited"
        ]
    )

    Occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    Gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    ProductPitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Deluxe",
            "Standard",
            "Super Deluxe",
            "King"
        ]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Married",
            "Divorced",
            "Unmarried",
            "Single"
        ]
    )

    Designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")

if st.button(
    "Predict Package Purchase",
    type="primary"
):

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "Age": Age,
                "CityTier": CityTier,
                "DurationOfPitch": DurationOfPitch,
                "NumberOfPersonVisiting": NumberOfPersonVisiting,
                "NumberOfFollowups": NumberOfFollowups,
                "PreferredPropertyStar": PreferredPropertyStar,
                "NumberOfTrips": NumberOfTrips,
                "Passport": Passport,
                "PitchSatisfactionScore": PitchSatisfactionScore,
                "OwnCar": OwnCar,
                "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
                "MonthlyIncome": MonthlyIncome,
                "TypeofContact": TypeofContact,
                "Occupation": Occupation,
                "Gender": Gender,
                "ProductPitched": ProductPitched,
                "MaritalStatus": MaritalStatus,
                "Designation": Designation
            }
        ]
    )


    # --------------------------------------------------------
    # Exact feature order used during model training
    # --------------------------------------------------------

    feature_order = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "MonthlyIncome",
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation"
    ]

    input_data = input_data[
        feature_order
    ]


    # --------------------------------------------------------
    # Apply saved preprocessing
    # --------------------------------------------------------

    try:

        processed_data = preprocessor.transform(
            input_data
        )

    except Exception as e:

        st.error(
            "Error while preprocessing the customer data."
        )

        st.exception(e)

        st.stop()


    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    prediction = model.predict(
        processed_data
    )[0]


    # --------------------------------------------------------
    # Generate probability
    # --------------------------------------------------------

    probability = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probability = model.predict_proba(
            processed_data
        )[0][1]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader(
        "Prediction Result"
    )


    if prediction == 1:

        st.success(
            "Prediction: Customer is likely to purchase "
            "the tourism package."
        )

    else:

        st.info(
            "Prediction: Customer is unlikely to purchase "
            "the tourism package."
        )


    if probability is not None:

        st.metric(
            "Purchase Probability",
            f"{probability * 100:.2f}%"
        )


    # --------------------------------------------------------
    # Display submitted customer information
    # --------------------------------------------------------

    with st.expander(
        "View Customer Input"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Tourism Package Purchase Prediction | "
    "Machine Learning Deployment"
)
