
import streamlit as st
import pandas as pd
import joblib

# Load trained model
MODEL_PATH = "tourism_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="🏨",
    layout="wide"
)

st.title("Wellness Tourism Package Prediction")
st.write(
    "Enter customer and interaction details to predict "
    "the likelihood of purchasing the Wellness Tourism Package."
)

st.subheader("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2
    )

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Divorced", "Single"]
    )

with col3:
    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=30,
        value=3
    )

    passport = st.selectbox(
        "Passport",
        [0, 1]
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1]
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

st.subheader("Interaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=25000
    )

with col2:
    pitch_satisfaction_score = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Deluxe",
            "Standard",
            "Super Deluxe",
            "King"
        ]
    )

with col3:
    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0,
        max_value=20,
        value=3
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        max_value=120,
        value=15
    )

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "ProductPitched": [product_pitched],
    "NumberOfFollowups": [number_of_followups],
    "DurationOfPitch": [duration_of_pitch]
})

if st.button("Predict", type="primary"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The customer is likely to purchase the Wellness Tourism Package."
        )
    else:
        st.info(
            "The customer is unlikely to purchase the Wellness Tourism Package."
        )

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )
