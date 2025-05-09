import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(
    page_title='Fraud Detection',
    page_icon=':gem:',
    layout='wide'
)

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

model = joblib.load(open(r"D:\python\fraudGui", 'rb'))

def predict(features):
    prediction = model.predict(features)
    return prediction

@st.cache
def load_data():
    data = pd.read_excel('fraudTest.xlsx') 
    return data

data = load_data()

with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Home", "Predict Fraud"],
        icons=["house", "clipboard-data"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    st.write("# Fraud Detection")
    st_lottie_animation = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
    if st_lottie_animation:
        st_lottie(st_lottie_animation, height=300)
    st.markdown("Welcome to the fraud detection app. Navigate through the menu to explore data and make predictions.")

elif selected == "Predict Fraud":
    st.write("# Fraud Detection")
    st.subheader("Enter Your Details to Classify Your Transaction")

    time = st.text_input("Enter Time (YYYY-MM-DD HH:MM:SS)", "2023-01-01 00:00:00")
    CardNumber = st.number_input("Enter Your Card Number", min_value=0)
    merchant = st.text_input("Enter Your Merchant Name", "Type Here ...")
    category = st.selectbox("Category: ",
                    ['misc_net', 'grocery_pos', 'entertainment' ,'gas_transport' ,'misc_pos',
                     'grocery_net', 'shopping_net' ,'shopping_pos' ,'food_dining', 'personal_care',
                     'health_fitness' ,'travel' ,'kids_pets' ,'home'])
    firstName = st.text_input("Enter Your First Name", "Type Here ...")
    lastName = st.text_input("Enter Your Last Name", "Type Here ...")
    amount = st.number_input("Enter Your Amount", min_value=0.0, step=0.01)
    trans = st.text_input("Enter Your Transaction Number", "Type Here ...")

    # Prepare the data for prediction
    try:
        data = pd.DataFrame({
            'Time': [pd.to_datetime(time)],
            'Card Number': [CardNumber],
            'merchant': [merchant],
            'category': [category],
            'Amount': [amount],
            'firstName': [firstName],
            'lastName': [lastName],
            'trans_num': [trans],
        })
    except ValueError as e:
        st.error(f"Error in date format: {e}")
        st.stop()

    # Feature engineering
    data['Time_Seconds'] = (data['Time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    data['time_diff_prev'] = data.groupby('Card Number')['Time'].diff().dt.total_seconds().fillna(0)
    data['amount_diff_card'] = data['Amount'] - data.groupby('Card Number')['Amount'].transform('mean')
    data['amount_diff_cat'] = data['Amount'] - data.groupby('category')['Amount'].transform('mean')
    data['amount_diff_mer'] = data['Amount'] - data.groupby('merchant')['Amount'].transform('mean')
    data['transactions_last_hour'] = data.groupby('Card Number')['Time'].transform(lambda x: x.diff().lt(pd.Timedelta(hours=1)).cumsum())

    data = pd.get_dummies(data, columns=['category'], prefix='category', drop_first=False)
    label = LabelEncoder()
    scaler = StandardScaler()
    data['fullName'] = data['firstName'] + data['lastName']
    data['fullName'] = label.fit_transform(data['fullName'].astype(str))
    data['fullName'] = scaler.fit_transform(data[['fullName']])
    
    data = data.drop(['Time', 'merchant', 'firstName', 'lastName', 'trans_num'], axis=1)

    # Ensure all expected dummy columns are present
    expected_columns = ['Card Number', 'Amount', 'Time_Seconds', 'time_diff_prev', 'amount_diff_card',
                        'amount_diff_cat', 'amount_diff_mer', 'transactions_last_hour', 'category_entertainment',
                        'category_food_dining', 'category_gas_transport', 'category_grocery_net', 'category_grocery_pos',
                        'category_health_fitness', 'category_home', 'category_kids_pets', 'category_misc_net',
                        'category_misc_pos', 'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
                        'category_travel', 'fullName']

    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[expected_columns]

    # Normalize the data
    for col in expected_columns:
        data[col] = scaler.fit_transform(data[[col]])

    features = data.to_numpy()

    if st.button('Detect'):
        prediction = predict(features)
        if prediction[0] == 1:
            st.error("Fraud detected!")
        else:
            st.success("No fraud detected.")
