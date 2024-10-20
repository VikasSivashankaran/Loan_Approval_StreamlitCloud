import streamlit as st
from pyspark.sql import SparkSession
import numpy as np
import random
from pyspark.sql.functions import avg, col, expr
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Loan Prediction Streamlit App") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

# Load the dataset
df = spark.read.csv("loan_approval_dataset.csv", header=True, inferSchema=True)

# Trim leading and trailing spaces from column names
for col_name in df.columns:
    df = df.withColumnRenamed(col_name, col_name.strip())

# Calculate threshold values from the dataset
thresholds = {
    'income_annum': df.select(avg(col("income_annum"))).first()[0],
    'loan_amount': df.select(avg(col("loan_amount"))).first()[0],
    'loan_term': df.select(expr('percentile_approx(loan_term, 0.5)')).first()[0],  # Median
    'cibil_score': df.select(expr('percentile_approx(cibil_score, 0.5)')).first()[0],  # Median
    'residential_assets_value': df.select(avg(col("residential_assets_value"))).first()[0],
    'commercial_assets_value': df.select(avg(col("commercial_assets_value"))).first()[0],
    'luxury_assets_value': df.select(avg(col("luxury_assets_value"))).first()[0],
    'bank_asset_value': df.select(avg(col("bank_asset_value"))).first()[0]
}

# Collect user inputs from Streamlit
st.title("Loan Eligibility Prediction")
user_data = {}
user_data['age'] = st.number_input("Enter your age:", min_value=0, max_value=120, value=30)
user_data['no_of_dependents'] = st.number_input("Enter number of dependents:", min_value=0, value=0)
user_data['education'] = st.selectbox("Select education level:", ("Graduate", "Not Graduate"))
user_data['self_employed'] = st.selectbox("Are you self-employed?", ("Yes", "No"))
user_data['income_annum'] = st.number_input("Enter annual income:", min_value=0, value=50000)
user_data['loan_amount'] = st.number_input("Enter loan amount:", min_value=0, value=100000)
user_data['loan_term'] = st.number_input("Enter loan term (in years):", min_value=1, value=15)
user_data['cibil_score'] = st.number_input("Enter CIBIL score:", min_value=0, max_value=900, value=650)
user_data['residential_assets_value'] = st.number_input("Enter value of residential assets:", min_value=0, value=100000)
user_data['commercial_assets_value'] = st.number_input("Enter value of commercial assets:", min_value=0, value=50000)
user_data['luxury_assets_value'] = st.number_input("Enter value of luxury assets:", min_value=0, value=20000)
user_data['bank_asset_value'] = st.number_input("Enter value of bank assets:", min_value=0, value=30000)

# Define all loan options
loan_options = [
    "Personal Loans",
    "Auto Loans",
    "Conventional Mortgages",
    "FHA Loans",
    "VA Loans",
    "Student Loans",
    "Credit Cards",
    "Small Business Loans",
    "Reverse Mortgages",
    "Home Equity Loans",
    "Home Equity Lines of Credit (HELOC)",
    "Secured Loans",
    "Medicare Advantage Loans"
]

# Display loan options as checkboxes
st.write("Select the loan options you are interested in:")
selected_options = []
for option in loan_options:
    if st.checkbox(option):
        selected_options.append(option)

# Highlight age-appropriate loans
if user_data['age'] < 60:
    st.write("Note: Since you are under 60, the following loans are generally more suitable for you:")
    st.write("- Personal Loans")
    st.write("- Auto Loans")
    st.write("- Conventional Mortgages")
    st.write("- FHA Loans")
    st.write("- VA Loans")
    st.write("- Student Loans")
    st.write("- Credit Cards")
    st.write("- Small Business Loans")
else:
    st.write("Note: Since you are 60 or older, consider the following loans:")
    st.write("- Reverse Mortgages")
    st.write("- Home Equity Loans")
    st.write("- Home Equity Lines of Credit (HELOC)")
    st.write("- Secured Loans")
    st.write("- Medicare Advantage Loans")

# Function to check loan eligibility
def check_loan_eligibility(data, thresholds):
    if data['cibil_score'] < thresholds['cibil_score']:
        return "Rejected due to low CIBIL score", None
    elif data['income_annum'] < thresholds['income_annum']:
        return "Rejected due to low income", None
    elif data['loan_amount'] > thresholds['loan_amount']:
        return "Rejected due to high loan amount", None
    elif data['residential_assets_value'] < thresholds['residential_assets_value']:
        return "Rejected due to low residential assets", None
    elif data['commercial_assets_value'] < thresholds['commercial_assets_value']:
        return "Rejected due to low commercial assets", None
    elif data['luxury_assets_value'] < thresholds['luxury_assets_value']:
        return "Rejected due to low luxury assets", None
    elif data['bank_asset_value'] < thresholds['bank_asset_value']:
        return "Rejected due to low bank assets", None
    else:
        return "Approved", selected_options

# Check loan eligibility
loan_decision, approved_loan_types = check_loan_eligibility(user_data, thresholds)

# Output the loan status
st.write(f"Loan Status: {loan_decision}")

if loan_decision == "Approved" and approved_loan_types:
    st.write("Loan Types Sanctioned:")
    for loan_type in approved_loan_types:
        st.write(f"- {loan_type}")

# Prepare data for visualization
df_pandas = df.toPandas()

st.header("Show 3D Plot")
    # Sample the DataFrame to reduce the number of points plotted
df_sampled = df_pandas.sample(frac=0.01)  # Adjust the fraction as needed (e.g., 0.1 for 10%)
fig_3d = px.line_3d(df_sampled, x='income_annum', y='loan_amount', z='cibil_score',
                         title='3D Line Plot of Income vs Loan Amount vs CIBIL Score')
    
st.plotly_chart(fig_3d)

df_sorted = df_pandas.sort_values(by='income_annum')  # Sort by the desired column

years = np.arange(1, user_data['loan_term'] + 1)  # Create an array of years based on loan term

# Generate loan amounts with fluctuations every 1.5 years
loan_amounts = []
base_loan_amount = user_data['loan_amount']

for year in years:
    # Introduce fluctuations every 1.5 years
    if year % 1.5 == 0:
        fluctuation = random.uniform(-10000, 10000)  # Random fluctuation in loan amount
        new_loan_amount = max(0, base_loan_amount + fluctuation)  # Ensure loan amount doesn't go negative
    else:
        new_loan_amount = base_loan_amount  # Keep loan amount the same for other years
    loan_amounts.append(new_loan_amount)

# Create a DataFrame for plotting
time_lapse_data = pd.DataFrame({
    'Years': years,
    'Loan Amount': loan_amounts
})

# Line Plot using Matplotlib
st.header("Show Loan Amount Over Time")
plt.figure(figsize=(10, 6))
plt.plot(time_lapse_data['Years'], time_lapse_data['Loan Amount'], marker='o')
plt.title('Loan Amount Over Time')
plt.xlabel('Years')
plt.ylabel('Loan Amount')
plt.xticks(years)  # Set x-ticks to be each year
plt.grid()
st.pyplot(plt)
    
# Violin Plot using Seaborn
st.header("Show Violin Plot")
plt.figure(figsize=(10, 6))
sns.violinplot(x='education', y='income_annum', data=df_pandas)
plt.title('Income Distribution by Education Level')
st.pyplot(plt)

# Stop the Spark session when done
spark.stop()
