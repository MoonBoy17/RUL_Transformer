import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
import matplotlib.pyplot as plt
import numpy as np
import random
import hashlib
import seaborn as sns

st.set_page_config(page_title="Interactive Data Analysis", page_icon=":bar_chart:", layout="wide")

class DGA:
    def __init__(self, H2, CH4, C2H6, C2H4, C2H2, CO, CO2):
        self.H2 = H2
        self.CH4 = CH4
        self.C2H6 = C2H6
        self.C2H4 = C2H4
        self.C2H2 = C2H2
        self.CO = CO
        self.CO2 = CO2

class CONC:
    def __init__(self, H2, CH4, C2H6, C2H4, C2H2, CO, CO2, Acidity, Resistance, BDV, Water, tand, Furan, age, ir, olk, wrd):
        self.H2 = H2
        self.CH4 = CH4
        self.C2H6 = C2H6
        self.C2H4 = C2H4
        self.C2H2 = C2H2
        self.CO = CO
        self.CO2 = CO2
        self.Acidity = Acidity
        self.Resistance = Resistance
        self.BDV = BDV
        self.Water = Water
        self.tand = tand
        self.Furan = Furan
        self.age = age
        self.ir = ir
        self.olk = olk
        self.wrd = wrd

class Thresholds:
    def __init__(self, acetylene, ethylene, acidity, bdv, waterContent, windingResDeviation, irValue, oilLeakage, age, furan):
        self.acetylene = acetylene
        self.ethylene = ethylene
        self.acidity = acidity
        self.bdv = bdv
        self.waterContent = waterContent
        self.windingResDeviation = windingResDeviation
        self.irValue = irValue
        self.oilLeakage = oilLeakage
        self.age = age
        self.furan = furan

htTrafo = Thresholds(2.5, 100, 0.2, 40, 30, 8, 500, 3, 50, 3000)
lfTrafo = Thresholds(1.5, 60, 0.2, 35, 35, 2, 100, 3, 50, 3000)
gtConverterTrafo = Thresholds(1.5, 60, 0.2, 35, 35, 2, 100, 3, 50, 3000)
ltTrafo = Thresholds(3.5, 150, 0.3, 30, 50, 10, 100, 4, 50, 3000)

def calculateScore(value, thresholds):
    for threshold, score in thresholds:
        if value <= threshold:
            return score
    return 1

def getAbnormalities(dga, thresholds):
    abnormalities = []
    if dga.C2H2 > thresholds.acetylene:
        abnormalities.append("Acetylene above threshold")
    if dga.C2H4 > thresholds.ethylene:
        abnormalities.append("Ethylene above threshold")
    if dga.Acidity >= thresholds.acidity:
        abnormalities.append("Acidity above threshold")
    if dga.BDV < thresholds.bdv:
        abnormalities.append("BDV below threshold")
    if dga.Water >= thresholds.waterContent:
        abnormalities.append("Water content above threshold")
    if dga.wrd > thresholds.windingResDeviation:
        abnormalities.append("Winding resistance deviation above threshold")
    if dga.ir < thresholds.irValue:
        abnormalities.append("IR value below threshold")
    if dga.olk >= thresholds.oilLeakage:
        abnormalities.append("Oil leakage above threshold")
    if dga.age >= thresholds.age and dga.Furan >= thresholds.furan:
        abnormalities.append("Age >= 50 and Furan above threshold")
    if dga.age < thresholds.age and dga.Furan >= 6000:
        abnormalities.append("Age < 50 and Furan above 6000")
    return abnormalities

def Health_index():
    st.title("Transformer Health Index and Abnormality Detection")

    st.header("Enter DGA Values")
    dga = CONC(
        H2=st.number_input('H2 (ppm):', min_value=0.0),
        CH4=st.number_input('CH4 (ppm):', min_value=0.0),
        C2H6=st.number_input('C2H6 (ppm):', min_value=0.0),
        C2H4=st.number_input('C2H4 (ppm):', min_value=0.0),
        C2H2=st.number_input('C2H2 (ppm):', min_value=0.0),
        CO=st.number_input('CO (ppm):', min_value=0.0),
        CO2=st.number_input('CO2 (ppm):', min_value=0.0),
        Acidity=st.number_input('Acidity (mgKOH/g):', min_value=0.0),
        Resistance=st.number_input('Sp. Resistance at 90 deg C (ohm-cm):', min_value=0.0),
        BDV=st.number_input('Breakdown Voltage (KV):', min_value=0.0),
        Water=st.number_input('Water content (ppm):', min_value=0.0),
        tand=st.number_input('tanδ at 90 deg C:', min_value=0.0),
        Furan=st.number_input('Furan:', min_value=0.0),
        age=st.number_input('Age:', min_value=0.0),
        ir=st.number_input('IR:', min_value=0.0),
        olk=st.number_input('Oil Leakage:', min_value=0.0),
        wrd=st.number_input('Winding Resistance Deviation (HT and LT):', min_value=0.0)
    )

    transformer_type = st.selectbox(
        "Select transformer category:",
        ["HT (≥ 132KV Trafo)", "LF", "GT_CONVERTER", "LT (< 132KV Trafo)"]
    )

    type_to_thresholds = {
        "HT (≥ 132KV Trafo)": htTrafo,
        "LF": lfTrafo,
        "GT_CONVERTER": gtConverterTrafo,
        "LT (< 132KV Trafo)": ltTrafo
    }

    thresholds = type_to_thresholds[transformer_type]

    if st.button('Calculate Health Index and Check Abnormalities'):
        scores = [
            calculateScore(dga.C2H2, [(0, 6), (2, 5), (5, 4), (10, 3), (15, 2)]),
            calculateScore(dga.C2H4, [(5, 6), (20, 5), (40, 4), (60, 3), (75, 2)]),
            calculateScore(dga.H2, [(40, 6), (60, 5), (70, 4), (75, 3), (80, 2)]),
            calculateScore(dga.CH4, [(10, 6), (25, 5), (40, 4), (55, 3), (60, 2)]),
            calculateScore(dga.C2H6, [(15, 6), (30, 5), (40, 4), (55, 3), (60, 2)]),
            calculateScore(dga.CO, [(200, 6), (350, 5), (540, 4), (650, 3), (700, 2)]),
            calculateScore(dga.CO2, [(3000, 6), (4500, 5), (5100, 4), (6000, 3), (6500, 2)]),
            calculateScore(dga.Acidity, [(0.04, 4), (0.1, 3), (0.15, 2)]),
            calculateScore(dga.Resistance, [(0.09, 1), (0.5, 2), (1, 3), (999999, 4)]),
            calculateScore(dga.BDV, [(35, 1), (47, 2), (51, 3), (999999, 4)]),
            calculateScore(dga.Water, [(20, 4), (25, 3), (30, 2)]),
            calculateScore(dga.tand, [(0.1, 4), (0.5, 3), (1.1, 2)]),
            calculateScore(dga.Furan, [(800, 5), (1500, 4), (3000, 3), (6000, 2)]),
            calculateScore(dga.age, [(10, 5), (20, 4), (35, 3), (50, 2)]),
            calculateScore(dga.ir, [(50, 1), (99, 2), (500, 3), (1000, 4), (999999, 4)]),
            calculateScore(dga.olk, [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2)]),
            calculateScore(dga.wrd, [(3, 5), (5, 4), (8, 3), (10, 2)])
        ]

        final_score = (
            6 * scores[0] + 4 * scores[1] + 3 * scores[2] + scores[3] + scores[4] + scores[5] + scores[6] +
            5 * scores[7] + 4 * scores[8] + 4 * scores[9] + 4 * scores[10] + 2 * scores[11] +
            5 * scores[12] + 5 * scores[13] + 2 * scores[14] + scores[15] + 3 * scores[16]
        )

        health_index = final_score * 100 / 257
        st.write(f"Final Health Index is: {health_index:.2f}")

        abnormalities = getAbnormalities(dga, thresholds)
        if abnormalities:
            st.write(f"Abnormalities detected for transformer category {transformer_type}:")
            for abnormality in abnormalities:
                st.write(f" - {abnormality}")
        else:
            st.write(f"No abnormalities detected for transformer category {transformer_type}")

# Helper functions
def evaluateDGA(dga, arr):
    ratios = [dga["CH4"] / 40, dga["C2H6"]/ 50, dga["C2H4"] / 60, dga["C2H2"] / 1, dga["H2"] / 60]
    max_ratio = max(ratios)

    if max_ratio ==0:
        return "Invalid Input: Total concentration of gases cannot be zero"
    elif max_ratio == dga["H2"] / 60:
        arr[0] = 1.5
        return "Case-1: Partial Discharges in voids"
    elif max_ratio == dga["CH4"]/ 40:
        arr[0] = 3
        return "Case-2: Sparking < 150°C"
    elif max_ratio == dga["C2H6"] / 50:
        arr[0] = 8.5
        return "Case-3: Local Overheating between 150°C and 300°C"
    elif max_ratio == dga["C2H4"] / 60:
        arr[0] = 10
        return "Case-4: Severe Overheating between 300°C and 700°C"
    elif max_ratio == dga["C2H2"] / 1:
        arr[0] = 12
        return "Case-5: Arcing > 700°C"
    return "Unknown condition."

def evaluateIEC(C10, C11, C12, arr):
    D10 = 0 if C10 < 0.1 else 1 if C10 < 3 else 2
    D11 = 1 if C11 < 0.1 else 0 if C11 < 1 else 2
    D12 = 0 if C12 < 0.1 else 0 if C12 < 1 else 1 if C12 < 3 else 2

    if C10 == 9999999 and  C11==9999999 and C12 == 9999999:
        return "Invalid Input: Total concentration of gases cannot be zero"
    elif D10 == 0 and D11 == 0 and D12 == 0:
        arr[2] = 0
        return "Case - 0: No Fault"
    elif D10 == 0 and D11 == 1 and D12 == 0:
        arr[2] = 1
        return "Case 1: Low energy Partial Discharges"
    elif D10 == 1 and D11 == 1 and D12 == 0:
        arr[2] = 2
        return "Case 2: Low Energy Partial Discharges with tracking"
    elif (D10 == 1 or D10 == 2) and D11 == 0 and (D12 == 1 or D12 == 2):
        arr[2] = 3
        return "Case - 3 : Discharges of Low Energy Density (Sparking)"
    elif D10 == 1 and D11 == 0 and D12 == 2:
        arr[2] = 12.5
        return "Case 4: Discharges of High Energy (Arcing)"
    elif D10 == 0 and D11 == 0 and D12 == 1:
        arr[2] = 5.5
        return "Case 5: Thermal Fault < 150°C - General insulated conductor overheating"
    elif D10 == 0 and D11 == 2 and D12 == 0:
        arr[2] = 8.5
        return "Case 6: Thermal Fault 150°C - 300°C"
    elif D10 == 0 and D11 == 2 and D12 == 1:
        arr[2] = 10
        return "Case 7: Thermal Fault 300°C - 700°C"
    else:
        arr[2] = 11
        return "Case 8: Thermal Fault > 700°C"

def evaluateRoger(C10a, C11a, C12a, C13a, arr):
    if C10a == 9999999 and  C11a==9999999 and C12a == 9999999:
        return "Invalid Input: Total concentration of gases cannot be zero"
    elif 0.1 < C10a < 1 and C12a < 1 and C13a < 0.1:
        arr[3] = 0
        return "Case-0: Normal"
    elif C10a < 0.1 and C12a < 1 and C13a < 0.1:
        arr[3] = 1.5
        return "Case-1: Low Energy density PD"
    elif 0.1 <= C10a <= 1 and C12a > 3 and 0.1 <= C13a <= 3:
        arr[3] = 12.5
        return "Case-2: Arcing High Energy discharge"
    elif 0.1 < C10a < 1 and 1 <= C12a <= 3 and C13a < 0.1:
        arr[3] = 7
        return "Case-3: Low temperature thermal"
    elif C10a > 1 and 1 <= C12a <= 3 and C13a < 0.1:
        arr[3] = 10
        return "Case-4: Thermal Fault < 700 C"
    elif C10a > 1 and C12a > 3 and C13a < 0.1:
        arr[3] = 11
        return "Case-5: Thermal Fault > 700 C"
    else:
        arr[3] = 0
        return "N/A"
    
def duvalTriangle(dga, arr):
    total = dga["CH4"] + dga["C2H2"] + dga["C2H4"]

    if total == 0:
        return "Invalid Input: Total concentration of gases cannot be zero."

    CH4_percent = (dga["CH4"] / total) * 100
    C2H2_percent = (dga["C2H2"] / total) * 100
    C2H4_percent = (dga["C2H4"] / total) * 100

    if C2H2_percent > 29:
        arr[1] = 2
        return "D1: Discharges of high energy (arcing)"
    elif C2H4_percent > 48:
        arr[1] = 11
        return "T3: Thermal fault > 700°C"
    elif CH4_percent > 98:
        arr[1] = 1
        return "PD: Partial Discharge"
    elif 13 < C2H2_percent <= 29 and 10 < C2H4_percent <= 48:
        arr[1] = 3
        return "D2: Discharges of low energy (sparking)"
    elif 87 < CH4_percent <= 98 and C2H2_percent <= 13 and C2H4_percent <= 10:
        arr[1] = 1
        return "PD: Partial Discharge"
    elif 4 < C2H2_percent <= 13 and 24 < C2H4_percent <= 48 and 33 < CH4_percent <= 87:
        arr[1] = 10
        return "T2: Thermal fault 300°C - 700°C"
    elif 10 < C2H4_percent <= 24 and 33 < CH4_percent <= 98 and C2H2_percent <= 4:
        arr[1] = 8.5
        return "T1: Thermal fault < 300°C"
    else:
        arr[1] = 0
        return "N/A: Fault type cannot be determined."

def GasAnalysis(score):
    if 0 < score <= 1:
        return "Normal"
    elif 1 <= score <= 2:
        return "Partial Discharge"
    elif 2 <= score <= 3:
        return "Partial Discharge with tracking"
    elif 3 <= score <= 4:
        return "Sparking"
    elif 4 <= score <= 5:
        return "General Conductor Overheating"
    elif 5 <= score <= 6:
        return "Overheating due to winding circulation current"
    elif 6 <= score <= 7:
        return "Overheating due to circulation current in core, tank and joints"
    elif 7 <= score <= 8:
        return "Thermal Overheating < 150 C"
    elif 8 <= score <= 9:
        return "Thermal Overheating 150 C - 300 C"
    elif 9 <= score <= 10:
        return "Thermal Overheating 300 C -700 C"
    elif 10 <= score <= 11:
        return "Thermal Overheating > 700 C"
    elif 11 <= score <= 12:
        return "Heating > 700 C and Pre-arcing"
    elif 12 <= score <= 13:
        return "Heavy Arcing"
    else:
        return "We don't know"

def pikachu(arr, arr1):
    if arr[0] != 0:
        arr1[0] = 1
    if arr[0] == 0:
        arr1[1] = 0
    if arr[1] != 0:
        arr1[1] = 1
    if arr[1] == 0:
        arr1[1] = 0
    if arr[2] != 0:
        arr1[2] = 1
    if arr[2] == 0:
        arr1[2] = 0
    if arr[3] != 0:
        arr1[3] = 1
    if arr[3] == 0:
        arr1[3] = 0
    return sum(arr1)

def c0c02(C0, CO2, arr):
    if C0 != 0:
        ratio = CO2 / C0
        if ratio > 11:
            arr[0] = 0
            return "Thermal Fault involving cellulose paper < 150 degree C. Check 5H2F (5-hydroxymethyl-2-furaldehyde) and 2FAL (2-furaldehyde) in Furan"
        elif ratio <3 :
            arr[0] = 12
            return "Thermal Fault involving cellulose paper > 200 degree C. Check 2FAL (2-furaldehyde)and 5M2F (5-methyl-2-furaldehyde) in Furan "
        elif ratio >=3 and ratio <= 11:
            return "Normal"
    return "Invalid Input: Total concentration of gases cannot be zero."



# Streamlit pages
def home_analysis():
    st.title("Transformer RUL Prediction")
    st.write("Upload your CSV file to get started.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Here is how your dataset looks like:")
        st.write(data.head(10))

        if st.button("Proceed to Analysis"):
            st.session_state["data"] = data

    if "data" in st.session_state:
        st.title("Analysis Report")
        data = st.session_state["data"]

        st.write("Summary Statistics:")
        st.write(data.describe())

        # Displaying various charts for the original dataset
        st.header("Data Visualizations")

        # Pie Chart of Health Index distribution
        st.subheader("Pie Chart of Health Index Distribution")
        # Create bins for the Health Index ranges
        bins = [0, 70, 80, 90, 100]
        labels = ['0-30','30-50','50-85', '85-100']
        health_index_binned = pd.cut(data['Health Indx'], bins=bins, labels=labels, right=False)
        health_index_distribution = health_index_binned.value_counts().sort_index()

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(health_index_distribution, labels=health_index_distribution.index, autopct='%1.1f%%', colors=sns.color_palette("Accent", len(health_index_distribution)))
            st.pyplot(fig)

        # Line Chart for a trend analysis (example: Age vs Health Index)
        st.subheader("Trend Analysis: Age vs Health Index")
        fig, ax = plt.subplots(figsize=(13, 3))
        ax.plot(data['Age(Yrs)'], data['Health Indx'], 'bo')
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Health Index")
        st.pyplot(fig)

        if "Furan" in data.columns and "Age(Yrs)" in data.columns:
            st.subheader("Trend Analysis: Age vs Furan")
            fig, ax = plt.subplots(figsize=(13, 3))
            ax.plot(data["Age(Yrs)"], data["Furan"], 'bo')
            ax.set_xlabel("Age (Years)")
            ax.set_ylabel("Furan")
            st.pyplot(fig)

        column = "Health Indx"
        if st.button("Train Model"):
            X = data.drop(columns=[column])
            y = data[column]

            reg = LazyRegressor()
            models, predictions = reg.fit(X, X, y, y)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Performance")
                st.write(models)

            with col2:
                st.subheader("Predictions")
                st.write(predictions)

            if st.button("Download Predictions"):
                predictions_csv = predictions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=predictions_csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            # Identifying the best model based on the highest R^2 score
            best_model_name = models.sort_values(by="R-Squared", ascending=False).index[0]
            best_model = reg.models[best_model_name]

            st.session_state["best_model"] = best_model
            st.write(f"The best model is: {best_model_name}")

        if "best_model" in st.session_state:
            # Upload file to predict health index and remaining life
            st.write("Upload a new CSV file to predict the health index and remaining life based on the best model")
            new_file = st.file_uploader("Choose a new CSV file", type="csv")

            if new_file is not None:
                new_data = pd.read_csv(new_file)
                st.write("Here is how your new dataset looks like:")
                st.write(new_data.head(10))

                if st.button("Predict Health Index and Remaining Life"):
                    new_X = new_data.drop(columns=[column], errors='ignore')  # Ensure the new data has the same columns except "Health Indx"
                    best_model = st.session_state["best_model"]
                    health_index_predictions = best_model.predict(new_X)
                    new_data["Health Indx Prediction"] = health_index_predictions

                    # Predict Remaining Life based on Furan values
                    aging_table = [
                        {'fal_ppb': 0, 'degree_of_polymerisation': 800, 'percentage_remaining_life': 100, 'interpretation': 'Normal Aging Rate'},
                        {'fal_ppb': 130, 'degree_of_polymerisation': 700, 'percentage_remaining_life': 90, 'interpretation': 'Normal Aging Rate'},
                        {'fal_ppb': 292, 'degree_of_polymerisation': 600, 'percentage_remaining_life': 79, 'interpretation': 'Normal Aging Rate'},
                        {'fal_ppb': 654, 'degree_of_polymerisation': 500, 'percentage_remaining_life': 66, 'interpretation': 'Accelerated Aging Rate'},
                        {'fal_ppb': 1464, 'degree_of_polymerisation': 400, 'percentage_remaining_life': 50, 'interpretation': 'Accelerated Aging Rate'},
                        {'fal_ppb': 1720, 'degree_of_polymerisation': 380, 'percentage_remaining_life': 46, 'interpretation': 'Accelerated Aging Rate'},
                        {'fal_ppb': 2021, 'degree_of_polymerisation': 360, 'percentage_remaining_life': 42, 'interpretation': 'Accelerated Aging Rate'},
                        {'fal_ppb': 2374, 'degree_of_polymerisation': 340, 'percentage_remaining_life': 38, 'interpretation': 'Excessive Aging Danger Zone'},
                        {'fal_ppb': 2789, 'degree_of_polymerisation': 320, 'percentage_remaining_life': 33, 'interpretation': 'Excessive Aging Danger Zone'},
                        {'fal_ppb': 3277, 'degree_of_polymerisation': 300, 'percentage_remaining_life': 29, 'interpretation': 'Excessive Aging Danger Zone'},
                        {'fal_ppb': 3851, 'degree_of_polymerisation': 280, 'percentage_remaining_life': 24, 'interpretation': 'High Risk of Failure'},
                        {'fal_ppb': 4524, 'degree_of_polymerisation': 260, 'percentage_remaining_life': 19, 'interpretation': 'High Risk of Failure'},
                        {'fal_ppb': 5315, 'degree_of_polymerisation': 240, 'percentage_remaining_life': 13, 'interpretation': 'End of expected life of paper insulation and of the transformer'},
                        {'fal_ppb': 6245, 'degree_of_polymerisation': 220, 'percentage_remaining_life': 7, 'interpretation': 'End of expected life of paper insulation and of the transformer'},
                        {'fal_ppb': 7377, 'degree_of_polymerisation': 200, 'percentage_remaining_life': 0, 'interpretation': 'End of expected life of paper insulation and of the transformer'}
                                    ]

                    remaining_life_predictions = []
                    if 'Furan' in new_data.columns:
                        for idx, fal_ppb in enumerate(new_data['Furan']):
                            result = interpolateData(aging_table, fal_ppb)
                            remaining_life_percentage = result['percentage_remaining_life']

                            if remaining_life_percentage is not None:  # Add this check
                                remaining_life_years = (remaining_life_percentage / 100) * new_data.loc[idx, 'Age(Yrs)']
                                remaining_life_predictions.append(remaining_life_years)
                            else:
                                remaining_life_predictions.append(None)  # Handle None case if needed

                        new_data['Predicted Remaining Life (in years)'] = remaining_life_predictions
                    else:
                        st.warning("The uploaded CSV file must contain a 'Furan' column.")
                    
                    st.write("Predictions added to dataset:")
                    st.write(new_data.head(25))

                    new_predictions_csv = new_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download New Predictions as CSV",
                        data=new_predictions_csv,
                        file_name="new_predictions.csv",
                        mime="text/csv"
                    )

                    # Additional Visualizations for the predicted data
                    st.header("Predicted Data Visualizations")

                    col1, col2 = st.columns(2)

                    # Histogram of predicted Health Index values
                    with col1:
                        st.subheader("Distribution of Predicted Health Index")
                        fig, ax = plt.subplots(figsize=(7.75, 6))
                        ax.hist(new_data["Health Indx Prediction"], bins=20, edgecolor='white')
                        ax.set_xlabel("Health Indx Prediction")
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)

                    # Correlation Heatmap
                    with col2:
                        st.subheader("Correlation Heatmap")
                        corr = new_data.corr()
                        fig, ax = plt.subplots(figsize=(18, 12))
                        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                        st.pyplot(fig)

def interpolate(x1, y1, x2, y2, x):
    return y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)


def interpolateData(table, fal_ppb):
    # Interpolation logic goes here
    for i in range(len(table)-1):
        if table[i]['fal_ppb'] <= fal_ppb <= table[i+1]['fal_ppb']:
            x0, y0 = table[i]['fal_ppb'], table[i]['degree_of_polymerisation']
            x1, y1 = table[i+1]['fal_ppb'], table[i+1]['degree_of_polymerisation']
            degree_of_polymerisation = y0 + (fal_ppb - x0) * (y1 - y0) / (x1 - x0)
            
            x0, y0 = table[i]['fal_ppb'], table[i]['percentage_remaining_life']
            x1, y1 = table[i+1]['fal_ppb'], table[i+1]['percentage_remaining_life']
            percentage_remaining_life = y0 + (fal_ppb - x0) * (y1 - y0) / (x1 - x0)
            
            interpretation = table[i]['interpretation'] if table[i]['interpretation'] else table[i+1]['interpretation']
            
            return {
                'degree_of_polymerisation': degree_of_polymerisation,
                'percentage_remaining_life': percentage_remaining_life,
                'interpretation': interpretation
            }
    return {
        'degree_of_polymerisation': None,
        'percentage_remaining_life': None,
        'interpretation': 'Value out of range'
    }

def furan_pred():
    st.title("Transformer Age Prediction using Furan")

    aging_table = [
        {'fal_ppb': 0, 'degree_of_polymerisation': 800, 'percentage_remaining_life': 100, 'interpretation': 'Normal Aging Rate'},
        {'fal_ppb': 130, 'degree_of_polymerisation': 700, 'percentage_remaining_life': 90, 'interpretation': 'Normal Aging Rate'},
        {'fal_ppb': 292, 'degree_of_polymerisation': 600, 'percentage_remaining_life': 79, 'interpretation': 'Normal Aging Rate'},
        {'fal_ppb': 654, 'degree_of_polymerisation': 500, 'percentage_remaining_life': 66, 'interpretation': 'Accelerated Aging Rate'},
        {'fal_ppb': 1464, 'degree_of_polymerisation': 400, 'percentage_remaining_life': 50, 'interpretation': 'Accelerated Aging Rate'},
        {'fal_ppb': 1720, 'degree_of_polymerisation': 380, 'percentage_remaining_life': 46, 'interpretation': 'Accelerated Aging Rate'},
        {'fal_ppb': 2021, 'degree_of_polymerisation': 360, 'percentage_remaining_life': 42, 'interpretation': 'Accelerated Aging Rate'},
        {'fal_ppb': 2374, 'degree_of_polymerisation': 340, 'percentage_remaining_life': 38, 'interpretation': 'Excessive Aging Danger Zone'},
        {'fal_ppb': 2789, 'degree_of_polymerisation': 320, 'percentage_remaining_life': 33, 'interpretation': 'Excessive Aging Danger Zone'},
        {'fal_ppb': 3277, 'degree_of_polymerisation': 300, 'percentage_remaining_life': 29, 'interpretation': 'Excessive Aging Danger Zone'},
        {'fal_ppb': 3851, 'degree_of_polymerisation': 280, 'percentage_remaining_life': 24, 'interpretation': 'High Risk of Failure'},
        {'fal_ppb': 4524, 'degree_of_polymerisation': 260, 'percentage_remaining_life': 19, 'interpretation': 'High Risk of Failure'},
        {'fal_ppb': 5315, 'degree_of_polymerisation': 240, 'percentage_remaining_life': 13, 'interpretation': 'End of expected life of paper insulation and of the transformer'},
        {'fal_ppb': 6245, 'degree_of_polymerisation': 220, 'percentage_remaining_life': 7, 'interpretation': 'End of expected life of paper insulation and of the transformer'},
        {'fal_ppb': 7377, 'degree_of_polymerisation': 200, 'percentage_remaining_life': 0, 'interpretation': 'End of expected life of paper insulation and of the transformer'}
    ]

    # Centering and styling the "Choose input method" radio button
    st.markdown(
        """
        <style>
        .centered-radio {
            display: flex;
            justify-content: center;
        }
        .centered-radio label {
            font-size: 20px;
            margin-right: 10px;
        }
        .centered-radio div {
            display: flex;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="centered-radio">', unsafe_allow_html=True)
    analysis_type = st.radio("Choose input method:", ("Manual Entry", "Upload CSV"))
    st.markdown('</div>', unsafe_allow_html=True)

    if analysis_type == "Manual Entry":
        fal_ppb = st.number_input('Enter the 2FAL (ppb) value:', min_value=0, value=0, step=100)
        if st.button('Predict'):
            result = interpolateData(aging_table, fal_ppb)
            st.write(f"Estimated degree of polymerisation: {result['degree_of_polymerisation']:.2f}")
            st.write(f"Estimated percentage of remaining life: {result['percentage_remaining_life']:.2f}")
            st.write(f"Interpretation: {result['interpretation']}")
            
    elif analysis_type == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            data = data.dropna(axis=1, how='all')  # Drop columns with all NaN values
            if "Furan" in data.columns:
                results = []
                for index, row in data.iterrows():
                    Furan = row['Furan']
                    result = interpolateData(aging_table, Furan)
                    results.append(result)

                results_df = pd.DataFrame(results)
                combined_df = pd.concat([data, results_df], axis=1)
                st.write(combined_df)

                csv = combined_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name='furan_analysis_results.csv',
                    mime='text/csv',
                )
            else:
                st.error("The uploaded file does not contain the required column: fal_ppb.")



def map_column_names(df):
    column_mapping = {
        "H2": ["H2", "Hydrogen"],
        "CH4": ["CH4", "Methane"],
        "C2H6": ["C2H6", "Ethane"],
        "C2H4": ["C2H4", "Ethylene"],
        "C2H2": ["C2H2", "Acetylene"],
        "CO": ["CO", "Carbon Monoxide"],
        "CO2": ["CO2", "Carbon Dioxide"]
    }

    mapped_columns = {}
    for key, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                mapped_columns[key] = df[name]
                break
    return pd.DataFrame(mapped_columns)

def process_dga_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    mapped_df = map_column_names(df)
    if mapped_df.shape[1] != 7:
        st.error("The uploaded file does not contain all the required columns.")
        return None
    return df, mapped_df

def dga_analysis():
    st.title("DGA Analysis")

    # Centering and styling the "Choose input method" radio button
    st.markdown(
        """
        <style>
        .centered-radio {
            display: flex;
            justify-content: center;
        }
        .centered-radio label {
            font-size: 20px;
            margin-right: 10px;
        }
        .centered-radio div {
            display: flex;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="centered-radio">', unsafe_allow_html=True)
    analysis_type = st.radio("Choose input method:", ("Manual Entry", "Upload CSV"))
    st.markdown('</div>', unsafe_allow_html=True)

    if analysis_type == "Manual Entry":
        dga_data = {}
        dga_data["H2"] = st.number_input("Enter H2 (ppm):", min_value=0.0)
        dga_data["CH4"] = st.number_input("Enter CH4 (ppm):", min_value=0.0)
        dga_data["C2H6"] = st.number_input("Enter C2H6 (ppm):", min_value=0.0)
        dga_data["C2H4"] = st.number_input("Enter C2H4 (ppm):", min_value=0.0)
        dga_data["C2H2"] = st.number_input("Enter C2H2 (ppm):", min_value=0.0)
        dga_data["CO"] = st.number_input("Enter CO (ppm):", min_value=0.0)
        dga_data["CO2"] = st.number_input("Enter CO2 (ppm):", min_value=0.0)

        if st.button("Evaluate DGA"):
            arr = [0, 0, 0, 0]
            arr1 = [0, 0, 0, 0]

            result = evaluateDGA(dga_data, arr)
            duval_result = duvalTriangle(dga_data, arr)
            C10 = dga_data["CH4"] / dga_data["H2"] if dga_data["H2"] != 0 else 9999999
            C11 = dga_data["C2H2"] / dga_data["H2"] if dga_data["H2"] != 0 else 9999999
            C12 = dga_data["C2H2"] / dga_data["C2H6"] if dga_data["C2H6"] != 0 else 9999999
            C10a = dga_data["CH4"] / dga_data["H2"] if dga_data["H2"] != 0 else 9999999
            C11a = dga_data["C2H2"] / dga_data["H2"] if dga_data["H2"] != 0 else 9999999
            C12a = dga_data["C2H2"] / dga_data["C2H6"] if dga_data["C2H6"] != 0 else 9999999
            C13a = dga_data["C2H4"] / dga_data["C2H6"] if dga_data["C2H6"] != 0 else 9999999

            iec_result = evaluateIEC(C10, C11, C12, arr)
            roger_result = evaluateRoger(C10a, C11a, C12a, C13a, arr)
            c0c02_result = c0c02(dga_data["CO"], dga_data["CO2"], arr)

            st.write("Key Gas Result: ", result)
            st.write("Duval Triangle Condition: ", duval_result)
            st.write("IEC Condition: ", iec_result)
            st.write("Roger Condition: ", roger_result)
            st.write("C0/CO2 Condition: ", c0c02_result)

    elif analysis_type == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df, mapped_df = process_dga_file(uploaded_file)
            if mapped_df is not None:
                results = []
                for index, row in mapped_df.iterrows():
                    dga_data = row.to_dict()
                    arr = [0, 0, 0, 0]
                    arr1 = [0, 0, 0, 0]

                    result = evaluateDGA(dga_data, arr)
                    duval_result = duvalTriangle(dga_data, arr)
                    C10 = dga_data["CH4"] / dga_data["H2"] if dga_data["H2"] != 0 else 9999999
                    C11 = dga_data["C2H2"] / dga_data["H2"] if dga_data["H2"] != 0 else 9999999
                    C12 = dga_data["C2H2"] / dga_data["C2H6"] if dga_data["C2H6"] != 0 else 9999999
                    C10a = dga_data["CH4"] / dga_data["H2"] if dga_data["H2"] != 0 else 9999999
                    C11a = dga_data["C2H2"] / dga_data["H2"] if dga_data["H2"] != 0 else 9999999
                    C12a = dga_data["C2H2"] / dga_data["C2H6"] if dga_data["C2H6"] != 0 else 9999999
                    C13a = dga_data["C2H4"] / dga_data["C2H6"] if dga_data["C2H6"] != 0 else 9999999

                    iec_result = evaluateIEC(C10, C11, C12, arr)
                    roger_result = evaluateRoger(C10a, C11a, C12a, C13a, arr)
                    c0c02_result = c0c02(dga_data["CO"], dga_data["CO2"], arr)

                    results.append({
                        "Key Gas Result": result,
                        "Duval Triangle Condition": duval_result,
                        "IEC Condition": iec_result,
                        "Roger Condition": roger_result,
                        "C0/CO2 Condition": c0c02_result
                    })
                
                results_df = pd.DataFrame(results)
                combined_df = pd.concat([df, results_df], axis=1)
                st.write(combined_df)

                csv = combined_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name='dga_analysis_results.csv',
                    mime='text/csv',
                )
            else:
                st.error("The uploaded file does not contain the required columns: H2, CH4, C2H6, C2H4, C2H2, CO, CO2.")

# Main Application




def add_custom_css():
    st.markdown("""
        <style>
            body {
                margin: 0;
                padding: 0;
                background-image: url('https://www.yourbackgroundimageurl.com');
                background-size: cover;
                background-repeat: no-repeat;
                font-family: Arial, sans-serif;
                color: #ffffff;
            }
            .login-header {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                margin: 50px 0 20px; /* Adjust margin as needed */
                color: #ffffff; /* White text */
                background-color: #1e90ff; /* Dark Blue background */
                padding: 15px 30px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            .login-footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #1e90ff; /* Dark Blue background */
                color: white; /* White text */
                text-align: center;
                padding: 10px;
                border-top: 1px solid #e0e0e0;
            }
            .login-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: calc(100vh - 100px); /* Adjust height to leave space for header and footer */
                flex-direction: column;
            }
            .login-box {
                width: 400px;
                padding: 30px;
                background: #ffffff; /* White background */
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
                margin-top: 40px; /* Adjust margin as needed */
            }
            .login-input {
                width: 100%;
                padding: 20px;
                margin: 15px 0;
                border: 2px solid #1e90ff; /* Dark Blue border */
                border-radius: 10px;
                box-sizing: border-box;
                font-size: 18px; /* Increased font size */
            }
            .login-button {
                width: 100%;
                background-color: #1e90ff; /* Dark Blue */
                color: white; /* White text */
                padding: 18px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 20px;
                transition: background-color 0.3s;
            }
            .login-button:hover {
                background-color: #007acc; /* Darker Blue on hover */
            }
            .captcha {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .captcha-text {
                font-size: 20px;
                color: #1e90ff; /* Dark Blue text */
            }
            @media (max-width: 600px) {
                .login-box {
                    width: 90%;
                    padding: 25px;
                }
            }
        </style>
    """, unsafe_allow_html=True)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_captcha():
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
    return num1, num2, f"{num1} + {num2} = ?"

def login():
    add_custom_css()

    st.markdown('<div class="login-header">Login to Transformer RUL Prediction</div>', unsafe_allow_html=True)

    username = st.text_input("**Username**", key="username", placeholder="Enter your username")
    password = st.text_input("**Password**", type="password", key="password", placeholder="Enter your password")

    # Initialize CAPTCHA
    if "captcha_num1" not in st.session_state:
        st.session_state.captcha_num1, st.session_state.captcha_num2, st.session_state.captcha_text = generate_captcha()

    captcha_answer = st.text_input("**" + st.session_state.captcha_text + "**", key="captcha", placeholder="Enter answer")

    # Mock stored credentials
    stored_username = "Harsh"
    stored_password_hash = hash_password("summerintern24")

    if st.button("**Login**", key="login", help="Click to login"):
        with st.spinner("Processing..."):
            if not captcha_answer:
                st.error("Please enter the CAPTCHA answer.")
            else:
                try:
                    if int(captcha_answer) == (st.session_state.captcha_num1 + st.session_state.captcha_num2):
                        if username == stored_username and hash_password(password) == stored_password_hash:
                            st.session_state["logged_in"] = True
                            st.success("Login successful")
                        else:
                            st.error("Invalid username or password. Please try again.")
                        # Refresh CAPTCHA
                        st.session_state.captcha_num1, st.session_state.captcha_num2, st.session_state.captcha_text = generate_captcha()
                    else:
                        st.error("Incorrect CAPTCHA. Please try again.")
                except ValueError:
                    st.error("Invalid CAPTCHA format. Please enter a valid number.")

    st.markdown('<div class="login-footer">Created by Harsh Sukhwal</div>', unsafe_allow_html=True)


def abnormality_detection():
    st.title("Abnormality Detection in Data")

    entry_mode = st.radio("Select Data Entry Mode", ("Manual Entry", "CSV File Upload"))

    def plot_with_zones(dates, values):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, values, marker='o')
        ax.set_xlabel("Date")
        ax.set_ylabel("Furan Value (ppb)")

        # Zone definitions
        zones = [
            (0, 654, 'Normal Aging Rate', 'green'),
            (654, 1464, 'Accelerated Aging Rate', 'yellow'),
            (1464, 2374, 'Excessive Aging Danger Zone', 'orange'),
            (2374, 4524, 'High Risk of Failure', 'crimson'),
            (4524, 8000, 'End of expected life of paper insulation and of the transformer', 'red')
        ]

        for lower, upper, label, color in zones:
            ax.axhspan(lower, upper, alpha=0.3, color=color, label=label)
        
        # Check for abnormality in slope
        for i in range(1, len(values)):
            delta_value = values[i] - values[i-1]
            delta_date = (dates[i] - dates[i-1]).days
            slope = delta_value / delta_date if delta_date != 0 else np.nan

            if (slope) > 5:
                st.error(f"Abnormality detected: Slope {slope:.2f} between {dates[i-1]} and {dates[i]}")
            elif (slope) < 0:
                st.success(f"Possible overhauling between {dates[i-1]} and {dates[i]}")
                


        # Predict when it will reach 8000 value
        if len(values) >= 2:
            last_slope = (values[-1] - values[-2]) / ((dates[-1] - dates[-2]).days)
            days_to_8000 = (8000 - values[-1]) / last_slope if last_slope != 0 else np.nan
            if days_to_8000 > 0:
                predicted_date_8000 = dates[-1] + pd.Timedelta(days=days_to_8000)
                st.divider()
                st.warning(f"Predicted date to reach 8000 Furan value: {predicted_date_8000.date()}")
                st.divider()
            else:
                st.divider()
                st.warning("The trend does not indicate reaching 8000 Furan value in the future.")
                st.divider()

        ax.legend()
        st.pyplot(fig)

    if entry_mode == "CSV File Upload":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Here is your dataset:")
            st.write(data)

            row_num = st.number_input("Enter the Row Number to Analyze", min_value=0, max_value=len(data)-1, step=1)
            selected_row = data.iloc[row_num]


            dates = [col for col in data.columns if "date" in col.lower()]
            values = [col for col in data.columns if "value" in col.lower()]

            date_values = []
            for date_col, value_col in zip(dates, values):
                date_values.append((pd.to_datetime(selected_row[date_col], dayfirst=True), selected_row[value_col]))

            date_values.sort()  # Sort by date
            sorted_dates, sorted_values = zip(*date_values)

            plot_with_zones(sorted_dates, sorted_values)

    elif entry_mode == "Manual Entry":
        dates = []
        values = []
        num_pairs = st.number_input("Enter the number of date-value pairs", min_value=1, value=1, step=1)

        for i in range(num_pairs):
            date = st.date_input(f"Enter Date {i+1}")
            value = st.number_input(f"Enter Value {i+1}", value=0.0)
            dates.append(date)
            values.append(value)

        if st.button("Plot Data"):
            plot_with_zones(dates, values)


def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    if not st.session_state["logged_in"]:
        login()
    else:    
        pages = {
            "ML-Powered Analysis & Prediction": home_analysis,
            "Multiple DGA Test Analysis": dga_analysis,
            "Transformer Health & Abnormality Detection": Health_index,
            "Age Prediction using Furan": furan_pred,
            "Furan-Based Abnormality Detection (Over Time)": abnormality_detection,
        }
        
        st.sidebar.title("Transformer Remaining Useful Life Prediction")
        
        with st.sidebar:

            st.image("https://www.sellintegro.cloud/images/cloud_asset_1-2.svg", use_column_width=True)

            st.markdown("## Created by Harsh Sukhwal")
            st.markdown("---")
            st.markdown("## Choose a page")
            page = st.radio("", list(pages.keys()))
            st.markdown("---")
            st.markdown("## About")
            st.info(
                """
                This application provides tools for Transformer Health Index prediction and analysis, including:
                - Data Visualization
                - Model Training and Predictions
                - Multiple Dissolved Gas Analysis (DGA) Test Results
                - Age Prediction using Furan Value
                - Health Index and Abnormality Detection
                - And all of that in downloadable csv format
                """
            )

            st.markdown("---")
            st.markdown("## Contact")
            
            st.markdown(
                """
                <style>
                .contact-icons {
                    display: flex;
                    align-items: center;
                }
                .contact-icons img {
                    width: 25px;
                    height: 25px;
                    margin-right: 10px;
                }
                </style>
                <div class="contact-icons">
                    <img src="https://img.icons8.com/ios-filled/50/000000/github.png"/>
                    <a href="https://github.com/MoonBoy17" target="_blank">GitHub</a>
                </div>
                <div class="contact-icons">
                    <img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png"/>
                    <a href="https://www.linkedin.com/in/Harsh-Sukhwal/" target="_blank">LinkedIn</a>
                </div>
                <div class="contact-icons">
                    <img src="https://img.icons8.com/ios-filled/50/000000/gmail.png"/>
                    <a href="mailto:harshsukhwal17@gmail.com" target="_blank">Email</a>
                </div>
                <div class="contact-icons">
                    <img src="https://img.icons8.com/ios-filled/50/000000/internet.png"/>
                    <a href="blank" target="_blank">Website</a>
                </div>
                """,
                unsafe_allow_html=True
            )

        if "page" not in st.session_state:
            st.session_state["page"] = page

        if st.session_state["page"] != page:
            st.session_state["page"] = page

        pages[page]()

if __name__ == "__main__":
    main()
