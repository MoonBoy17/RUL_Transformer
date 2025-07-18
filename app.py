import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
import matplotlib.pyplot as plt
import numpy as np
import random
import hashlib
import streamlit.components.v1 as components
import seaborn as sns
import plotly.express as px


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
        st.subheader("Health Index Distribution")
        # Create bins for the Health Index ranges
        bins = [0, 70, 80, 90, 101] # Changed 100 to 101 to include values of 100
        labels = ['Poor (0-70)', 'Fair (70-80)', 'Good (80-90)', 'Excellent (90-100)']
        health_index_binned = pd.cut(data['Health Indx'], bins=bins, labels=labels, right=False)
        health_index_distribution = health_index_binned.value_counts().sort_index()

        fig_pie = px.pie(
            health_index_distribution,
            values=health_index_distribution.values,
            names=health_index_distribution.index,
            color_discrete_sequence=px.colors.sequential.Agsunset
        )
        st.plotly_chart(fig_pie, use_container_width=True)
  

        # Line Chart for a trend analysis (example: Age vs Health Index)
        st.subheader("Trend Analysis: Age vs Health Index")
        fig_scatter_health = px.scatter(
            data,
            x='Age(Yrs)',
            y='Health Indx',
            hover_data=['Furan'] # Example of adding more context on hover
        )
        st.plotly_chart(fig_scatter_health, use_container_width=True)
  
        from sklearn.model_selection import train_test_split
        from lazypredict.Supervised import LazyRegressor
        if "Furan" in data.columns and "Age(Yrs)" in data.columns:
            # --- REPLACED MATPLOTLIB SCATTER PLOT WITH PLOTLY ---
            st.subheader("Trend Analysis: Age vs Furan")
            fig_scatter_furan = px.scatter(
                data,
                x="Age(Yrs)",
                y="Furan"
            )
            st.plotly_chart(fig_scatter_furan, use_container_width=True)
  
        # Model Training and Evaluation
        column = "Health Indx"
        if st.button("Train Model"):
            X = data.drop(columns=[column])
            y = data[column]

            # Split the data into training and testing sets (80-20 split)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize LazyRegressor
            reg = LazyRegressor()
            
            # Train and evaluate the models
            models_train, _ = reg.fit(X_train, X_train, y_train, y_train)  # Training performance
            _, predictions_test = reg.fit(X_train, X_test, y_train, y_test)  # Testing performance

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Performance on Training Data")
                st.write(models_train)

            with col2:
                st.subheader("Predictions on Testing Data")
                st.write(predictions_test)

            if st.button("Download Predictions"):
                predictions_csv = predictions_test.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=predictions_csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            # Identifying the best model based on the highest R^2 score
            best_model_name = predictions_test.sort_values(by="R-Squared", ascending=False).index[0]
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
                        fig_hist = px.histogram(new_data, x="Health Indx Prediction")
                        st.plotly_chart(fig_hist, use_container_width=True)
  

                    # Correlation Heatmap
                    with col2:
                        st.subheader("Correlation Heatmap of Predicted Data")
                        st.subheader("Correlation Heatmap of Predicted Data") 
                        numeric_df = new_data.select_dtypes(include=np.number)
                        corr = numeric_df.corr()
                        if not corr.empty and corr.notna().any().any():
                            st.subheader("Correlation Heatmap of Predicted Data")
                            fig_heatmap = px.imshow(
                                corr,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale='coolwarm'
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        else:
                            # If the matrix is invalid, inform the user instead of crashing.
                            st.warning("Correlation Heatmap could not be generated. The uploaded data for prediction must contain at least two numeric columns with valid data.")


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
                background-image: components.html(particles_js, height=500, scrolling=False));
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

    def calculate_intervals(rul):
        if rul > 15:
            return {
                "- First Test Interval":      "After 10 years",
                "- Second Test Interval":     "After 2 years if RUL > 5 years ||| Annually if RUL ≤ 5 years",
                "- Third Test Interval":      "Based on updated RUL: Quarterly if RUL ≤ 2 years ||| Every 6 months if 2 < RUL ≤ 5 years ||| Annually if RUL > 5 years"
            }
        elif 10 < rul <= 15:
            return {
                "- First Test Interval":      "After 5 years",
                "- Second Test Interval":     "After 2 years if RUL > 5 years ||| Annually if RUL ≤ 5 years",
                "- Third Test Interval":      "Based on updated RUL: Quarterly if RUL ≤ 2 years ||| Every 6 months if 2 < RUL ≤ 5 years ||| Annually if RUL > 5 years"
            }
        elif 5 < rul <= 10:
            return {
                "- First Test Interval":      "After 3 years",
                "- Second Test Interval":     "Annually",
                "- Third Test Interval":      "Based on updated RUL: Quarterly if RUL ≤ 2 years ||| Every 6 months if 2 < RUL ≤ 5 years"
            }
        elif 3 < rul <= 5:
            return {
                "- First Test Interval":      "After 2 years",
                "- Second Test Interval":     "Annually",
                "- Third Test Interval":      "Based on updated RUL:Quarterly if RUL ≤ 2 years ||| Every 6 months if 2 < RUL ≤ 3 years"
            }
        elif 1 < rul <= 3:
            return {
                "- First Test Interval":      "After 6 months",
                "- Second Test Interval":     "Quarterly",
                "- Third Test Interval":      "Monthly"
            }
        else:
            return {"- Recommendation": "Take down Transformer from grid to prevent heavy damages"}

    def plot_with_zones(dates, values, vendors):
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

        # Check for abnormality in slope and vendor change
        for i in range(1, len(values)):
            delta_value = values[i] - values[i-1]
            delta_date = (dates[i] - dates[i-1]).days
            slope = delta_value / delta_date if delta_date != 0 else np.nan

            if slope > 5:
                st.error(f"Abnormality detected: Slope {slope:.2f} between {dates[i-1]} and {dates[i]}", icon="⚠️")
            elif slope < 0:
                change_percent = abs(delta_value / values[i-1]) * 100
                if change_percent > 20 and vendors[i] != vendors[i-1]:
                    st.warning(f"Possible overhauling but vendor changed recommended to take one more reading: Change {change_percent:.2f}% between {dates[i-1]} and {dates[i]} with vendor change from {vendors[i-1]} to {vendors[i]}", icon="⚠️")
                else:
                    st.success(f"Possible overhauling between {dates[i-1]} and {dates[i]}", icon="✅")

        # Predict when it will reach 8000 value
        predicted_date_8000 = None
        if len(values) >= 2:
            last_slope = (values[-1] - values[-2]) / ((dates[-1] - dates[-2]).days)
            days_to_8000 = (8000 - values[-1]) / last_slope 
            if days_to_8000 > 0:
                predicted_date_8000 = dates[-1] + pd.Timedelta(days=days_to_8000)
                st.divider()
                st.info(f"Predicted date to reach 8000 Furan value: {predicted_date_8000}", icon="📉")
                st.divider()
            else:
                st.divider()
                st.info("The trend does not indicate reaching 8000 Furan value in the future.", icon="📉")
                st.divider()

        ax.legend()
        st.pyplot(fig)

        return predicted_date_8000

    if entry_mode == "CSV File Upload":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Here is your dataset:")
            st.write(data)

            search_mode = st.radio("Search by", ("Row Number", "SAP ID"))

            selected_row = None

            if search_mode == "Row Number":
                row_num = st.number_input("Enter the Row Number to Analyze", min_value=0, max_value=len(data)-1, step=1)
                selected_row = data.iloc[row_num]
            elif search_mode == "SAP ID":
                sap_id = st.text_input("Enter the SAP ID to Search", value="0")
                if sap_id:
                    selected_row = data[data["SAP ID"].astype(str) == str(sap_id)]
                    if selected_row.empty:
                        st.error("SAP ID not found in the dataset.")
                        return
                    selected_row = selected_row.iloc[0]

            if selected_row is not None:
                dates = [col for col in data.columns if "date" in col.lower()]
                values = [col for col in data.columns if "value" in col.lower()]
                vendors = [col for col in data.columns if "vendor" in col.lower()]

                date_values_vendors = []
                for date_col, value_col, vendor_col in zip(dates, values, vendors):
                    date_values_vendors.append((pd.to_datetime(selected_row[date_col], dayfirst=True), selected_row[value_col], selected_row[vendor_col]))

                date_values_vendors.sort()  # Sort by date
                sorted_dates, sorted_values, sorted_vendors = zip(*date_values_vendors)

            predicted_date_8000 = plot_with_zones(sorted_dates, sorted_values, sorted_vendors)
            if predicted_date_8000:
                rul_years = (predicted_date_8000 - sorted_dates[-1]).days / 365
                intervals = calculate_intervals(rul_years)
                st.info(f"Recommended Test Intervals based on Predicted RUL:")
                st.write(f"- Predicted RUL: {round(rul_years,2)} years")
                for test, interval in intervals.items():
                    st.write(f"{test}: {interval}")

    elif entry_mode == "Manual Entry":
        dates = []
        values = []
        vendors = []
        num_pairs = st.number_input("Enter the number of date-value pairs", min_value=1, value=1, step=1)

        for i in range(num_pairs):
            date = st.date_input(f"Enter Date {i+1}")
            value = st.number_input(f"Enter Value {i+1}", value=0.0)
            vendor = st.text_input(f"Enter Vendor {i+1}")
            dates.append(date)
            values.append(value)
            vendors.append(vendor)

        if st.button("Plot Data"):
            predicted_date_8000 = plot_with_zones(dates, values, vendors)
            if predicted_date_8000:
                rul_years = (predicted_date_8000 - dates[-1]).days / 365
                intervals = calculate_intervals(rul_years)
                st.write("Recommended Test Intervals based on Predicted RUL:", icon="💡")
                st.write(f"- Predicted RUL: {round(rul_years,2)} years")
                for test, interval in intervals.items():
                    st.write(f"{test}: {interval}")

if "show_animation" not in st.session_state:
    st.session_state.show_animation = True  # or True, depending on your default preference

def add_particles_background():
    particles_js = """<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Particles.js</title>
      <style>
      #particles-js {
        position: fixed;
        width: 100vw;
        height: 100vh;
        top: 0;
        left: 0;
        z-index: -1; /* Send the animation to the back */
      }
      .content {
        position: relative;
        z-index: 1;
        color: white;
      }
      
    </style>
    </head>
    <body>
      <div id="particles-js"></div>
      <div class="content">
        <!-- Placeholder for Streamlit content -->
      </div>
      <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
      <script>
        particlesJS("particles-js", {
          "particles": {
            "number": {
              "value": 300,
              "density": {
                "enable": true,
                "value_area": 800
              }
            },
            "color": {
              "value": "#ffffff"
            },
            "shape": {
              "type": "circle",
              "stroke": {
                "width": 0,
                "color": "#000000"
              },
              "polygon": {
                "nb_sides": 5
              },
              "image": {
                "src": "img/github.svg",
                "width": 100,
                "height": 100
              }
            },
            "opacity": {
              "value": 0.5,
              "random": false,
              "anim": {
                "enable": false,
                "speed": 1,
                "opacity_min": 0.2,
                "sync": false
              }
            },
            "size": {
              "value": 2,
              "random": true,
              "anim": {
                "enable": false,
                "speed": 40,
                "size_min": 0.1,
                "sync": false
              }
            },
            "line_linked": {
              "enable": true,
              "distance": 100,
              "color": "#ffffff",
              "opacity": 0.22,
              "width": 1
            },
            "move": {
              "enable": true,
              "speed": 0.2,
              "direction": "none",
              "random": false,
              "straight": false,
              "out_mode": "out",
              "bounce": true,
              "attract": {
                "enable": false,
                "rotateX": 600,
                "rotateY": 1200
              }
            }
          },
          "interactivity": {
            "detect_on": "canvas",
            "events": {
              "onhover": {
                "enable": true,
                "mode": "grab"
              },
              "onclick": {
                "enable": true,
                "mode": "repulse"
              },
              "resize": true
            },
            "modes": {
              "grab": {
                "distance": 100,
                "line_linked": {
                  "opacity": 1
                }
              },
              "bubble": {
                "distance": 400,
                "size": 2,
                "duration": 2,
                "opacity": 0.5,
                "speed": 1
              },
              "repulse": {
                "distance": 200,
                "duration": 0.4
              },
              "push": {
                "particles_nb": 2
              },
              "remove": {
                "particles_nb": 3
              }
            }
          },
          "retina_detect": true
        });
      </script>
    </body>
    </html>
    """
    components.html(particles_js, height=350, scrolling=False)



def display_duval_triangle_page():

    html_code = """
    <html>
        <head>
            <style>
                body {
                    background-color: white;
                    padding: 10px;
                    margin: 0px;
                    font-family: Arial, sans-serif;
                }

                #canvas {
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                }

                 div.inputs {
            text-align: left;
            padding: 20px;
            border: 4px solid black;
            display: inline-block;
            background-color: #f9f9f9;
            margin-left: 20px;
            vertical-align: top;
            border-radius: 10px;
        }

        .inputs label {
            display: inline-block;
            width: 80px;
            font-weight: bold;
        }

        .inputs input {
            padding: 5px;
            margin-bottom: 10px;
            width: 150px;
            border: 2px solid #ccc;
            border-radius: 5px;
        }

        #bt1 {
            background-color: rgb(67, 10, 143);
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin-top: 10px;
            cursor: pointer;
            border-radius: 5px;
        }

        #head1 {
            color: white;
            text-align: center;
            background-color: #101D6B;
            font-size: 26px;
            padding: 15px;
            border-radius: 85px;
        }

        #result {
            text-align: center;
            font-size: 18px;
            margin-top: 20px;
            color: black;
            font-weight: bold;
        }

        #result-box {
    border: 2px solid #007BFF; /* Blue border */
    padding: 15px;
    margin-top: 20px;
    background-color: #f1f9ff; /* Light blue background */
    border-radius: 10px;
    font-family: Arial, sans-serif;
    color: #333;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

#result-box h3 {
    margin: 0 0 10px 0;
    font-size: 24px;
    color: #007BFF; /* Blue text for the title */
}

#result-box p {
    margin: 5px 0;
    font-size: 16px;
}
            </style>
        </head>
        <body>
    <h3 id="head1">Enter the values for the gas analysis</h3>
    <div class="inputs">
        <div style="margin:0 0 10px 0;">
            <label>CH4 = </label><input  type="text" id="ch4" name="value1" value=""/> <label>ppm </label><br/>
        </div>
        <div style="margin:0 0 10px 0;">
            <label>C2H2 = </label><input type="text" id="c2h2" name="value2" value=""/> <label>ppm </label><br/>
        </div>
        <div>
            <label>C2H4 = </label><input type="text" id="c2h4" name="value3" value=""/> <label>ppm </label><br/>
        </div>
        <div>
            <button id="bt1" onClick="calcOpr()">Locate</button>
        </div>
    </div>
    <canvas id="canvas" width="650" height="580" class="canvas"></canvas>
    <div id="result"></div>
            <script>
                var canvas = document.getElementById("canvas");
                var ctx = canvas.getContext("2d");

                var pointSize = 4.5;
                var v0 = { x: 114, y: 366 };
                var v1 = { x: 306, y: 30 };
                var v2 = { x: 498, y: 366 };
                var triangle = [v0, v1, v2];

                ctx.font = '14px arial black';
                ctx.fillText("Duval's Triangle DGA", 220, 10, 300);

                var colors = {
                    PD: 'black',
                    T1: 'navajoWhite',
                    T2: 'tan',
                    T3: 'peru',
                    D1: 'rgb(172,236,222)',
                    D2: 'deepskyblue',
                    DT: 'lightCyan'
                };

                var legend = [
                    { label: 'PD = Partial Discharge', color: colors.PD, x: 0, y: 454 },
                    { label: 'T1 = Thermal fault < 300 celcius', color: colors.T1, x: 0, y: 469 },
                    { label: 'T2 = Thermal fault 300 < T < 700 celcius', color: colors.T2, x: 0, y: 484 },
                    { label: 'T3 = Thermal fault < 300 celcius', color: colors.T3, x: 0, y: 499 },
                    { label: 'D1 = Thermal fault T > 700 celcius', color: colors.D1, x: 0, y: 514 },
                    { label: 'D2 = Discharge of High Energy', color: colors.D2, x: 0, y: 529 },
                    { label: 'DT = Electrical and Thermal', color: colors.DT, x: 0, y: 544 }
                ];

                legend.forEach(item => {
                    ctx.fillStyle = item.color;
                    ctx.fillRect(item.x, item.y, 20, 10);
                    ctx.fillStyle = 'black';
                    ctx.fillText(item.label, item.x + 30, item.y + 10);
                });



                var segments = [
                    { points: [v0, { x: 281, y: 76 }, { x: 324, y: 150 }, { x: 201, y: 366 }], fill: colors.D1, label: 'D1', labelPos: { x: 165, y: 395 } },
                    { points: [{ x: 385, y: 366 }, { x: 201, y: 366 }, { x: 324, y: 150 }, { x: 356, y: 204 }, { x: 321, y: 256 }], fill: colors.D2, label: 'D2', labelPos: { x: 300, y: 395 } },
                    { points: [{ x: 297, y: 46 }, { x: 392, y: 214 }, { x: 372, y: 248 }, { x: 441, y: 366 }, { x: 385, y: 366 }, { x: 321, y: 256 }, { x: 356, y: 204 }, { x: 281, y: 76 }], fill: colors.DT, label: 'DT', labelPos: { x: 400, y: 60 }, labelLine: { x: 300, y: 50 }  },
                    { points: [{ x: 306, y: 30 }, { x: 312, y: 40 }, { x: 300, y: 40 }], fill: colors.PD, label: 'PD', labelPos: { x: 356, y: 40 }, labelLine: { x: 321, y: 40 } },
                    { points: [{ x: 312, y: 40 }, { x: 348, y: 103 }, { x: 337, y: 115 }, { x: 297, y: 46 }, { x: 300, y: 40 }], fill: colors.T1, label: 'T1', labelPos: { x: 375, y: 90 }, labelLine: { x: 340, y: 75 } },
                    { points: [{ x: 348, y: 103 }, { x: 402, y: 199 }, { x: 392, y: 214 }, { x: 337, y: 115 }], fill: colors.T2, label: 'T2', labelPos: { x: 400, y: 135 }, labelLine: { x: 366, y: 120 } },
                    { points: [{ x: 402, y: 199 }, { x: 498, y: 366 }, { x: 441, y: 366 }, { x: 372, y: 248 }], fill: colors.T3, label: 'T3', labelPos: { x: 480, y: 285 }, labelLine: { x: 450, y: 270 } },
                    { points: [v0, { x: 281, y: 76 }, { x: 324, y: 150 }, { x: 201, y: 366 }], fill: colors.D1, label: 'Methane', labelPos: { x: 105, y: 200 } },
                    { points: [v0, { x: 281, y: 76 }, { x: 324, y: 150 }, { x: 201, y: 366 }], fill: colors.D1, label: 'Ethylene', labelPos: { x: 455, y: 200 } },
                    { points: [v0, { x: 281, y: 76 }, { x: 324, y: 150 }, { x: 201, y: 366 }], fill: colors.D1, label: 'Acetylene', labelPos: { x: 280, y: 420 } },
                
                ];


                
                segments.forEach(segment => {
                    ctx.beginPath();
                    ctx.moveTo(segment.points[0].x, segment.points[0].y);
                    for (var i = 1; i < segment.points.length; i++) {
                        var point = segment.points[i];
                        ctx.lineTo(point.x, point.y);
                    }
                    ctx.closePath();
                    ctx.fillStyle = segment.fill;
                    ctx.fill();
                    ctx.stroke();

                    ctx.fillStyle = 'black';
                    ctx.fillText(segment.label, segment.labelPos.x, segment.labelPos.y);
                    if (segment.labelLine) {
                        ctx.beginPath();
                        ctx.moveTo(segment.labelPos.x, segment.labelPos.y-4);
                        ctx.lineTo(segment.labelLine.x, segment.labelLine.y+10);
                        ctx.stroke();
                    }
                });

                ctx.beginPath();
                ctx.strokeStyle = 'Balck';
                ctx.moveTo(v0.x, v0.y);
                ctx.lineTo(v1.x, v1.y);
                ctx.lineTo(v2.x, v2.y);
                ctx.lineTo(v0.x, v0.y);
                ctx.stroke();
                ctx.fillStyle = 'colors.D1';
                ctx.font = '12pt bold';
                ctx.fillText("100% C2H2", 440, 390);
                ctx.fillText("100% CH4", 230, 30);
                ctx.fillText("100% C2H4", 40, 390);

                function calcOpr() {
    var ch4 = parseFloat(document.getElementById("ch4").value) || 0;
    var c2h2 = parseFloat(document.getElementById("c2h2").value) || 0;
    var c2h4 = parseFloat(document.getElementById("c2h4").value) || 0;
    var total = ch4 + c2h2 + c2h4;

    var ch4Pct = (ch4 / total) * 100;
    var c2h2Pct = (c2h2 / total) * 100;
    var c2h4Pct = (c2h4 / total) * 100;

    var b = (c2h2Pct / 100) * 306 + ((100 - c2h2Pct) / 100) * 498;
    var a = (c2h2Pct / 100) * 30 + ((100 - c2h2Pct) / 100) * 366;
    var px = (ch4Pct / 100) * 114 + ((100 - ch4Pct) / 100) * b;
    var py = (ch4Pct / 100) * 366 + ((100 - ch4Pct) / 100) * a;

    ctx.fillStyle = 'red';
    ctx.beginPath();
    ctx.arc(px, py, pointSize, 0, Math.PI * 2);
    ctx.fill();

    var diagnosis = "";
    var resultColors = Object.values(colors);
    var pixelData = ctx.getImageData(px, py, 1, 1).data;
    var hexColor = "#" + ((1 << 24) + (pixelData[0] << 16) + (pixelData[1] << 8) + pixelData[2]).toString(16).slice(1).toUpperCase();

    for (var key in colors) {
        if (colors[key] === hexColor) {
            diagnosis = key;
            break;
        }
    }

    var zone = getDiagnosisResult(ch4Pct, c2h2Pct, c2h4Pct);

     document.getElementById("result").innerHTML = `
        <h3>Diagnosis Result</h3>
        <p><strong>${zone}</strong></p>
        <p>CH4: ${ch4Pct.toFixed(2)}%, C2H2: ${c2h2Pct.toFixed(2)}%, C2H4: ${c2h4Pct.toFixed(2)}%</p>`;
}

                function drawDot(ch4x, ch4y, c2h2x, c2h2y, c2h4x, c2h4y) {
                    var x = (ch4x + c2h2x + c2h4x) / 3;
                    var y = (ch4y + c2h2y + c2h4y) / 3;
                    ctx.fillStyle = "#ff2626";
                    ctx.beginPath();
                    ctx.arc(x, y, pointSize, 0, Math.PI * 2, true);
                    ctx.fill();
                }

                ticklines(v0,v1,9,0,20);
                ticklines(v1,v2,9,Math.PI*3/4,20);
                ticklines(v2,v0,9,Math.PI*5/4,20);
                

                function moleculeLabel(start,end,offsetLength,angle,text){
                ctx.textAlign='center';
                ctx.textBaseline='middle'
                ctx.font='14px verdana';
                var dx=end.x-start.x;
                var dy=end.y-start.y;
                var x0=parseInt(start.x+dx*0.50);
                var y0=parseInt(start.y+dy*0.50);
                var x1=parseInt(x0+offsetLength*Math.cos(angle));
                var y1=parseInt(y0+offsetLength*Math.sin(angle));
                ctx.fillStyle='black';
                ctx.fillText(text,x1,y1);
                // arrow
                var x0=parseInt(start.x+dx*0.35);
                var y0=parseInt(start.y+dy*0.35);
                var x1=parseInt(x0+50*Math.cos(angle));
                var y1=parseInt(y0+50*Math.sin(angle));
                var x2=parseInt(start.x+dx*0.65);
                var y2=parseInt(start.y+dy*0.65);
                var x3=parseInt(x2+50*Math.cos(angle));
                var y3=parseInt(y2+50*Math.sin(angle));
                ctx.beginPath();
                ctx.moveTo(x1,y1);
                ctx.lineTo(x3,y3);
                ctx.strokeStyle='black';
                ctx.lineWidth=1;
                ctx.stroke();
                var angle=Math.atan2(dy,dx);
                ctx.translate(x3,y3);
                ctx.rotate(angle);
                ctx.drawImage(arrowhead,-arrowheadLength,-arrowheadWidth/2);
                ctx.setTransform(1,0,0,1,0,0);
                }

                function ticklines(start,end,count,angle,length){
                var dx=end.x-start.x;
                var dy=end.y-start.y;
                ctx.lineWidth=2;
                for(var i=1;i<count;i++){
                    var x0=parseInt(start.x+dx*i/count);
                    var y0=parseInt(start.y+dy*i/count);
                    var x1=parseInt(x0+length*Math.cos(angle));
                    var y1=parseInt(y0+length*Math.sin(angle));
                    ctx.beginPath();
                    ctx.moveTo(x0,y0);
                    ctx.lineTo(x1,y1);
                    ctx.stroke();
                    if(i==2 || i==4 || i==6 || i==8){
                    var labelOffset=length*3/4;
                    var x1=parseInt(x0-labelOffset*Math.cos(angle));
                    var y1=parseInt(y0-labelOffset*Math.sin(angle));
                    ctx.fillStyle='black';
                    ctx.font='14px verdana';
                    ctx.fillText(parseInt(i*10),x1,y1);
                    }
                }
                }

                function getDiagnosisResult(ch4Percentage, c2h2Percentage, c2h4Percentage) {
                var result = "";
                if (ch4Percentage < 5 && c2h2Percentage < 25 && c2h4Percentage > 20) {
                    result = "PD = Partial Discharge";
                } else if (ch4Percentage > 6 && c2h2Percentage < 12 && c2h4Percentage < 14) {
                    result = "T1 = Thermal fault < 300 celcius";
                } else if (ch4Percentage > 12 && c2h2Percentage < 23 && c2h4Percentage < 40) {
                    result = "T2 = Thermal fault 300 < T < 700 celcius";
                } else if (ch4Percentage > 30 && c2h2Percentage < 22 && c2h4Percentage < 50) {
                    result = "T3 = Thermal fault < 300 celcius";
                } else if (ch4Percentage < 30 && c2h2Percentage < 40 && c2h4Percentage > 35) {
                    result = "D1 = Thermal fault T > 700 celcius";
                } else if (ch4Percentage < 50 && c2h2Percentage > 23 && c2h4Percentage < 55) {
                    result = "D2 = Discharge of High Energy";
                } else if (ch4Percentage < 80 && c2h2Percentage > 60 && c2h4Percentage < 70) {
                    result = "DT = Electrical and Thermal";
                } else {
                    result = "Undefined";
                }
                return result;
            }

            </script>
        </body>
    </html>
    """

    components.html(html_code, height=1100)


html_content = """

<html>
    <head>
        <link rel="stylesheet" type="text/css" href="mysheet.css"/>
    </head>
    <body>
             <canvas id="canvas" width=650 height=580 class="canvas"></canvas>
             <div class="inputs">
                <h3 id="head1">  Give the inputs for the gases</h3>
                <div style="margin:0 0 0px 0;">
                    <label>CH4 = </label><input type="text" id="ch4" name="value1" value=""/> <label>ppm </label><br/>
                    </div>
                <div style="margin:0 0 0px 0;">
                   <label>C2H2 = </label><input type="text" id="c2h2" name="value2" value=""/> <label>ppm </label><br/>
                    </div>
                <div>
                    <label>C2H4 = </label><input type="text" id="c2h4" name="value3" value=""/> <label>ppm </label><br/>
                    </div>
                <div>
                    <button id="bt1" onClick="calcOpr()"> Analyse </button>
                    </div>
            </div>
            <script src=duval.js></script>
            
    </body>
</html>
"""

# Your CSS code
css_content = """

    body {
        background-color: rgb(249, 251, 252);
        padding: 10px;
        margin: 0px;
        font-family: Arial, sans-serif;
    }

    #canvas {

        margin-left: 350px;
    }

    .inputs {
        position: absolute;
        top: 150px;
        left: 10px;
        padding: 5px;
        border: 4px solid black;
        background-color: #f9f9f9;
        border-radius: 10px;
        
    }

    .inputs label {
        display: inline-block;
        width: 80px;
        left: 50px;
        font-weight: bold;
        margin: 0px 0px 0px 15px;
    }

    .inputs input {
        padding: 5px;
        margin-bottom: 10px;
        width: 150px;
        border: 2px solid #ccc;
        border-radius: 5px;
    }

    #bt1 {
        background-color: #031273;
        border: none;
        color: white;
        padding: 10px 22px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 0px 0px 5px;
        cursor: pointer;
        border-radius: 5px;
    }

    #head1 {
        color: #f2f4f7;
        text-align: center;
        background-color: #031273;
        font-size: 18px;
        padding: 15px;
        border-radius: 5px;
    }

    #result {
        text-align: center;
        font-size: 18px;
        margin-top: 20px;
        color: black;
        font-weight: bold;
    }

    #result-box {
        border: 2px solid royalblue;
        padding: 10px;
        margin-top: 20px;
        background-color: royalblue;
        border-radius: 10px; /* Increased border radius for a smoother look */
        font-family: Arial, sans-serif;
        color: #fff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        max-width: 400px; /* Limiting the maximum width */
        margin-left: auto; /* Centering horizontally */
        margin-right: auto;
    }

    #result-box h3 {
        margin: 0 0 10px 0;
        font-size: 24px;
        color: #fff;
    }

    #result-box p {
        margin: 5px 0;
        font-size: 16px;
    }

"""

# Your JS code
js_content = """
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

// https://www.researchgate.net/publication/4345236_A_Software_Implementation_of_the_Duval_Triangle_Method
//To vary the point size of Dot
var pointSize = 4.5;

var v0 = {
  x: 114,
  y: 366
};
var v1 = {
  x: 306,
  y: 30
};
var v2 = {
  x: 498,
  y: 366
};
var triangle = [v0, v1, v2];
//Legends color
ctx.font = '14px arial black';
ctx.fillText("Duval's Triangle DGA", 220, 20, 300);
//PD
ctx.fillStyle = 'rgb(255,0,0)';
ctx.fillRect(50, 454, 20, 10);
//T1
ctx.fillStyle = 'rgb(255,102,153)';
ctx.fillRect(50, 469, 20, 10);
//T2
ctx.fillStyle = 'rgb(255,204,0)';
ctx.fillRect(50, 484, 20, 10);
//T3
ctx.fillStyle = 'rgb(0,0,0)';
ctx.fillRect(50, 499, 20, 10);
//D1
ctx.fillStyle = 'rgb(172,236,222)';
ctx.fillRect(50, 514, 20, 10);
//D2
ctx.fillStyle = 'rgb(51,51,153)';
ctx.fillRect(50, 529, 20, 10);
//DT
ctx.fillStyle = 'rgb(153,0,153)';
ctx.fillRect(50, 544, 20, 10);
ctx.fillStyle="black";
ctx.fillText("Diagnosis Result:",350,538,300);
//TextFields for Gases:
var ch4x, ch4y, c2h4x, c2h4y, c2h2x, c2h2y;
// Define all your segments here
var segments = [{
  points: [{
    x: 114,
    y: 366
  }, {
    x: 281,
    y: 76
  }, {
    x: 324,
    y: 150
  }, {
    x: 201,
    y: 366
  }],
  fill: 'rgb(172,236,222)',
  label: {
    text: 'D1',
    cx: 165,
    cy: 395,
    withLine: false,
    endX: null,
    endY: null
  },
},
{
  points: [{
    x: 385,
    y: 366
  }, {
    x: 201,
    y: 366
  }, {
    x: 324,
    y: 150
  }, {
    x: 356,
    y: 204
  }, {
    x: 321,
    y: 256
  }],
  fill: 'rgb(51,51,153)',
  label: {
    text: 'D2',
    cx: 300,
    cy: 395,
    withLine: false,
    endX: null,
    endY: null
  },
},
{
  points: [{
    x: 297,
    y: 46
  }, {
    x: 392,
    y: 214
  }, {
    x: 372,
    y: 248
  }, {
    x: 441,
    y: 366
  }, {
    x: 385,
    y: 366
  }, {
    x: 321,
    y: 256
  }, {
    x: 356,
    y: 204
  }, {
    x: 281,
    y: 76
  }],
  fill: 'rgb(153,0,153)',
  label: {
    text: 'DT',
    cx: 245,
    cy: 60,
    withLine: true,
    endX: 280,
    endY: 55
  },
},
{
  points: [{
    x: 306,
    y: 30
  }, {
    x: 312,
    y: 40
  }, {
    x: 300,
    y: 40
  }],
  fill: 'rgb(255,0,0)',
  label: {
    text: 'PD',
    cx: 356,
    cy: 40,
    withLine: true,
    endX: 321,
    endY: 40
  },
},
{
  points: [{
    x: 312,
    y: 40
  }, {
    x: 348,
    y: 103
  }, {
    x: 337,
    y: 115
  }, {
    x: 297,
    y: 46
  }, {
    x: 300,
    y: 40
  }],
  fill: 'rgb(255,153,153)',
  label: {
    text: 'T1',
    cx: 375,
    cy: 70,
    withLine: true,
    endX: 340,
    endY: 75
  },
},
{
  points: [{
    x: 348,
    y: 103
  }, {
    x: 402,
    y: 199
  }, {
    x: 392,
    y: 214
  }, {
    x: 337,
    y: 115
  }],
  fill: 'rgb(255,204,0)',
  label: {
    text: 'T2',
    cx: 400,
    cy: 125,
    withLine: true,
    endX: 366,
    endY: 120
  },
},
{
  points: [{
    x: 402,
    y: 199
  }, {
    x: 498,
    y: 366
  }, {
    x: 441,
    y: 366
  }, {
    x: 372,
    y: 248
  }],
  fill: 'rgb(0,0,0)',
  label: {
    text: 'T3',
    cx: 480,
    cy: 270,
    withLine: true,
    endX: 450,
    endY: 270
  },
},
];

// label styles
var labelfontsize = 12;
var labelfontface = 'verdana';
var labelpadding = 3;

// pre-create a canvas-image of the arrowhead
var arrowheadLength = 10;
var arrowheadWidth = 8;
var arrowhead = document.createElement('canvas');
premakeArrowhead();

var legendTexts = ['PD = Partial Discharge',
  'T1 = Thermal fault < 300 celcius',
  'T2 = Thermal fault 300 < T < 700 celcius',
  'T3 = Thermal fault < 300 celcius',
  'D1 = Thermal fault T > 700 celcius',
  'D2 = Discharge of High Energy',
  'DT = Electrical and Thermal'
];


// start drawing
/////////////////////


// draw colored segments inside triangle
for (var i = 0; i < segments.length; i++) {
  drawSegment(segments[i]);
}
// draw ticklines
ticklines(v0, v1, 9, 0, 20);
ticklines(v1, v2, 9, Math.PI * 3 / 4, 20);
ticklines(v2, v0, 9, Math.PI * 5 / 4, 20);
// molecules
moleculeLabel(v0, v1, 100, Math.PI, '% CH4');
moleculeLabel(v1, v2, 100, 0, '% C2H4');
moleculeLabel(v2, v0, 75, Math.PI / 2, '% C2H2');
// draw outer triangle
drawTriangle(triangle);
// draw legend
drawLegend(legendTexts, 75, 450, 15);
// end drawing
/////////////////////

function drawSegment(s) {
  // draw and fill the segment path
  ctx.beginPath();
  ctx.moveTo(s.points[0].x, s.points[0].y);
  for (var i = 1; i < s.points.length; i++) {
    ctx.lineTo(s.points[i].x, s.points[i].y);
  }
  ctx.closePath();
  ctx.fillStyle = s.fill;
  ctx.fill();
  ctx.lineWidth = 0;
  ctx.strokeStyle = 'black';
  ctx.stroke();
  // draw segment's box label
  if (s.label.withLine) {
    lineBoxedLabel(s, labelfontsize, labelfontface, labelpadding);
  } else {
    boxedLabel(s, labelfontsize, labelfontface, labelpadding);
  }
}


function moleculeLabel(start, end, offsetLength, angle, text) {
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = '14px verdana';
  var dx = end.x - start.x;
  var dy = end.y - start.y;
  var x0 = parseInt(start.x + dx * 0.50);
  var y0 = parseInt(start.y + dy * 0.50);
  var x1 = parseInt(x0 + offsetLength * Math.cos(angle));
  var y1 = parseInt(y0 + offsetLength * Math.sin(angle));
  ctx.fillStyle = 'black';
  ctx.fillText(text, x1, y1);
  // arrow
  x0 = parseInt(start.x + dx * 0.35);
  y0 = parseInt(start.y + dy * 0.35);
  x1 = parseInt(x0 + 50 * Math.cos(angle));
  y1 = parseInt(y0 + 50 * Math.sin(angle));
  var x2 = parseInt(start.x + dx * 0.65);
  var y2 = parseInt(start.y + dy * 0.65);
  var x3 = parseInt(x2 + 50 * Math.cos(angle));
  var y3 = parseInt(y2 + 50 * Math.sin(angle));
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x3, y3);
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 1;
  ctx.stroke();
  angle = Math.atan2(dy, dx);
  ctx.translate(x3, y3);
  ctx.rotate(angle);
  ctx.drawImage(arrowhead, -arrowheadLength, -arrowheadWidth / 2);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}
function boxedLabel(s, fontsize, fontface, padding) {
  var centerX = s.label.cx;
  var centerY = s.label.cy;
  var text = s.label.text;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = fontsize + 'px ' + fontface;
  var textwidth = ctx.measureText(text).width;
  var textheight = fontsize * 1.286;
  var leftX = centerX - textwidth / 2 - padding;
  var topY = centerY - textheight / 2 - padding;
  ctx.fillStyle = 'white';
  ctx.fillRect(leftX, topY, textwidth + padding * 2, textheight + padding * 2);
  ctx.lineWidth = 1;
  ctx.strokeRect(leftX, topY, textwidth + padding * 2, textheight + padding * 2);
  ctx.fillStyle = 'black';
  ctx.fillText(text, centerX, centerY);
}


function lineBoxedLabel(s, fontsize, fontface, padding) {
  var centerX = s.label.cx;
  var centerY = s.label.cy;
  var text = s.label.text;
  var lineToX = s.label.endX;
  var lineToY = s.label.endY;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = fontsize + 'px ' + fontface;
  var textwidth = ctx.measureText(text).width;
  var textheight = fontsize * 1.286;
  var leftX = centerX - textwidth / 2 - padding;
  var topY = centerY - textheight / 2 - padding;
  // the line
  ctx.beginPath();
  ctx.moveTo(leftX, topY + textheight / 2);
  ctx.lineTo(lineToX, topY + textheight / 2);
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 1;
  ctx.stroke();
  // the boxed text
  ctx.fillStyle = 'white';
  ctx.fillRect(leftX, topY, textwidth + padding * 2, textheight + padding * 2);
  ctx.strokeRect(leftX, topY, textwidth + padding * 2, textheight + padding * 2);
  ctx.fillStyle = 'black';
  ctx.fillText(text, centerX, centerY);
}


function ticklines(start, end, count, angle, length) {
  var dx = end.x - start.x;
  var dy = end.y - start.y;
  ctx.lineWidth = 1;
  for (var i = 1; i < count; i++) {
    var x0 = parseInt(start.x + dx * i / count);
    var y0 = parseInt(start.y + dy * i / count);
    var x1 = parseInt(x0 + length * Math.cos(angle));
    var y1 = parseInt(y0 + length * Math.sin(angle));
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    if (i == 2 || i == 4 || i == 6 || i == 8) {
      var labelOffset = length * 3 / 4;
      x1 = parseInt(x0 - labelOffset * Math.cos(angle));
      y1 = parseInt(y0 - labelOffset * Math.sin(angle));
      ctx.fillStyle = 'black';
      ctx.fillText(parseInt(i * 10), x1, y1);
    }
  }
}


function premakeArrowhead() {
  var actx = arrowhead.getContext('2d');
  arrowhead.width = arrowheadLength;
  arrowhead.height = arrowheadWidth;
  actx.beginPath();
  actx.moveTo(0, 0);
  actx.lineTo(arrowheadLength, arrowheadWidth / 2);
  actx.lineTo(0, arrowheadWidth);
  actx.closePath();
  actx.fillStyle = 'black';
  actx.fill();
}


function drawTriangle(t) {
  ctx.beginPath();
  ctx.moveTo(t[0].x, t[0].y);
  ctx.lineTo(t[1].x, t[1].y);
  ctx.lineTo(t[2].x, t[2].y);
  ctx.closePath();
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;
  ctx.stroke();
}

//Function to draw legends
function drawLegend(texts, x, y, lineheight) {
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = 'black';
  ctx.font = '14px Verdana';
  for (var i = 0; i < texts.length; i++) {
    ctx.fillText(texts[i], x, y + i * lineheight);
  }
}
//Red Dot function
function drawCoordinates(x, y) {
  ctx.fillStyle = "white"; // Red color
  ctx.beginPath();
  ctx.arc(x, y, pointSize, 0, Math.PI * 2, true);
  ctx.fill();
}
//Function to draw final cords connecting intersection with the % contribution
function drawCords(x, y) {
  ctx.moveTo(x, y);
  ctx.lineTo(ch4x, ch4y);
  ctx.moveTo(x, y);
  ctx.lineTo(c2h4x, c2h4y);
  ctx.moveTo(x, y);
  ctx.lineTo(c2h2x, c2h2y);
  ctx.strokeStyle = 'white';
  ctx.stroke();
}
//Function to fetch the value from database
function calcOprByValue(ch4, c2h2, c2h4) {
  total = ch4 + c2h2 + c2h4;
  var ch4_contr = (ch4 / total);
  var c2h2_contr = (c2h2 / total);
  var c2h4_contr = (c2h4 / total);
  //Draw Bottom Point for bottom line
  var c2h2_line = BottomCoordinates(c2h2_contr);

  //drawCoordinates(c2h2_line.x, c2h2_line.y);
  //Left Coordinates
  var ch4_line = LeftCoordinates(ch4_contr);
  //drawCoordinates(ch4_line.x, ch4_line.y);
  //Right Coordinates
  var c2h4_line = RightCoordinates(c2h4_contr);
  //drawCoordinates(c2h4_line.x, c2h4_line.y);
  //Updating coordinates values
  ch4x = ch4_line.x;
  ch4y = ch4_line.y;
  c2h4x = c2h4_line.x;
  c2h4y = c2h4_line.y;
  c2h2x = c2h2_line.x;
  c2h2y = c2h2_line.y;
  //2 Reflection Coordinates
  var ref_ch4 = refLeftCoordinates(ch4_contr);
  //drawCoordinates(ref_ch4.x,ref_ch4.y);
  var ref_c2h4 = refRightCoordinates(c2h4_contr);
  //drawCoordinates(ref_c2h4.x,ref_c2h4.y);
  var res = checkLineIntersection(ch4_line.x, ch4_line.y, ref_ch4.x, ref_ch4.y, c2h4_line.x, c2h4_line.y, ref_c2h4.x, ref_c2h4.y);
  var color=detectColor(res.x,res.y);
  findAndDisplayColor(color);
  drawCoordinates(res.x, res.y);
  drawCords(res.x, res.y);
}
function findAndDisplayColor(color){
  var red,green,blue;
    red=color.r;
    green=color.g;
    blue=color.b;
    var diagResult;
  if(color.r==255&&color.g==0&&color.b==0){
    diagResult="PD = Partial Discharge";
  }
  else if(color.r==255&&color.g==102&&color.b==153){
    diagResult='T1 = Thermal fault < 300 celcius';
  }
  else if(color.r==255&&color.g==204&&color.b==0){
    diagResult='T2 = Thermal fault 300 < T < 700 celcius';
  }
  else if(color.r==0&&color.g==0&&color.b==0){
    diagResult='T3 = Thermal fault < 300 celcius';
  }
  else if(color.r==172&&color.g==236&&color.b==222){
    diagResult='D1 = Thermal fault T > 700 celcius';
  }
  else if(color.r==51&&color.g==51&&color.b==153){
    diagResult='D2 = Discharge of High Energy';
  }
  else{
    diagResult='DT = Electrical and Thermal';
  }
  ctx.fillStyle = 'rgb('+red+','+green+','+blue+')';
  ctx.fillRect(350, 550, 25, 12);
  ctx.fillStyle="black";
  ctx.fillText(diagResult,380,546,300);
}
//Detect color of perticular pixel
function detectColor(x,y){
 data=ctx.getImageData(x,y,1,1).data;
 col={
   r:data[0],
   g:data[1],
   b:data[2]
 };
 return col;
}
//Function to get trigger while clicking button
function calcOpr() {
  var val1 = parseFloat(document.getElementById("ch4").value);
  var val2 = parseFloat(document.getElementById("c2h2").value);
  var val3 = parseFloat(document.getElementById("c2h4").value);
  calcOprByValue(val1, val2, val3);
}

function refRightCoordinates(c2h4_contr) {
  var dx = (v2.x - v0.x) * c2h4_contr;
  var coor_x = v0.x + dx;
  var coor_y = v0.y;
  return ({
    x: coor_x,
    y: coor_y
  });
}

function refLeftCoordinates(ch4_contr) {
  var l = Math.sqrt(Math.pow((v2.x - v1.x), 2) + Math.pow((v2.y - v1.y), 2));
  var l_eff = l * ch4_contr;
  var coor_x = v2.x - l_eff * Math.cos(Math.PI / 3);
  var coor_y = v2.y - l_eff * Math.sin(Math.PI / 3);
  return ({
    x: coor_x,
    y: coor_y
  });
} // Calculating coordinates with three gases value
function LeftCoordinates(ch4_contr) {
  var l = Math.sqrt(Math.pow((v1.x - v0.x), 2) + Math.pow((v1.y - v0.y), 2));
  var l_eff = l * ch4_contr;
  var coor_x = v0.x + l_eff * Math.cos(Math.PI / 3);
  var coor_y = v0.y - l_eff * Math.sin(Math.PI / 3);
  //console.log(coor1_y);
  return ({
    x: coor_x,
    y: coor_y
  });
}

function BottomCoordinates(c2h2_contr) {
  var dx = (v2.x - v0.x) * c2h2_contr;
  var coor_x = v2.x - dx;
  var coor_y = v0.y;
  return ({
    x: coor_x,
    y: coor_y
  });
}

function RightCoordinates(c2h4_contr) {
  var l = Math.sqrt(Math.pow((v1.x - v2.x), 2) + Math.pow((v1.y - v2.y), 2));
  var l_eff = l * c2h4_contr;
  var coor_x = v1.x + l_eff * Math.cos(Math.PI / 3);
  var coor_y = v1.y + l_eff * Math.sin(Math.PI / 3);
  return ({
    x: coor_x,
    y: coor_y
  });
}
//Intesection and get Point
function checkLineIntersection(line1StartX, line1StartY, line1EndX, line1EndY, line2StartX, line2StartY, line2EndX, line2EndY) {
  // if the lines intersect, the result contains the x and y of the intersection (treating the lines as infinite) and booleans for whether line segment 1 or line segment 2 contain the point
  var denominator, a, b, numerator1, numerator2, result = {
    x: null,
    y: null,
    onLine1: false,
    onLine2: false
  };
  denominator = ((line2EndY - line2StartY) * (line1EndX - line1StartX)) - ((line2EndX - line2StartX) * (line1EndY - line1StartY));
  if (denominator == 0) {
    return result;
  }
  a = line1StartY - line2StartY;
  b = line1StartX - line2StartX;
  numerator1 = ((line2EndX - line2StartX) * a) - ((line2EndY - line2StartY) * b);
  numerator2 = ((line1EndX - line1StartX) * a) - ((line1EndY - line1StartY) * b);
  a = numerator1 / denominator;
  b = numerator2 / denominator;

  // if we cast these lines infinitely in both directions, they intersect here:
  result.x = line1StartX + (a * (line1EndX - line1StartX));
  result.y = line1StartY + (a * (line1EndY - line1StartY));
  /*
          // it is worth noting that this should be the same as:
          x = line2StartX + (b * (line2EndX - line2StartX));
          y = line2StartX + (b * (line2EndY - line2StartY));
          */
  // if line1 is a segment and line2 is infinite, they intersect if:
  if (a > 0 && a < 1) {
    result.onLine1 = true;
  }
  // if line2 is a segment and line1 is infinite, they intersect if:
  if (b > 0 && b < 1) {
    result.onLine2 = true;
  }
  // if line1 and line2 are segments, they intersect if both of the above are true
  return result;
}
"""

# Combine CSS and JS into the HTML content
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>{css_content}</style>
</head>
<body>
    {html_content}
    <script>{js_content}</script>
</body>
</html>
"""
def final_duval():
    # Render the HTML in Streamlit
    st.components.v1.html(full_html, height=600, width= 1050)




def main():
    #add_particles_background()
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
            "Duvals' Triangle":  final_duval,
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
