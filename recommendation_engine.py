# =====================================================
# ADVANCED XAI RECOMMENDATION ENGINE
# =====================================================

import numpy as np

# =====================================================
# SEVERITY CALCULATOR
# =====================================================

def calculate_severity(

    shap_value,

    lime_value
):

    combined = (
        abs(shap_value) +
        abs(lime_value)
    ) / 2

    if combined > 1.5:

        return "Critical"

    elif combined > 0.8:

        return "High"

    elif combined > 0.4:

        return "Moderate"

    else:

        return "Low"

# =====================================================
# MAIN ENGINE
# =====================================================

def generate_recommendations(

    user_input,

    shap_data,

    lime_df
):

    recommendations = []

    # =================================================
    # FEATURE RULES
    # =================================================

    rules = {

        "BS": {

            "high": 8.5,

            "observation":
                "Elevated blood sugar levels detected.",

            "clinical_risk":
                "Potential gestational diabetes complications.",

            "recommendation":
                "Immediate glucose monitoring, diabetic diet planning, and endocrinology consultation recommended."
        },

        "SystolicBP": {

            "high": 140,

            "observation":
                "Elevated systolic blood pressure detected.",

            "clinical_risk":
                "Possible hypertensive disorder during pregnancy.",

            "recommendation":
                "Regular BP monitoring, reduced sodium intake, and obstetric consultation recommended."
        },

        "DiastolicBP": {

            "high": 90,

            "observation":
                "Elevated diastolic blood pressure detected.",

            "clinical_risk":
                "Increased risk of preeclampsia.",

            "recommendation":
                "Close maternal monitoring and cardiovascular evaluation advised."
        },

        "BMI": {

            "high": 30,

            "observation":
                "High BMI observed.",

            "clinical_risk":
                "Elevated pregnancy-related metabolic risk.",

            "recommendation":
                "Nutrition management and supervised maternal fitness recommended."
        },

        "BodyTemp": {

            "high": 99.5,

            "observation":
                "Elevated body temperature detected.",

            "clinical_risk":
                "Possible infection or inflammatory condition.",

            "recommendation":
                "Clinical temperature assessment and infection screening recommended."
        },

        "HeartRate": {

            "high": 110,

            "observation":
                "Elevated maternal heart rate detected.",

            "clinical_risk":
                "Possible cardiovascular strain or stress response.",

            "recommendation":
                "Cardiac evaluation and hydration assessment recommended."
        },

        "MentalHealth": {

            "high": 7,

            "observation":
                "Mental stress indicators elevated.",

            "clinical_risk":
                "Potential maternal emotional distress.",

            "recommendation":
                "Psychological support and stress management recommended."
        }
    }

    # =================================================
    # BUILD LIME MAP
    # =================================================

    lime_map = {}

    for _, row in lime_df.iterrows():

        lime_map[
            row["Feature"]
        ] = row["Contribution"]

    # =================================================
    # PROCESS FEATURES
    # =================================================

    for i, feature in enumerate(
        user_input.keys()
    ):

        if feature not in rules:

            continue

        value = user_input[feature]

        shap_value = float(
            shap_data[i]
        )

        lime_value = 0.0

        # =============================================
        # MATCH LIME FEATURE
        # =============================================

        for lime_feature in lime_map:

            if feature.lower() in lime_feature.lower():

                lime_value = float(
                    lime_map[lime_feature]
                )

                break

        # =============================================
        # FILTER IMPORTANT FEATURES
        # =============================================

        if (
            abs(shap_value) < 0.15 and
            abs(lime_value) < 0.15
        ):

            continue

        # =============================================
        # RULES
        # =============================================

        rule = rules[feature]

        severity = calculate_severity(

            shap_value,

            lime_value
        )

        # =============================================
        # DYNAMIC RECOMMENDATION
        # =============================================

        recommendation_text = (
            rule["recommendation"]
        )

        if severity == "Critical":

            recommendation_text += (
                " Immediate physician intervention advised."
            )

        elif severity == "High":

            recommendation_text += (
                " Frequent maternal monitoring recommended."
            )

        recommendations.append({

            "feature":
                feature,

            "value":
                value,

            "severity":
                severity,

            "shap_value":
                round(shap_value, 3),

            "lime_value":
                round(lime_value, 3),

            "observation":
                rule["observation"],

            "clinical_risk":
                rule["clinical_risk"],

            "recommendation":
                recommendation_text
        })

    # =================================================
    # SORT BY SEVERITY
    # =================================================

    severity_order = {

        "Critical": 4,

        "High": 3,

        "Moderate": 2,

        "Low": 1
    }

    recommendations = sorted(

        recommendations,

        key=lambda x:
            severity_order[
                x["severity"]
            ],

        reverse=True
    )

    return recommendations