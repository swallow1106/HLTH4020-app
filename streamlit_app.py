import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import statsmodels.formula.api as smf

# Streamlit App Title
st.title("📊 HLTH 4020 Python Analysis using Panda/Numpy/MatPlot/StatsModels")

# Upload CSV File
uploaded_file = st.file_uploader("CSV file", type=["csv"])

if uploaded_file:
    # Read CSV into Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    st.subheader("🔹 Raw Data (Before Transformation)")
    st.dataframe(df)

    # --- Step 1: Convert a Numeric Column into Y/N ---
    st.subheader("🧮 Convert to Binary for Odds Ratio Analysis")

    binary_column = st.selectbox("Select Column to Convert to Yes/No", df.columns)
    cutoff = st.number_input("Enter Cutoff Value", min_value=float(df[binary_column].min()),
                             max_value=float(df[binary_column].max()), value=float(df[binary_column].mean()))

    # Copy dataset before transformation
    df_transformed = df.copy()

    # Convert selected column to Satisfaction_YN in a new column
    df_transformed["Satisfaction_YN"] = np.where(df_transformed[binary_column] > cutoff, "Y", "N")

    st.write(f"Converted `{binary_column}` into Yes/No based on cutoff `{cutoff}`")
    st.dataframe(df_transformed[[binary_column, 'Satisfaction_YN']])

    # Ensure dataset still contains all independent variables
    st.subheader("🔹 Data After Transformation (All Columns Retained)")
    st.dataframe(df_transformed)

    # Check if both Y and N exist in the dataset
    if df_transformed["Satisfaction_YN"].nunique() < 2:
        st.warning("⚠️ Warning: The dataset only has one category after transformation. Adjust cutoff value!")
    else:
        # Convert Satisfaction_YN to numeric for logistic regression
        df_transformed["Satisfaction_YN_Num"] = df_transformed["Satisfaction_YN"].map({"Y": 1, "N": 0})

        # --- Step 2: Odds Ratio Analysis ---
        st.subheader("📈 Odds Ratio Analysis (with 95% Confidence Intervals)")

        # Select Independent Variables for Odds Ratio Analysis
        X_columns = st.multiselect("Select Independent Variables for Odds Ratio Analysis", df.columns)

        if X_columns:
            # Fix column names (remove spaces, special chars)
            df_transformed.columns = df_transformed.columns.str.replace(" ", "_").str.replace("-", "_")
            X_columns = [col.replace(" ", "_").replace("-", "_") for col in X_columns]

            # Build logistic regression formula
            formula = "Satisfaction_YN_Num ~ " + " + ".join(X_columns)

            # Run Logistic Regression
            model = smf.logit(formula=formula, data=df_transformed).fit()

            # Show Summary
            st.write("### Logistic Regression Summary")
            st.text(model.summary())

            # Compute Confidence Intervals
            params = model.params
            conf = model.conf_int()
            conf['OR'] = np.exp(params)
            conf.columns = ['CI Lower', 'CI Upper', 'Odds Ratio']
            conf['P-Value'] = model.pvalues
            conf = conf.loc[X_columns]  # Select only relevant variables

            # Display Results
            st.write("### Odds Ratio with Confidence Intervals")
            st.dataframe(conf)

            # --- Odds Ratio Visualization with Confidence Intervals ---
            st.subheader("📊 Odds Ratio Visualization (with CI)")

            # Compute error bars correctly (Ensure CI bounds are valid)
            conf["CI Lower Error"] = np.maximum(conf["Odds Ratio"] - conf["CI Lower"], 0)
            conf["CI Upper Error"] = np.maximum(conf["CI Upper"] - conf["Odds Ratio"], 0)

            fig, ax = plt.subplots()
            ax.errorbar(conf["Odds Ratio"], conf.index,
                        xerr=[conf["CI Lower Error"], conf["CI Upper Error"]],
                        fmt="o", capsize=5)

            ax.set_xscale("log")
            ax.set_xlabel("Odds Ratio (log scale)")
            ax.axvline(x=1, color="gray", linestyle="--")  # Reference line at OR = 1
            st.pyplot(fig)

    # --- Step 3: Linear Regression Analysis (Optional) ---
    st.subheader("📉 Linear Regression Analysis")

    Y_column = st.selectbox("Select Dependent Variable for Regression", df_transformed.columns)

    if X_columns and Y_column:
        X = df_transformed[X_columns]
        y = df_transformed[Y_column]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Model Evaluation
        st.subheader("🔍 Regression Results")
        st.write(f"**Intercept:** {model.intercept_}")
        coef_df = pd.DataFrame(model.coef_, index=X_columns, columns=["Coefficient"])
        st.dataframe(coef_df)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**Mean Squared Error:** {mse:.4f}")

        # Scatter Plot for Simple Regression
        if len(X_columns) == 1:
            st.subheader("📊 Regression Plot")
            fig, ax = plt.subplots()
            sns.scatterplot(x=X_test[X_columns[0]], y=y_test, ax=ax, label="Actual")
            sns.lineplot(x=X_test[X_columns[0]], y=y_pred, ax=ax, color="red", label="Predicted")
            ax.set_xlabel(X_columns[0])
            ax.set_ylabel(Y_column)
            st.pyplot(fig)

    # --- Step 4: Variance Inflation Factor (VIF) ---
    st.subheader("📊 Variance Inflation Factor (VIF)")

    if X_columns:
        X_vif = sm.add_constant(df_transformed[X_columns])  # Add constant for regression
        vif_df = pd.DataFrame({
            "Variable": X_vif.columns,
            "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        })

        st.dataframe(vif_df)

    # --- Step 5: Download Modified Data ---
    csv = df_transformed.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Modified CSV", data=csv, file_name="modified_data.csv", mime="text/csv")
    # --- Step 6: Coefficient Adjustability & Prediction Accuracy Monitor ---
    if X_columns:
        formula = "Satisfaction_YN_Num ~ " + " + ".join(X_columns)
        logistic_model = smf.logit(formula=formula, data=df_transformed).fit()
        st.write("### Logistic Regression Summary")
        st.text(logistic_model.summary())

    st.subheader("🔍 Adjust Coefficients for Multiple Variables and Monitor Prediction Accuracy")

    # Assume you've already fit your logistic regression model using statsmodels, for example:
    #   formula = "Satisfaction_YN_Num ~ " + " + ".join(X_columns)
    #   logistic_model = smf.logit(formula=formula, data=df_transformed).fit()
    # And that df_transformed and X_columns are defined.

    # Build the design matrix for prediction from the full dataset used in the model:
    X_model = df_transformed[X_columns]
    X_design = sm.add_constant(X_model)

    # Let the user select multiple variables to adjust.
    adj_vars = st.multiselect("Select Variables to Adjust:", X_columns, key="adj_vars")

    # Initialize a dictionary to hold new coefficient adjustments.
    new_coeffs = {}

    # For each selected variable, create a slider to adjust its coefficient.
    for var in adj_vars:
        # Use a unique key per variable (e.g., "adjust_<var>")
        default_coeff = logistic_model.params[var]
        new_value = st.slider(
            f"Adjust coefficient for {var} (default = {default_coeff:.3f})",
            min_value=float(default_coeff * 0.5),
            max_value=float(default_coeff * 1.5),
            value=float(default_coeff),
            step=0.01,
            key=f"adjust_{var}"
        )
        new_coeffs[var] = new_value

    # Create a new parameter vector, starting with the fitted ones:
    new_params = logistic_model.params.copy()
    # Update selected variables with the adjusted coefficients:
    for var, coeff in new_coeffs.items():
        new_params[var] = coeff

    # Compute new predicted probabilities over the design matrix:
    new_pred_probs = 1 / (1 + np.exp(-np.dot(X_design, new_params)))
    new_pred_classes = (new_pred_probs > 0.5).astype(int)

    # To ensure we're comparing apples to apples, extract the corresponding true labels:
    y_true = df_transformed.loc[X_design.index, "Satisfaction_YN_Num"].astype(int)

    # Compute performance metrics:
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    accuracy_adjusted = accuracy_score(y_true, new_pred_classes)
    conf_matrix_adjusted = confusion_matrix(y_true, new_pred_classes)
    class_report_adjusted = classification_report(y_true, new_pred_classes)

    st.write("### Adjusted Prediction Performance")
    st.write(f"**Accuracy:** {accuracy_adjusted:.2f}")
    st.write("**Confusion Matrix:**")
    st.dataframe(pd.DataFrame(conf_matrix_adjusted, index=["Actual N", "Actual Y"],
                              columns=["Predicted N", "Predicted Y"]))
    st.write("**Classification Report:**")
    st.text(class_report_adjusted)

    # Optional: Plot the distribution of new predicted probabilities.
    fig_adj, ax_adj = plt.subplots(figsize=(6, 4))
    sns.histplot(new_pred_probs, bins=20, kde=True, ax=ax_adj)
    ax_adj.set_xlabel("Predicted Probability")
    ax_adj.set_ylabel("Frequency")
    ax_adj.set_title("Distribution of Predicted Probabilities with Adjusted Coefficients")
    st.pyplot(fig_adj)
