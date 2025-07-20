import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

model = joblib.load('iris_model.pkl')
iris = load_iris()

st.title("Iris Flower Prediction App")
st.markdown("Enter flower measurements to predict the Iris species.")

st.sidebar.header("Feature Inputs")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_df = pd.DataFrame(input_data, columns=iris.feature_names)

st.subheader("Input Data")
st.dataframe(input_df)

prediction = model.predict(input_data)
pred_proba = model.predict_proba(input_data)

st.subheader("Prediction")
st.write(f"**Predicted Class:** {iris.target_names[prediction[0]]}")
st.write("**Class Probabilities:**")
st.bar_chart(pred_proba[0])

st.subheader("Feature Importances")
feature_importance = pd.Series(model.feature_importances_, index=iris.feature_names)
fig, ax = plt.subplots()
sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax)
ax.set_xlabel("Importance Score")
st.pyplot(fig)
