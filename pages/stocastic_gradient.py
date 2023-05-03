import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from funciones_gradiente import sdg

st.markdown("# Decenso por gradiente (Estocastico)")

height = np.array([167, 145, 170, 180, 189, 155, 163, 178, 173, 176])
weight = height * 0.5

data = pd.DataFrame(list(zip(height, weight)), columns=["height", "weight"])
x = data["height"]
y = data["weight"]


st.code(
    """
def sdg(x, y, w, gamma=0.001, iteraciones=60):
    x = x
    y = y
    all_costs = []
    w = w
    gamma = gamma
    iteraciones = iteraciones

    for k in range(iteraciones):
        for i in range(len(x)):
            cost = costo_individual_punto(x[i], y[i], w)
            grad_w = grad(x[i], y[i], w)
            w = w - (gamma * grad_w)  # in sgd update takes place after every point
        all_costs.append(cost)
    return w, all_costs
    """
)
st.dataframe(data)


iteraciones = 60
w, all_costs = sdg(x=x, y=y, w=0, gamma=0.000001, iteraciones=iteraciones)

fig1 = px.line(x=np.arange(iteraciones), y=all_costs, markers=True)
fig = go.Figure(data=fig1.data)

st.write(f"Optimal value: {w}")
st.plotly_chart(fig)
