import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from funciones_gradiente import mini_batch

st.markdown("# Decenso por gradiente (Mini-Batch)")

height = np.array([167, 145, 170, 180, 189, 155, 163, 178, 173, 176])
weight = height * 0.5

data = pd.DataFrame(list(zip(height, weight)), columns=["height", "weight"])
x = data["height"]
y = data["weight"]

st.code(
    """
def batch(x, y, w, gamma=0.001, iteraciones=60):
    x = x
    y = y
    gamma = gamma
    w = w
    iteraciones = iteraciones
    all_costs = []

    for k in range(iteraciones):
        cost = 0
        for i in range(len(x)):
            a = costo_individual_punto(x[i], y[i], w)
            cost += a
        cost_med = cost / len(x)
        all_costs.append(cost_med)
        grad_w = 0
        for j in range(len(x)):
            b = grad(x[j], y[j], w)
            grad_w += b
        grad_w_med = grad_w / len(x)
        w = w - (gamma * grad_w_med)
    return w, all_costs
"""
)

st.dataframe(data)

iteraciones = 60
w, all_costs = mini_batch(x=x, y=y, w=0, gamma=0.000001)

fig1 = px.line(x=np.arange(iteraciones), y=all_costs, markers=True)
fig = go.Figure(data=fig1.data)


st.write(f"Optimal value: {w}")
st.plotly_chart(fig)
