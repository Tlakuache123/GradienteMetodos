import streamlit as st
import streamlit.components.v1 as components
import sympy as sp
import numpy as np
import mpld3
import matplotlib.pyplot as mlp


# Creating x_n+1
def decenso_gradiente(gradiente, x_0, alpha, n_iteraciones=50, error=1e-06):
    vector = x_0
    vector_history = []
    for _ in range(n_iteraciones):
        vector_history.append(vector)
        diff = -alpha * gradiente(vector)
        if np.all(np.abs(diff) <= error):
            break
        vector += diff
    return vector, vector_history


x, y = sp.symbols("x, y")
f = x**4 - 5 * x**2 - 3 * x
df = f.diff(x)

lamb_fx = sp.lambdify(x, f, modules=["numpy"])
lamb_dfx = sp.lambdify(x, df, modules=["numpy"])


v, v_h = decenso_gradiente(gradiente=lamb_dfx, x_0=0, alpha=0.1)

st.write(
    """
    # Gradiente descenso
"""
)

st.latex(r"x_{n+1} = x_n - \alpha * \nabla f(x)")

st.code(
    """
    def decenso_gradiente(gradiente, x_0, alpha, n_iteraciones=50, error=1e-06):
    vector = x_0
    for _ in range(n_iteraciones):
        diff = -alpha * gradiente(vector)
        if np.all(np.abs(diff) <= error):
            break
        vector += diff
    return vector
    """,
    language="python",
)

st.write("Nuestra funcion a evaluar")
st.latex(f)

st.write("Su derivada")
st.latex(df)

# Plotting function


vals_x = np.linspace(-3, 3, 100)
fx = lamb_fx(vals_x)

fig = mlp.figure()
mlp.plot(vals_x, fx)
mlp.plot(v_h, lamb_fx(np.array(v_h)), "-go")
mlp.plot(v, lamb_fx(v), "ro")
fig_html = mpld3.fig_to_html(fig)

components.html(fig_html, height=600)
