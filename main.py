import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from funciones_gradiente import decenso_gradiente, sgd


x, y = sp.symbols("x, y")
f = x**4 - 5 * x**2 - 3 * x
df = f.diff(x)

lamb_fx = sp.lambdify(x, f, modules=["numpy"])
lamb_dfx = sp.lambdify(x, df, modules=["numpy"])

v, v_h = decenso_gradiente(gradiente=lamb_dfx, x_0=0, alpha=0.1)

vals_x = np.linspace(-3, 3, 100)
fx = lamb_fx(vals_x)
vfx = lamb_fx(np.array(v_h))


st.write(
    """
    # Descenso por gradiente
"""
)

st.latex(r"x_{n+1} = x_n - \alpha * \nabla f(x)")

st.code(
    """
    import numpy as np

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
fig1 = px.line(x=vals_x, y=fx)
fig1.update_traces(line=dict(color="rgba(125,125,125,0.6)"))

fig2 = px.line(x=v_h, y=vfx, markers=True)

fig = go.Figure(data=fig1.data + fig2.data)

st.write(
    r"""
### Caracteristicas
- $\alpha$: 0.1
- $x_0$: 0
- Maximo de iteraciones: 100
- Error: 1e-06
"""
)
st.plotly_chart(fig, use_container_width=True)
st.write(f"Minimo calculado f({v}) = {lamb_fx(v)}")
