# ✅ FASTAPI - API pentru generarea unui plot 3D și 2D (cross section)
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import numpy as np
import pandas as pd
import io
import plotly.graph_objects as go

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = os.path.join("static", "plots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔵 Ruta pentru 3D
@app.post("/api/upload")
async def upload_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))

        required_cols = {'x', 'y', 'z', 'value'}
        if not required_cols.issubset(set(col.lower() for col in df.columns)):
            return JSONResponse({"error": "Missing columns (x, y, z, value)"}, status_code=400)

        df.columns = [col.lower() for col in df.columns]
        x = np.unique(df['x'])
        y = np.unique(df['y'])
        z = np.unique(df['z'])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        value_cube = np.full_like(X, fill_value=np.nan, dtype=float)

        for _, row in df.iterrows():
            xi = np.where(x == row['x'])[0][0]
            yi = np.where(y == row['y'])[0][0]
            zi = np.where(z == row['z'])[0][0]
            value_cube[xi, yi, zi] = row['value']

        if np.isnan(value_cube).any():
            value_cube = np.where(np.isnan(value_cube), np.nanmean(value_cube), value_cube)

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=value_cube.flatten(),
            isomin=0.05,
            isomax=np.max(value_cube),
            opacity=0.1,
            surface_count=25,
            colorscale='Jet',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        fig.update_layout(
            title='Magnetic Field 3D',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        html_path = os.path.join(UPLOAD_FOLDER, "plot.html")
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True, "displaylogo": False, "modeBarButtonsToAdd": ["toImage"]})

        return {"url": "/static/plots/plot.html"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 🔵 Ruta pentru 2D
@app.post("/api/upload-2d")
async def upload_2d_api(
    file: UploadFile = File(...),
    x: str = Form(""),
    y: str = Form(""),
    z: str = Form("")
):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        df.columns = [col.lower() for col in df.columns]

        # Convertim inputul
        x = x.strip()
        y = y.strip()
        z = z.strip()

        # Validare: trebuie exact două coordonate completate
        filled = [(name, val) for name, val in zip(["x", "y", "z"], [x, y, z]) if val != ""]
        if len(filled) != 2:
            return JSONResponse({"error": "Exactly two coordinates must be filled."}, status_code=400)

        filters = {}
        for axis, val in filled:
            try:
                filters[axis] = float(val)
            except ValueError:
                return JSONResponse({"error": f"{axis.upper()} must be numeric."}, status_code=400)

        # Filtrăm dataframe-ul
        for axis, val in filters.items():
            df = df[df[axis] == val]

        if df.empty:
            return JSONResponse({"error": "No data found for the selected section."}, status_code=404)

        # Determinăm ce axă a rămas variabilă
        remaining_axis = [axis for axis in ["x", "y", "z"] if axis not in filters][0]

        df_sorted = df.sort_values(by=[remaining_axis])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted[remaining_axis],
            y=df_sorted['value'],
            mode='lines+markers'
        ))

        fig.update_layout(
            title=f'Magnetic Field along {remaining_axis.upper()}',
            xaxis_title=f'{remaining_axis.upper()} (mm)',
            yaxis_title='Magnetic Field Value (T)',
            margin=dict(l=40, r=40, t=40, b=40),
            autosize=True
        )

        html_path = os.path.join(UPLOAD_FOLDER, "plot2d.html")
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True, "displaylogo": False, "modeBarButtonsToAdd": ["toImage"]})

        return {"url": "/static/plots/plot2d.html"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Static files serving
app.mount("/static", StaticFiles(directory="static"), name="static")
