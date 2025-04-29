# ✅ FASTAPI - API pentru generarea unui plot 3D interactiv cu posibilitate de export PNG
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Static files serving
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
