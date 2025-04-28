from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import pandas as pd
import io
import plotly.graph_objects as go

app = FastAPI()

# 🔵 Adăugăm middleware CORS (important pentru frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sau pui domeniul frontend-ului tău pentru mai multă securitate
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = os.path.join("static", "plots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/api/upload")
async def upload_api(file: UploadFile = File(...)):
    try:
        # Citim fișierul primit
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))

        required_cols = {'X', 'Y', 'Z', 'Value'}
        if not required_cols.issubset(df.columns):
            return JSONResponse({"error": "Missing columns"}, status_code=400)

        x = np.unique(df['X'])
        y = np.unique(df['Y'])
        z = np.unique(df['Z'])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        value_cube = np.full_like(X, fill_value=np.nan, dtype=float)

        for _, row in df.iterrows():
            xi = np.where(x == row['X'])[0][0]
            yi = np.where(y == row['Y'])[0][0]
            zi = np.where(z == row['Z'])[0][0]
            value_cube[xi, yi, zi] = row['Value']

        if np.isnan(value_cube).any():
            value_cube = np.where(np.isnan(value_cube), np.nanmean(value_cube), value_cube)

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=value_cube.flatten(),
            isomin=np.min(value_cube),
            isomax=np.max(value_cube),
            opacity=0.15,
            surface_count=20,
            colorscale='Jet',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        fig.update_layout(
            title='Magnetic Field Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            autosize=True
        )

        # Salvează HTML-ul
        filename = "plot.html"
        path = os.path.join(UPLOAD_FOLDER, filename)
        fig.write_html(path, full_html=True, include_plotlyjs='cdn', config={"responsive": True})

        # Returnează URL-ul plotului
        # ATENȚIE: Railway nu garantează că request.host_url e mereu corect, așa că poți trimite doar path-ul relativ
        return {"url": f"/static/plots/{filename}"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
