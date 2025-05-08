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

# CORS pentru acces din frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = os.path.join("static", "plots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === RUTA 3D ===
@app.post("/api/upload")
async def upload_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        df.columns = [col.lower() for col in df.columns]

        required_cols = {'x', 'y', 'z', 'value'}
        if not required_cols.issubset(df.columns):
            return JSONResponse({"error": "Missing columns (x, y, z, value)"}, status_code=400)

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

# === RUTA 2D (Cross-Section) ===
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

        x = x.strip()
        y = y.strip()
        z = z.strip()

        filled = [(name, val) for name, val in zip(["x", "y", "z"], [x, y, z]) if val != ""]
        if len(filled) != 2:
            return JSONResponse({"error": "Please fill in exactly two coordinates."}, status_code=400)

        filters = {}
        for axis, val in filled:
            try:
                filters[axis] = float(val)
            except ValueError:
                return JSONResponse({"error": f"{axis.upper()} must be a number."}, status_code=400)

        # Verificăm dacă valorile sunt în interval
        for axis, val in filters.items():
            if val > df[axis].max() or val < df[axis].min():
                return JSONResponse({"error": f"{axis.upper()}={val} is out of range."}, status_code=400)

        for axis, val in filters.items():
            df = df[df[axis] == val]

        if df.empty:
            return JSONResponse({"error": "No data found for the selected coordinates."}, status_code=404)

        remaining_axis = [axis for axis in ["x", "y", "z"] if axis not in filters][0]
        df_sorted = df.sort_values(by=[remaining_axis])

        fixed_labels = ", ".join([f"{k} = {v}" for k, v in filters.items()])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted[remaining_axis],
            y=df_sorted['value'],
            mode='lines+markers'
        ))

        fig.update_layout(
            title=f'Magnetic Field along {remaining_axis.upper()} ({fixed_labels})',
            xaxis_title=f'{remaining_axis.upper()} (mm)',
            yaxis_title='Magnetic Field Value (T)',
            margin=dict(l=40, r=40, t=60, b=40),
            autosize=True
        )

        # ... după generarea graficului
        html_path = os.path.join(UPLOAD_FOLDER, "plot2d.html")
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True, "displaylogo": False, "modeBarButtonsToAdd": ["toImage"]})
        
        table_data = df_sorted[["x", "y", "z", "value"]].to_dict(orient="records")
        
        return {
            "url": "/static/plots/plot2d.html",
            "table": table_data
        }


    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/api/z-section")
async def z_section_api(file: UploadFile = File(...), z: str = Form(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        df.columns = [col.lower() for col in df.columns]

        if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns or 'value' not in df.columns:
            return JSONResponse({"error": "Missing required columns"}, status_code=400)

        try:
            z_val = float(z.strip())
        except ValueError:
            return JSONResponse({"error": "Z must be a number"}, status_code=400)

        if z_val < df["z"].min() or z_val > df["z"].max():
            return JSONResponse({"error": f"Z={z_val} is out of range."}, status_code=400)

        df_z = df[df["z"] == z_val]
        if df_z.empty:
            return JSONResponse({"error": "No data found for this Z"}, status_code=404)

        # Sort for clean heatmap
        df_z = df_z.sort_values(by=["x", "y"])

        fig = go.Figure(data=go.Contour(
            x=np.sort(df_z["x"].unique()),
            y=np.sort(df_z["y"].unique()),
            z=df_z.pivot(index='y', columns='x', values='value').values,
            colorscale='Jet',
            contours_coloring='heatmap',
            showscale=True
        ))

        fig.update_layout(
            title=f"Z Section at Z = {z_val}",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            margin=dict(l=40, r=40, t=60, b=40),
            autosize=True
        )

        html_path = os.path.join(UPLOAD_FOLDER, "z_section.html")
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True})

        table_data = df_z[["x", "y", "z", "value"]].to_dict(orient="records")
        return {
            "url": "/static/plots/z_section.html",
            "table": table_data,
            "z": z_val
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Servește fișierele statice
app.mount("/static", StaticFiles(directory="static"), name="static")
