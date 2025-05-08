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
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True})

        return {"url": "/static/plots/plot.html"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/upload-2d")
async def upload_2d_api(file: UploadFile = File(...), x: str = Form(""), y: str = Form(""), z: str = Form("")):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        df.columns = [col.lower() for col in df.columns]

        filled = [(name, val.strip()) for name, val in zip(["x", "y", "z"], [x, y, z]) if val.strip() != ""]
        if len(filled) != 2:
            return JSONResponse({"error": "Please fill in exactly two coordinates."}, status_code=400)

        filters = {}
        for axis, val in filled:
            try:
                filters[axis] = float(val)
            except ValueError:
                return JSONResponse({"error": f"{axis.upper()} must be a number."}, status_code=400)

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
        fig.add_trace(go.Scatter(x=df_sorted[remaining_axis], y=df_sorted['value'], mode='lines+markers'))

        fig.update_layout(
            title=f'Magnetic Field along {remaining_axis.upper()} ({fixed_labels})',
            xaxis_title=f'{remaining_axis.upper()} (mm)',
            yaxis_title='Magnetic Field Value (T)',
            margin=dict(l=40, r=40, t=60, b=40),
            autosize=True
        )

        html_path = os.path.join(UPLOAD_FOLDER, "plot2d.html")
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True})

        table_data = df_sorted[["x", "y", "z", "value"]].to_dict(orient="records")
        return {"url": "/static/plots/plot2d.html", "table": table_data}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/section-1d")
async def section_1d(file: UploadFile = File(...), axis: str = Form(...), value: str = Form(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        df.columns = [col.lower() for col in df.columns]

        if axis not in ["x", "y", "z"]:
            return JSONResponse({"error": "Invalid axis. Must be one of x, y, z."}, status_code=400)

        try:
            value = float(value.strip())
        except ValueError:
            return JSONResponse({"error": f"{axis.upper()} must be a number."}, status_code=400)

        if value < df[axis].min() or value > df[axis].max():
            return JSONResponse({"error": f"{axis.upper()}={value} is out of range."}, status_code=400)

        df_filtered = df[df[axis] == value]
        if df_filtered.empty:
            return JSONResponse({"error": f"No data found for {axis}={value}"}, status_code=404)

        axes = [a for a in ["x", "y", "z"] if a != axis]
        pivot_table = df_filtered.pivot(index=axes[1], columns=axes[0], values="value")

        fig = go.Figure(data=go.Contour(
            x=np.sort(df_filtered[axes[0]].unique()),
            y=np.sort(df_filtered[axes[1]].unique()),
            z=pivot_table.values,
            colorscale='Jet',
            contours_coloring='heatmap',
            showscale=True
        ))

        fig.update_layout(
            title=f"Section at {axis.upper()} = {value}",
            xaxis_title=f"{axes[0].upper()} (mm)",
            yaxis_title=f"{axes[1].upper()} (mm)",
            margin=dict(l=40, r=40, t=60, b=40),
            autosize=True
        )

        filename = f"section_{axis}{int(value)}.html"
        html_path = os.path.join(UPLOAD_FOLDER, filename)
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True})

        table_data = df_filtered[["x", "y", "z", "value"]].to_dict(orient="records")
        return {"url": f"/static/plots/{filename}", "table": table_data}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

app.mount("/static", StaticFiles(directory="static"), name="static")
