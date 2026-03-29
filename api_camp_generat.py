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

FLOAT_TOLERANCE = 1e-6


def read_dataframe(contents: bytes) -> pd.DataFrame:
    """Read Excel file, auto-detecting whether a header row is present."""
    df = pd.read_excel(io.BytesIO(contents))
    df.columns = [str(col).lower().strip() for col in df.columns]

    required_cols = {'x', 'y', 'z', 'value'}
    if not required_cols.issubset(df.columns):
        # No proper header — re-read without header and assign column names
        df = pd.read_excel(io.BytesIO(contents), header=None)
        if len(df.columns) == 4:
            df.columns = ['x', 'y', 'z', 'value']
        else:
            raise ValueError(
                f"Expected 4 columns (x, y, z, value) but found {len(df.columns)}. "
                "Please ensure the file has columns: x, y, z, value."
            )

    for col in ['x', 'y', 'z', 'value']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def filter_by_value(df: pd.DataFrame, axis: str, val: float) -> pd.DataFrame:
    """Filter rows where axis == val, using tolerance to handle float precision."""
    return df[(df[axis] - val).abs() <= FLOAT_TOLERANCE]


@app.post("/api/upload")
async def upload_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = read_dataframe(contents)

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

        val_min = float(np.nanmin(value_cube))
        val_max = float(np.nanmax(value_cube))

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=value_cube.flatten(),
            isomin=val_min,
            isomax=val_max,
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

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/upload-2d")
async def upload_2d_api(file: UploadFile = File(...), x: str = Form(""), y: str = Form(""), z: str = Form("")):
    try:
        contents = await file.read()
        df = read_dataframe(contents)

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
            if val > df[axis].max() + FLOAT_TOLERANCE or val < df[axis].min() - FLOAT_TOLERANCE:
                return JSONResponse({"error": f"{axis.upper()}={val} is out of range [{df[axis].min()}, {df[axis].max()}]."}, status_code=400)

        filtered_df = df.copy()
        for axis, val in filters.items():
            filtered_df = filter_by_value(filtered_df, axis, val)

        if filtered_df.empty:
            return JSONResponse({"error": "No data found for the selected coordinates. Check that the values exist in the dataset."}, status_code=404)

        remaining_axis = [axis for axis in ["x", "y", "z"] if axis not in filters][0]
        df_sorted = filtered_df.sort_values(by=[remaining_axis])

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

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/section-1d")
async def section_1d(file: UploadFile = File(...), axis: str = Form(...), value: str = Form(...)):
    try:
        contents = await file.read()
        df = read_dataframe(contents)

        if axis not in ["x", "y", "z"]:
            return JSONResponse({"error": "Invalid axis. Must be one of x, y, z."}, status_code=400)

        try:
            value = float(value.strip())
        except ValueError:
            return JSONResponse({"error": f"{axis.upper()} must be a number."}, status_code=400)

        if value < df[axis].min() - FLOAT_TOLERANCE or value > df[axis].max() + FLOAT_TOLERANCE:
            return JSONResponse({"error": f"{axis.upper()}={value} is out of range [{df[axis].min()}, {df[axis].max()}]."}, status_code=400)

        df_filtered = filter_by_value(df, axis, value)
        if df_filtered.empty:
            return JSONResponse({"error": f"No data found for {axis}={value}. Check that this value exists in the dataset."}, status_code=404)

        axes = [a for a in ["x", "y", "z"] if a != axis]
        pivot_table = df_filtered.pivot_table(index=axes[1], columns=axes[0], values="value", aggfunc="mean")

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

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

app.mount("/static", StaticFiles(directory="static"), name="static")
