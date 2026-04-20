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

# Column aliases for the "new" measurement format
NEW_FORMAT_COORD_COLS = {
    'x': ['x_work_mm', 'x_mm', 'x'],
    'y': ['y_work_mm', 'y_mm', 'y'],
    'z': ['z_work_mm', 'z_mm', 'z'],
}
NEW_FORMAT_VALUE_COLS = ['value_t', 'value', 'b', 'bz']


def _pick_col(cols_lower, candidates):
    for c in candidates:
        if c in cols_lower:
            return c
    return None


def read_dataframe(contents: bytes, filename: str = "") -> pd.DataFrame:
    """Read either the old format (x, y, z, value) as xlsx/csv or the new
    measurement format (x_work_mm, y_work_mm, z_work_mm, value_T, region, ...).

    Returns a DataFrame with normalized columns: x, y, z, value
    and — when present — an optional 'region' column (uppercased).
    """
    lower_name = (filename or "").lower()

    # Try to read based on extension, with sensible fallback.
    def _try_csv():
        return pd.read_csv(io.BytesIO(contents))

    def _try_xlsx():
        return pd.read_excel(io.BytesIO(contents))

    df = None
    if lower_name.endswith(".csv"):
        df = _try_csv()
    elif lower_name.endswith((".xlsx", ".xls", ".xlsm")):
        df = _try_xlsx()
    else:
        # Unknown extension — try CSV first (cheaper), then xlsx.
        try:
            df = _try_csv()
        except Exception:
            df = _try_xlsx()

    df.columns = [str(col).lower().strip() for col in df.columns]
    cols = set(df.columns)

    # --- New format detection: look for any recognised coordinate + value col
    x_col = _pick_col(cols, NEW_FORMAT_COORD_COLS['x'])
    y_col = _pick_col(cols, NEW_FORMAT_COORD_COLS['y'])
    z_col = _pick_col(cols, NEW_FORMAT_COORD_COLS['z'])
    v_col = _pick_col(cols, NEW_FORMAT_VALUE_COLS)

    if x_col and y_col and z_col and v_col and (
        x_col != 'x' or y_col != 'y' or z_col != 'z' or v_col != 'value' or 'region' in cols
    ):
        # This is the new-format branch (or an old-format file that happens to
        # use these exact names — behaviour is the same either way).
        result = pd.DataFrame({
            'x': pd.to_numeric(df[x_col], errors='coerce'),
            'y': pd.to_numeric(df[y_col], errors='coerce'),
            'z': pd.to_numeric(df[z_col], errors='coerce'),
            'value': pd.to_numeric(df[v_col], errors='coerce'),
        })
        if 'region' in cols:
            result['region'] = df['region'].astype(str).str.upper().str.strip()
        result = result.dropna(subset=['x', 'y', 'z', 'value']).reset_index(drop=True)
        return result

    # --- Old format: plain x, y, z, value with header
    if {'x', 'y', 'z', 'value'}.issubset(cols):
        for col in ['x', 'y', 'z', 'value']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df[['x', 'y', 'z', 'value']].dropna().reset_index(drop=True)

    # --- Old format without header
    if lower_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents), header=None)
    else:
        df = pd.read_excel(io.BytesIO(contents), header=None)

    if len(df.columns) == 4:
        df.columns = ['x', 'y', 'z', 'value']
        for col in ['x', 'y', 'z', 'value']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df[['x', 'y', 'z', 'value']].dropna().reset_index(drop=True)

    raise ValueError(
        "Unrecognized file format. Expected either:\n"
        "  • New format columns: x_work_mm, y_work_mm, z_work_mm, value_T (optional: region)\n"
        "  • Old format columns: x, y, z, value"
    )


def filter_by_value(df: pd.DataFrame, axis: str, val: float) -> pd.DataFrame:
    """Filter rows where axis == val, using tolerance to handle float precision."""
    return df[(df[axis] - val).abs() <= FLOAT_TOLERANCE]


def _build_value_cube(df: pd.DataFrame):
    """Build a regular (x,y,z) grid and place values on it. Returns
    (x, y, z, X, Y, Z, full_cube, region_cube_or_None)."""
    x = np.sort(df['x'].unique())
    y = np.sort(df['y'].unique())
    z = np.sort(df['z'].unique())
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    full_cube = np.full(X.shape, np.nan, dtype=float)

    # Vectorised index lookup — faster than iterrows for large files.
    xi = np.searchsorted(x, df['x'].values)
    yi = np.searchsorted(y, df['y'].values)
    zi = np.searchsorted(z, df['z'].values)
    full_cube[xi, yi, zi] = df['value'].values

    region_cube = None
    if 'region' in df.columns:
        region_cube = np.full(X.shape, '', dtype=object)
        region_cube[xi, yi, zi] = df['region'].values

    return x, y, z, X, Y, Z, full_cube, region_cube


@app.post("/api/upload")
async def upload_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = read_dataframe(contents, file.filename or "")

        # NB: bool(...) — pandas/numpy returns numpy.bool_, which Plotly's strict
        # layout validator rejects for the showlegend property.
        has_region = bool('region' in df.columns and df['region'].notna().any())

        x, y, z, X, Y, Z, full_cube, region_cube = _build_value_cube(df)

        # Overall value range (ignore any remaining NaN holes in the full grid).
        val_min = float(np.nanmin(full_cube))
        val_max = float(np.nanmax(full_cube))
        val_mean = float(np.nanmean(full_cube))

        # For the main Volume trace, fill NaN holes with the mean so
        # interpolation doesn't break, same behaviour as the original script.
        full_filled = np.where(np.isnan(full_cube), val_mean, full_cube)

        # Sentinel below isomin — Plotly clips it out of the rendered isosurfaces.
        span = max(val_max - val_min, 1e-9)
        sentinel = val_min - span * 100.0

        traces = []

        # ── Trace 1: full magnetic-field Volume (always visible) ───────────
        traces.append(go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=full_filled.flatten(),
            isomin=val_min, isomax=val_max,
            opacity=0.1,
            surface_count=25,
            colorscale='Jet',
            caps=dict(x_show=False, y_show=False, z_show=False),
            name='Full field',
            showscale=True,
            colorbar=dict(title='B (T)'),
        ))

        if has_region:
            magnet_mask = (region_cube == 'MAGNET')
            extended_mask = (region_cube == 'EXTENDED')

            # ── Trace 2: MAGNET shell (convex hull, semi-transparent green) ─
            mag_pts = df[df['region'] == 'MAGNET'][['x', 'y', 'z']].values
            if len(mag_pts) >= 4:
                traces.append(go.Mesh3d(
                    x=mag_pts[:, 0], y=mag_pts[:, 1], z=mag_pts[:, 2],
                    alphahull=0,            # convex hull
                    color='lightgreen',
                    opacity=0.20,
                    name='Magnet shell',
                    showlegend=True,
                    flatshading=True,
                    hoverinfo='name',
                ))

            # ── Trace 3: MAGNET-only Volume (hidden by default) ────────────
            mag_cube = np.where(magnet_mask, full_cube, sentinel)
            mag_cube = np.where(np.isnan(mag_cube), sentinel, mag_cube)
            traces.append(go.Volume(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=mag_cube.flatten(),
                isomin=val_min, isomax=val_max,
                opacity=0.25,
                surface_count=25,
                colorscale='Jet',
                caps=dict(x_show=False, y_show=False, z_show=False),
                name='Field — MAGNET only',
                showscale=False,
                visible='legendonly',
            ))

            # ── Trace 4: EXTENDED-only Volume (hidden by default, faded) ───
            ext_cube = np.where(extended_mask, full_cube, sentinel)
            ext_cube = np.where(np.isnan(ext_cube), sentinel, ext_cube)
            traces.append(go.Volume(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=ext_cube.flatten(),
                isomin=val_min, isomax=val_max,
                opacity=0.07,
                surface_count=20,
                colorscale='Jet',
                caps=dict(x_show=False, y_show=False, z_show=False),
                name='Field — EXTENDED only',
                showscale=False,
                visible='legendonly',
            ))

            # ── Trace 5: MAGNET points (scatter, hidden by default) ────────
            mag_df = df[df['region'] == 'MAGNET']
            traces.append(go.Scatter3d(
                x=mag_df['x'], y=mag_df['y'], z=mag_df['z'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=mag_df['value'],
                    colorscale='Jet',
                    cmin=val_min, cmax=val_max,
                    opacity=0.85,
                    showscale=False,
                    line=dict(width=0),
                ),
                name='MAGNET points',
                visible='legendonly',
                hovertemplate='x=%{x}<br>y=%{y}<br>z=%{z}<br>B=%{marker.color:.4f} T<extra>MAGNET</extra>',
            ))

            # ── Trace 6: EXTENDED points (scatter, hidden by default) ──────
            ext_df = df[df['region'] == 'EXTENDED']
            traces.append(go.Scatter3d(
                x=ext_df['x'], y=ext_df['y'], z=ext_df['z'],
                mode='markers',
                marker=dict(
                    size=2,
                    color=ext_df['value'],
                    colorscale='Jet',
                    cmin=val_min, cmax=val_max,
                    opacity=0.30,
                    showscale=False,
                    line=dict(width=0),
                ),
                name='EXTENDED points',
                visible='legendonly',
                hovertemplate='x=%{x}<br>y=%{y}<br>z=%{z}<br>B=%{marker.color:.4f} T<extra>EXTENDED</extra>',
            ))

        fig = go.Figure(data=traces)

        fig.update_layout(
            title='Magnetic Field 3D' + (' — with MAGNET / EXTENDED overlays' if has_region else ''),
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='cube',
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255,255,255,0.75)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
            ),
            showlegend=has_region,
        )

        html_path = os.path.join(UPLOAD_FOLDER, "plot.html")
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True})

        return {
            "url": "/static/plots/plot.html",
            "has_region": has_region,
            "point_count": int(len(df)),
            "region_counts": (
                df['region'].value_counts().to_dict() if has_region else None
            ),
        }

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/upload-2d")
async def upload_2d_api(file: UploadFile = File(...), x: str = Form(""), y: str = Form(""), z: str = Form("")):
    try:
        contents = await file.read()
        df = read_dataframe(contents, file.filename or "")

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

        # If we have region info, split into MAGNET vs EXTENDED for clarity.
        if 'region' in df_sorted.columns and df_sorted['region'].notna().any():
            mag = df_sorted[df_sorted['region'] == 'MAGNET']
            ext = df_sorted[df_sorted['region'] == 'EXTENDED']
            other = df_sorted[~df_sorted['region'].isin(['MAGNET', 'EXTENDED'])]

            if not ext.empty:
                fig.add_trace(go.Scatter(
                    x=ext[remaining_axis], y=ext['value'],
                    mode='lines+markers', name='EXTENDED',
                    line=dict(color='#d4a017'),
                    marker=dict(color='#d4a017'),
                ))
            if not mag.empty:
                fig.add_trace(go.Scatter(
                    x=mag[remaining_axis], y=mag['value'],
                    mode='lines+markers', name='MAGNET',
                    line=dict(color='#2ca02c'),
                    marker=dict(color='#2ca02c'),
                ))
            if not other.empty:
                fig.add_trace(go.Scatter(
                    x=other[remaining_axis], y=other['value'],
                    mode='lines+markers', name='other',
                ))
        else:
            fig.add_trace(go.Scatter(x=df_sorted[remaining_axis], y=df_sorted['value'], mode='lines+markers'))

        fig.update_layout(
            title=f'Magnetic Field along {remaining_axis.upper()} ({fixed_labels})',
            xaxis_title=f'{remaining_axis.upper()} (mm)',
            yaxis_title='Magnetic Field Value (T)',
            margin=dict(l=40, r=40, t=60, b=40),
            autosize=True,
        )

        html_path = os.path.join(UPLOAD_FOLDER, "plot2d.html")
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True})

        table_cols = ["x", "y", "z", "value"]
        if 'region' in df_sorted.columns:
            table_cols.append('region')
        table_data = df_sorted[table_cols].to_dict(orient="records")
        return {"url": "/static/plots/plot2d.html", "table": table_data}

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/section-1d")
async def section_1d(file: UploadFile = File(...), axis: str = Form(...), value: str = Form(...)):
    try:
        contents = await file.read()
        df = read_dataframe(contents, file.filename or "")

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
            showscale=True,
        ))

        # If region info exists, overlay a light outline of the MAGNET points
        # so users can see where the magnet sits on this slice.
        if 'region' in df_filtered.columns and df_filtered['region'].notna().any():
            mag = df_filtered[df_filtered['region'] == 'MAGNET']
            if not mag.empty:
                fig.add_trace(go.Scatter(
                    x=mag[axes[0]], y=mag[axes[1]],
                    mode='markers',
                    marker=dict(
                        symbol='square-open',
                        size=6,
                        color='lime',
                        line=dict(width=1.2),
                    ),
                    name='MAGNET cells',
                    hoverinfo='skip',
                ))

        fig.update_layout(
            title=f"Section at {axis.upper()} = {value}",
            xaxis_title=f"{axes[0].upper()} (mm)",
            yaxis_title=f"{axes[1].upper()} (mm)",
            margin=dict(l=40, r=40, t=60, b=40),
            autosize=True,
        )

        filename = f"section_{axis}{int(value)}.html"
        html_path = os.path.join(UPLOAD_FOLDER, filename)
        fig.write_html(html_path, include_plotlyjs='cdn', config={"responsive": True})

        table_cols = ["x", "y", "z", "value"]
        if 'region' in df_filtered.columns:
            table_cols.append('region')
        table_data = df_filtered[table_cols].to_dict(orient="records")
        return {"url": f"/static/plots/{filename}", "table": table_data}

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


app.mount("/static", StaticFiles(directory="static"), name="static")
