from flask import Flask, request, jsonify, send_from_directory
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "plots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/upload', methods=['POST'])
def upload_api():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "Missing file"}), 400

    df = pd.read_excel(file)
    required_cols = {'X', 'Y', 'Z', 'Value'}
    if not required_cols.issubset(df.columns):
        return jsonify({"error": "Missing columns"}), 400

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

    # Salvează plot-ul
    filename = "plot.html"
    path = os.path.join(UPLOAD_FOLDER, filename)
    fig.write_html(path, full_html=True, include_plotlyjs='cdn', config={"responsive": True})

    # Construim URL-ul corect folosind host-ul automat
    full_url = request.host_url.rstrip('/') + '/static/plots/' + filename
    return jsonify({"url": full_url})

# IMPORTANT: Railway o să pornească app-ul cu gunicorn/uvicorn -> deci NU mai punem app.run()
