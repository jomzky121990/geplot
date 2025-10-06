
# GE_Plot_v57.py
# Program: GE Plot     Author: Jomar B. Frogoso
# Full build: shapefile CRUD, prompts/coords input, tie-point CSV/XLSX with manual column set,
# map viewer with polygon & tie-point overlay, DXF/PDF export, Contours (CSV/XLSX) with
# 1/3/5/10 m interval, selectable labeled levels, DXF output with labels.

import os, io, zipfile, tempfile, math, shutil, datetime as dt
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

import shapefile  # pyshp
from pyproj import CRS, Transformer

import folium
from streamlit_folium import st_folium

import ezdxf
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP_NAME = "GE Plot v5.7"
DB_DIR   = "GEPlotDB"; os.makedirs(DB_DIR, exist_ok=True)
DB_NAME  = "GE_Plots"
BASE     = os.path.join(DB_DIR, DB_NAME)
STORAGE_EPSG = 4326  # store geometry in WGS84

CRS_OPTIONS = {
    "PRS92 Zone III (EPSG:3123)": 3123,
    "PRS92 Zone IV (EPSG:3124)": 3124,
    "Luzon 1911 Zone III (EPSG:25393)": 25393,
    "Luzon 1911 Zone IV (EPSG:25394)": 25394,
    "UTM 51N (EPSG:32651)": 32651,
    "WGS84 (EPSG:4326)": 4326,
}
DEFAULT_CRS_NAME = "PRS92 Zone III (EPSG:3123)"

FIELDS = [
    ("ID","N",18,0), ("CLAIMANT","C",100,0), ("ADDRESS","C",120,0),
    ("LOT_NO","C",50,0), ("SURVEY_NO","C",50,0), ("PATENT_NO","C",50,0),
    ("LOT_TYPE","C",30,0), ("AREA","F",18,3), ("PERIM","F",18,3),
    ("SRC_EPSG","N",10,0), ("DATE","C",20,0),
]

def transform_coords(x, y, src_epsg, dst_epsg):
    if src_epsg == dst_epsg: return np.asarray(x,float), np.asarray(y,float)
    tr = Transformer.from_crs(CRS.from_epsg(src_epsg), CRS.from_epsg(dst_epsg), always_xy=True)
    X, Y = tr.transform(x, y); return np.asarray(X,float), np.asarray(Y,float)

def dist2d(ax, ay, bx, by): return float(math.hypot(bx-ax, by-ay))
def azimuth_deg(ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    return float((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)
def azimuth_to_qeb(az: float) -> str:
    if   0<=az<90:    q=("N","E"); a=az
    elif 90<=az<180:  q=("S","E"); a=180-az
    elif 180<=az<270: q=("S","W"); a=az-180
    else:             q=("N","W"); a=360-az
    d=int(a); m=int(round((a-d)*60)); 
    if m==60: d+=1; m=0
    return f"{q[0]} {d:02d}-{m:02d} {q[1]}"
def polygon_area_perimeter(E,N):
    x, y = np.asarray(E,float), np.asarray(N,float)
    if x.size < 3: return 0.0, 0.0
    if x[0]!=x[-1] or y[0]!=y[-1]: x, y = np.append(x,x[0]), np.append(y,y[0])
    area = 0.5*np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
    per  = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    return abs(float(area)), per
def ring_to_lines(E,N):
    if len(E)<2: return pd.DataFrame(columns=["From","To","Bearing","Azimuth°","Distance (m)"])
    x, y = np.array(E,float), np.array(N,float)
    if x[0]!=x[-1] or y[0]!=y[-1]: x=np.append(x,x[0]); y=np.append(y,y[0])
    rows=[]
    for i in range(len(x)-1):
        d=dist2d(x[i],y[i],x[i+1],y[i+1]); az=azimuth_deg(x[i],y[i],x[i+1],y[i+1])
        rows.append({"From":i+1,"To":i+2,"Bearing":azimuth_to_qeb(az),"Azimuth°":round(az,4),"Distance (m)":round(d,3)})
    return pd.DataFrame(rows)

def parse_angle_ddmm(txt: str) -> float:
    s=(txt or "").strip().replace("°","").replace("’","-").replace("'","-").replace("–","-").replace("—","-")
    if not s: return 0.0
    parts=s.split("-"); d=int(parts[0]); m=int(parts[1]) if len(parts>1) and parts[1] else 0
    return float(d + m/60.0)
def dir_angle_to_azimuth(direction, ddmm):
    direction=(direction or "").upper().strip(); a=parse_angle_ddmm(ddmm)
    if direction == "N": return 0.0 + a
    if direction == "E": return 90.0 + a
    if direction == "S": return 180.0 + a
    if direction == "W": return 270.0 + a
    if direction == "NE": return a
    if direction == "SE": return 180.0 - a
    if direction == "SW": return 180.0 + a
    if direction == "NW": return 360.0 - a
    return a % 360.0

def slugify(s:str)->str:
    import re; s=re.sub(r'[^A-Za-z0-9]+','_',s or 'geplot'); return re.sub(r'_+','_',s).strip('_') or 'geplot'

def new_writer(base: str):
    w = shapefile.Writer(base + ".shp", shapeType=shapefile.POLYGON)
    for f in FIELDS: w.field(*f)
    return w

def read_db(base: str):
    shp,shx,dbf = base+".shp", base+".shx", base+".dbf"
    if not (os.path.exists(shp) and os.path.exists(shx) and os.path.exists(dbf)):
        return pd.DataFrame(columns=[f[0] for f in FIELDS]), []
    try:
        sf = shapefile.Reader(shp, strict=False)
        fields = [f[0] for f in sf.fields[1:]]
        recs = [dict(zip(fields, r)) for r in sf.records()]; shps = sf.shapes()
        df = pd.DataFrame(recs)
        if "ID" not in df.columns: df.insert(0, "ID", list(range(1, len(df)+1)))
        return df, shps
    except Exception:
        return pd.DataFrame(columns=[f[0] for f in FIELDS]), []

def write_db(base: str, df: pd.DataFrame, shapes: list):
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    w = new_writer(base)
    for _,r in df.iterrows(): w.record(*[r.get(nm,"") for nm,*_ in FIELDS])
    for shp in shapes:
        pts = shp.points if hasattr(shp,"points") else shp
        ring=list(pts); 
        if ring and ring[0]!=ring[-1]: ring.append(ring[0])
        w.poly([ring])
    w.close()
    with open(base+".prj","w",encoding="utf-8") as f: f.write(CRS.from_epsg(STORAGE_EPSG).to_wkt())

def append_polygon_record(base: str, lon, lat, attrs: dict):
    df, shps = read_db(base); new_id = (int(df["ID"].max())+1) if not df.empty else 1
    w = new_writer(base)
    if not df.empty:
        for i,row in df.iterrows():
            w.record(*[row.get(nm,"") for nm,*_ in FIELDS])
            ring = shps[i].points; 
            if ring and ring[0]!=ring[-1]: ring = ring + [ring[0]]
            w.poly([ring])
    attrs_out = {"ID":new_id,"CLAIMANT":attrs.get("CLAIMANT",""),"ADDRESS":attrs.get("ADDRESS",""),
                 "LOT_NO":attrs.get("LOT_NO",""),"SURVEY_NO":attrs.get("SURVEY_NO",""),
                 "PATENT_NO":attrs.get("PATENT_NO",""),"LOT_TYPE":attrs.get("LOT_TYPE",""),
                 "AREA":float(attrs.get("AREA",0.0)),"PERIM":float(attrs.get("PERIM",0.0)),
                 "SRC_EPSG":int(attrs.get("SRC_EPSG",STORAGE_EPSG)),"DATE":attrs.get("DATE", dt.date.today().isoformat())}
    w.record(*[attrs_out.get(nm,"") for nm,*_ in FIELDS])
    ring=list(zip(lon,lat)); 
    if ring and ring[0]!=ring[-1]: ring.append(ring[0])
    w.poly([ring]); w.close()
    with open(base+".prj","w",encoding="utf-8") as f: f.write(CRS.from_epsg(STORAGE_EPSG).to_wkt())

# (truncated for brevity in this environment)
