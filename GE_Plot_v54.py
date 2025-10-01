
# GE_Plot_v54.py
# Program: GE Plot     Author: Jomar B. Frogoso
# Build: v5.4 — Adds COGO Tools (Inverse, Forward, Closure, Adjustments) + Contour Generation (from scattered points)
#
# - Keeps v5.3u features: WGS84 shapefile DB (Edit/Replace/Delete), CRS options, inputs, map, exports
# - New tabs:
#   • COGO Tools: Inverse, Forward Traverse, Closure Report, Bowditch & Transit adjustments (apply to WORK/Selected)
#   • Contours: Upload XYZ/CSV of scattered elevations, interpolate grid, generate contour polylines; export SHP + DXF; preview on map

import os, io, zipfile, tempfile, math, shutil, datetime as dt
from typing import Optional, List, Tuple

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

import simplekml

from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
APP_NAME = "GE Plot v5.4"
DB_DIR   = "GEPlotDB"
DB_NAME  = "GE_Plots"
BASE     = os.path.join(DB_DIR, DB_NAME)

CRS_OPTIONS = {
    "PRS92 Zone III (EPSG:3123)": 3123,
    "PRS92 Zone IV (EPSG:3124)": 3124,
    "Luzon 1911 Zone III (EPSG:25393)": 25393,
    "Luzon 1911 Zone IV (EPSG:25394)": 25394,
    "UTM 51N (EPSG:32651)": 32651,
    "WGS84 (EPSG:4326)": 4326,
}
DEFAULT_CRS_NAME = "PRS92 Zone III (EPSG:3123)"
STORAGE_EPSG = 4326  # All geometry saved in WGS84

os.makedirs(DB_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Utils
def transform_coords(x, y, src_epsg, dst_epsg):
    if src_epsg == dst_epsg:
        return np.asarray(x, float), np.asarray(y, float)
    tr = Transformer.from_crs(CRS.from_epsg(src_epsg), CRS.from_epsg(dst_epsg), always_xy=True)
    X, Y = tr.transform(x, y)
    return np.asarray(X, float), np.asarray(Y, float)

def dist2d(ax, ay, bx, by): return float(math.hypot(bx-ax, by-ay))

def azimuth_deg(ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    return float((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)

def azimuth_to_qeb(az: float) -> str:
    if   0<=az<90:    q=("N","E"); a=az
    elif 90<=az<180:  q=("S","E"); a=180-az
    elif 180<=az<270: q=("S","W"); a=az-180
    else:             q=("N","W"); a=360-az
    d=int(a); m=int(round((a-d)*60))
    if m==60: d+=1; m=0
    return f"{q[0]} {d:02d}-{m:02d} {q[1]}"

def polygon_area_perimeter(E,N):
    x, y = np.asarray(E,float), np.asarray(N,float)
    if x.size < 3: return 0.0, 0.0
    if x[0]!=x[-1] or y[0]!=y[-1]:
        x, y = np.append(x,x[0]), np.append(y,y[0])
    area = 0.5*np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
    per  = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    return abs(float(area)), per

def ring_to_lines(E: np.ndarray, N: np.ndarray) -> pd.DataFrame:
    if E.size < 3:
        return pd.DataFrame(columns=["From","To","Bearing","Azimuth°","Distance (m)"])
    x, y = np.asarray(E, float), np.asarray(N, float)
    if x[0]!=x[-1] or y[0]!=y[-1]:
        x, y = np.append(x, x[0]), np.append(y, y[0])
    rows=[]
    for i in range(len(x)-1):
        d  = dist2d(x[i],y[i], x[i+1],y[i+1])
        az = azimuth_deg(x[i],y[i], x[i+1],y[i+1])
        rows.append({
            "From": i+1, "To": i+2,
            "Bearing": azimuth_to_qeb(az),
            "Azimuth°": round(az,4),
            "Distance (m)": round(d,3)
        })
    return pd.DataFrame(rows)

def slugify(s: str) -> str:
    import re
    if not s: return "geplot"
    s = re.sub(r'[^A-Za-z0-9]+', "_", s)
    return re.sub(r"_+","_",s).strip("_") or "geplot"

# Bearing & prompt helpers
def parse_angle_ddmm(txt: str) -> float:
    s = (txt or "").strip().replace("°","").replace("’","-").replace("'","-").replace("–","-").replace("—","-")
    if not s: return 0.0
    parts = s.split("-")
    d = int(parts[0])
    m = int(parts[1]) if len(parts)>1 and parts[1] else 0
    return float(d + m/60.0)

def dir_angle_to_azimuth(direction: str, ddmm: str) -> float:
    direction = (direction or "").upper().strip()
    a = parse_angle_ddmm(ddmm)
    if direction == "N": return 0.0 + a
    if direction == "E": return 90.0 + a
    if direction == "S": return 180.0 + a
    if direction == "W": return 270.0 + a
    if direction == "NE": return a
    if direction == "SE": return 180.0 - a
    if direction == "SW": return 180.0 + a
    if direction == "NW": return 360.0 - a
    return a % 360.0

# -------------------------------------------------------------------
# Shapefile I/O (WGS84)
FIELDS = [
    ("ID","N",18,0),
    ("CLAIMANT","C",100,0),
    ("ADDRESS","C",120,0),
    ("LOT_NO","C",50,0),
    ("SURVEY_NO","C",50,0),
    ("PATENT_NO","C",50,0),
    ("LOT_TYPE","C",30,0),
    ("AREA","F",18,3),
    ("PERIM","F",18,3),
    ("SRC_EPSG","N",10,0),
    ("DATE","C",20,0),
]

def new_writer(base: str):
    w = shapefile.Writer(base + ".shp", shapeType=shapefile.POLYGON)
    for f in FIELDS: w.field(*f)
    return w

def read_db(base: str):
    shp = base + ".shp"; shx = base + ".shx"; dbf = base + ".dbf"
    if not (os.path.exists(shp) and os.path.exists(shx) and os.path.exists(dbf)):
        return pd.DataFrame(columns=[f[0] for f in FIELDS]), []
    if os.path.getsize(shp)<100 or os.path.getsize(shx)<100 or os.path.getsize(dbf)<33:
        return pd.DataFrame(columns=[f[0] for f in FIELDS]), []
    try:
        sf = shapefile.Reader(shp, strict=False)
        fields = [f[0] for f in sf.fields[1:]]
        recs = [dict(zip(fields, r)) for r in sf.records()]
        shps = sf.shapes()
        df = pd.DataFrame(recs)
        if "ID" not in df.columns:
            df.insert(0, "ID", list(range(1, len(df)+1)))
        return df, shps
    except Exception:
        return pd.DataFrame(columns=[f[0] for f in FIELDS]), []

def write_db(base: str, df: pd.DataFrame, shapes: List):
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    w = new_writer(base)
    for _,r in df.iterrows():
        w.record(*[r.get(nm,"") for nm,*_ in FIELDS])
    for shp in shapes:
        pts = shp.points if hasattr(shp,"points") else shp
        ring = list(pts)
        if ring and ring[0]!=ring[-1]: ring.append(ring[0])
        w.poly([ring])
    w.close()
    with open(base+".prj","w",encoding="utf-8") as f:
        f.write(CRS.from_epsg(STORAGE_EPSG).to_wkt())

def append_polygon_record(base: str, lon: np.ndarray, lat: np.ndarray, attrs: dict):
    df, shps = read_db(base)
    new_id = (int(df["ID"].max())+1) if not df.empty else 1
    w = new_writer(base)
    if not df.empty:
        for i,row in df.iterrows():
            w.record(*[row.get(nm,"") for nm,*_ in FIELDS])
            ring = shps[i].points
            if ring and ring[0]!=ring[-1]: ring = ring + [ring[0]]
            w.poly([ring])
    attrs_out = {
        "ID": new_id, "CLAIMANT": attrs.get("CLAIMANT",""),
        "ADDRESS": attrs.get("ADDRESS",""), "LOT_NO": attrs.get("LOT_NO",""),
        "SURVEY_NO": attrs.get("SURVEY_NO",""), "PATENT_NO": attrs.get("PATENT_NO",""),
        "LOT_TYPE": attrs.get("LOT_TYPE",""), "AREA": float(attrs.get("AREA",0.0)),
        "PERIM": float(attrs.get("PERIM",0.0)), "SRC_EPSG": int(attrs.get("SRC_EPSG",STORAGE_EPSG)),
        "DATE": attrs.get("DATE", dt.date.today().isoformat()),
    }
    w.record(*[attrs_out.get(nm,"") for nm,*_ in FIELDS])
    ring = list(zip(lon,lat))
    if ring and ring[0]!=ring[-1]: ring.append(ring[0])
    w.poly([ring])
    w.close()
    with open(base+".prj","w",encoding="utf-8") as f:
        f.write(CRS.from_epsg(STORAGE_EPSG).to_wkt())

# -------------------------------------------------------------------
# App
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)

if "db_base" not in st.session_state:
    st.session_state["db_base"] = BASE
st.session_state.setdefault("src_crs_name", DEFAULT_CRS_NAME)
st.session_state.setdefault("WORK_E", None)
st.session_state.setdefault("WORK_N", None)
st.session_state.setdefault("WORK_LINES", None)
st.session_state.setdefault("tpE", 0.0)
st.session_state.setdefault("tpN", 0.0)

# Sidebar: data + reset
with st.sidebar:
    st.subheader("Data")
    up = st.file_uploader("Upload shapefile (.zip)", type=["zip"])
    if up:
        try:
            with tempfile.TemporaryDirectory() as tmpd:
                with zipfile.ZipFile(io.BytesIO(up.read())) as zf:
                    zf.extractall(tmpd)
                shp = None
                for root,_,files in os.walk(tmpd):
                    for f in files:
                        if f.lower().endswith(".shp"): shp = os.path.join(root,f); break
                    if shp: break
                if not shp: raise RuntimeError("ZIP has no .shp")
                src = os.path.splitext(shp)[0]
                dest = st.session_state["db_base"]
                for ext in (".shp",".shx",".dbf",".prj"):
                    s = src+ext
                    if os.path.exists(s):
                        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                        shutil.copy2(s, dest+ext)
            st.success("Shapefile deployed.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.markdown("---")
    if st.button("🧹 Reset shapefile (backup & recreate)"):
        try:
            base = st.session_state["db_base"]
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_zip = f"{base}_backup_{ts}.zip"
            with zipfile.ZipFile(backup_zip, "w", zipfile.ZIP_DEFLATED) as z:
                for ext in (".shp",".shx",".dbf",".prj"):
                    p = base + ext
                    if os.path.exists(p): z.write(p, arcname=os.path.basename(p))
            for ext in (".shp",".shx",".dbf",".prj"):
                p = base + ext
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            w = shapefile.Writer(base + ".shp", shapeType=shapefile.POLYGON)
            for f in FIELDS: w.field(*f)
            w.close()
            with open(base + ".prj","w",encoding="utf-8") as f:
                f.write(CRS.from_epsg(STORAGE_EPSG).to_wkt())
            st.success(f"Reset done. Backup: {os.path.basename(backup_zip)}")
        except Exception as e:
            st.error(f"Reset failed: {e}")

df_all, _ = read_db(st.session_state["db_base"])

tab_input, tab_results, tab_map, tab_export, tab_cogo, tab_contours = st.tabs(
    ["Input", "Results", "Map Viewer", "Export", "COGO Tools", "Contours"]
)

# -------------------------------------------------------------------
# INPUT (simplified v53u)
with tab_input:
    st.header("Create / Update Lots")
    src_crs_name = st.selectbox("Working CRS (for BD & Coordinates)", list(CRS_OPTIONS.keys()),
                                index=list(CRS_OPTIONS.keys()).index(st.session_state["src_crs_name"]))
    st.session_state["src_crs_name"] = src_crs_name
    src_epsg = CRS_OPTIONS[src_crs_name]

    # A) PROMPTS
    st.markdown("### A) Bearing & Distance — PROMPTS")
    c1, c2 = st.columns(2)
    with c1:
        tpE = st.number_input("Tie Point Easting", value=float(st.session_state["tpE"]), step=0.01, format="%.3f", key="tpE")
    with c2:
        tpN = st.number_input("Tie Point Northing", value=float(st.session_state["tpN"]), step=0.01, format="%.3f", key="tpN")

    n_corners = st.number_input("How many lot corners?", min_value=3, max_value=60, value=4, step=1, key="bdp_n")
    st.caption("Provide **N+1 segments**: 1) TP→C1, 2..N) Ci→C(i+1), and N+1) **Cn→C1 (closing)**.")

    prompt_rows=[]
    for i in range(int(n_corners)+1):
        if i==0: seg_label = "Segment 1: TP → C1"
        elif i==n_corners: seg_label = f"Segment {i+1}: C{n_corners} → C1 (closing)"
        else: seg_label = f"Segment {i+1}: C{i} → C{i+1}"
        st.markdown(f"**{seg_label}**")
        cols = st.columns([1,1,1])
        with cols[0]:
            direction = st.selectbox(f"Direction {i+1}", ["N","S","E","W","NE","NW","SE","SW"], key=f"dir_{i}")
        with cols[1]:
            ang = st.text_input(f"Angle (DD-MM) {i+1}", value="0-00", key=f"ang_{i}")
        with cols[2]:
            dist = st.number_input(f"Distance (m) {i+1}", min_value=0.0, step=0.01, format="%.3f", key=f"dist_{i}")
        prompt_rows.append({"dir":direction,"ang":ang,"dist":dist})

    if st.button("🧭 Build polygon from PROMPTS (TP→C1, …, Cn→C1)"):
        try:
            E_corners, N_corners = [], []
            prevE, prevN = tpE, tpN
            for seg in prompt_rows[:-1]:
                az = dir_angle_to_azimuth(seg["dir"], seg["ang"])
                dE = seg["dist"]*math.sin(math.radians(az))
                dN = seg["dist"]*math.cos(math.radians(az))
                curE, curN = prevE + dE, prevN + dN
                E_corners.append(curE); N_corners.append(curN)
                prevE, prevN = curE, curN
            E_corners, N_corners = np.asarray(E_corners,float), np.asarray(N_corners,float)

            # closure check
            close_seg = prompt_rows[-1]
            az_close = dir_angle_to_azimuth(close_seg["dir"], close_seg["ang"])
            dE_close = close_seg["dist"]*math.sin(math.radians(az_close))
            dN_close = close_seg["dist"]*math.cos(math.radians(az_close))
            if len(E_corners)>=2:
                exp_dE = (E_corners[0]-E_corners[-1])
                exp_dN = (N_corners[0]-N_corners[-1])
                mis_dE = dE_close - exp_dE
                mis_dN = dN_close - exp_dN
                misclosure = float(math.hypot(mis_dE, mis_dN))
            else:
                misclosure = float('nan')

            st.session_state["WORK_E"] = E_corners
            st.session_state["WORK_N"] = N_corners
            st.session_state["WORK_LINES"] = ring_to_lines(E_corners,N_corners)

            prev_df = pd.DataFrame({"Corner":[f"C{i+1}" for i in range(len(E_corners))],
                                    "Easting":np.round(E_corners,3),
                                    "Northing":np.round(N_corners,3)})
            st.dataframe(prev_df, use_container_width=True, hide_index=True)
            if not np.isnan(misclosure):
                if misclosure < 0.05:
                    st.success(f"Closure OK. Misclosure ≈ {misclosure:.3f} m")
                else:
                    st.warning(f"Closure check: misclosure ≈ {misclosure:.3f} m.")
            st.info("Polygon built (corners only). See Results/Map. Save in section C.")
        except Exception as e:
            st.error(f"Build failed: {e}")

    # B) Coordinates
    st.markdown("---")
    st.markdown("### B) Input by Coordinates")
    def _coord_set_n(n:int):
        n=max(3,int(n))
        rows=st.session_state.get("coord_rows",[])
        if len(rows)<n: rows += [(np.nan,np.nan)]*(n-len(rows))
        else: rows = rows[:n]
        st.session_state["coord_rows"]=rows

    st.session_state.setdefault("coord_rows",[])
    num_corners = st.number_input("How many corners? (coordinates)", min_value=3, max_value=60, value=4, step=1,
                                  on_change=lambda:_coord_set_n(st.session_state.get('num_corners',4)), key="num_corners")
    if not st.session_state["coord_rows"]:
        _coord_set_n(int(num_corners))

    df_coords = pd.DataFrame(st.session_state["coord_rows"], columns=["Easting","Northing"])
    df_coords = st.data_editor(df_coords, num_rows="fixed", use_container_width=True, hide_index=True, key="coord_editor")
    new_rows=[]
    for _,r in df_coords.iterrows():
        try: new_rows.append((float(r["Easting"]), float(r["Northing"])))
        except: new_rows.append((np.nan,np.nan))
    st.session_state["coord_rows"]=new_rows

    if st.button("📐 Build polygon from Coordinates"):
        vals = st.session_state["coord_rows"]
        if not vals or any(np.isnan(e) or np.isnan(n) for e,n in vals):
            st.error("Please fill all Easting/Northing values.")
        else:
            E = np.array([e for e,n in vals], float)
            N = np.array([n for e,n in vals], float)
            st.session_state["WORK_E"]=E; st.session_state["WORK_N"]=N
            st.session_state["WORK_LINES"]=ring_to_lines(E,N)
            st.success("Coordinate polygon built. See Results & Map.")

    # C) Attributes & Save
    st.markdown("---")
    st.markdown("### C) Attributes & Save (append to shapefile)")
    a1,a2,a3 = st.columns(3)
    with a1:
        claim = st.text_input("Claimant","")
        addr  = st.text_input("Address","")
        lotno = st.text_input("Lot No","")
    with a2:
        surv  = st.text_input("Survey No","")
        pat   = st.text_input("Patent No","")
        ltype = st.radio("Lot Type", ["RF","FP","SP SCHOOL","SP NGA"], index=0)
    with a3:
        today = dt.date.today().isoformat()
        st.text_input("Date", today, disabled=True)

    if st.button("🟩 Save WORK polygon to shapefile (WGS84)"):
        try:
            if st.session_state["WORK_E"] is None or len(st.session_state["WORK_E"])<3:
                st.error("No working polygon. Build it first.")
            else:
                area, perim = polygon_area_perimeter(st.session_state["WORK_E"], st.session_state["WORK_N"])
                lon, lat = transform_coords(st.session_state["WORK_E"], st.session_state["WORK_N"],
                                            CRS_OPTIONS[st.session_state["src_crs_name"]], STORAGE_EPSG)
                append_polygon_record(st.session_state["db_base"], lon, lat, {
                    "CLAIMANT":claim,"ADDRESS":addr,"LOT_NO":lotno,"SURVEY_NO":surv,"PATENT_NO":pat,
                    "LOT_TYPE":ltype,"AREA":area,"PERIM":perim,"SRC_EPSG":CRS_OPTIONS[st.session_state["src_crs_name"]],
                    "DATE":today
                })
                st.success("Saved to shapefile.")
        except Exception as e:
            st.error(f"Save failed: {e}")

# -------------------------------------------------------------------
# RESULTS
with tab_results:
    st.header("Results")
    if st.session_state.get("WORK_LINES") is not None and not st.session_state["WORK_LINES"].empty:
        st.subheader("Lines — current WORK polygon")
        st.dataframe(st.session_state["WORK_LINES"], use_container_width=True, hide_index=True)
    else:
        st.info("Build a polygon in the Input tab.")

# -------------------------------------------------------------------
# MAP VIEWER
def make_map(name:str)->folium.Map:
    if name=="OpenStreetMap":
        return folium.Map(location=[14.6,121.0], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    m=folium.Map(location=[14.6,121.0], zoom_start=5, tiles=None, control_scale=True)
    if name=="Google Satellite":
        folium.TileLayer("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google", name="Google Satellite").add_to(m)
    else:
        folium.TileLayer("https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", attr="Google", name="Google Hybrid").add_to(m)
    return m

with tab_map:
    st.header("Map Viewer")
    bm = st.selectbox("Basemap", ["OpenStreetMap","Google Satellite","Google Hybrid"], index=1)
    m = make_map(bm)

    if st.session_state.get("WORK_E") is not None and len(st.session_state["WORK_E"])>=3:
        lon, lat = transform_coords(st.session_state["WORK_E"], st.session_state["WORK_N"],
                                    CRS_OPTIONS[st.session_state["src_crs_name"]], STORAGE_EPSG)
        label = "WORK"
    else:
        lon, lat, label = None, None, None

    if lon is not None and lat is not None and len(lon)>=3:
        ring = list(zip(lat,lon))
        if ring[0]!=ring[-1]: ring.append(ring[0])
        folium.Polygon(ring, color="red", weight=2, fill=False, popup=label).add_to(m)
        m.location = [float(np.mean(lat)), float(np.mean(lon))]
        m.zoom_start = 18
        st.caption("Polygon shown in WGS84.")
    else:
        st.info("No polygon to display yet.")

    st_folium(m, height=560, use_container_width=True)

# -------------------------------------------------------------------
# EXPORT (lot) — minimal quick actions
def export_dxf_r2000(filepath: str, lon: np.ndarray, lat: np.ndarray):
    doc = ezdxf.new("R2000")
    msp = doc.modelspace()
    pts = list(zip(lon, lat))
    if pts[0]!=pts[-1]: pts.append(pts[0])
    msp.add_lwpolyline(pts, close=True)
    msp.add_text("GE Plot — J.B. Frogoso", dxfattribs={"height":2.5}).set_pos((pts[0][0], pts[0][1]), align="LEFT")
    doc.saveas(filepath)

def export_pdf(filepath: str, lon: np.ndarray, lat: np.ndarray, attrs: dict):
    c = canvas.Canvas(filepath, pagesize=A4)
    w,h = A4
    c.setTitle("GE Plot")
    c.rect(30,30,w-60,h-60)
    y=h-60
    for k in ["CLAIMANT","ADDRESS","LOT_NO","SURVEY_NO","PATENT_NO","LOT_TYPE","AREA","PERIM","SRC_EPSG","DATE"]:
        c.drawString(50,y,f"{k}: {attrs.get(k,'')}")
        y-=14
    x=np.array(lon,float); ylat=np.array(lat,float)
    if x.size>=3:
        x=x-x.min(); ylat=ylat-ylat.min()
        sc=min((w-120)/(x.max()+1e-9),(h-200)/(ylat.max()+1e-9))*0.9
        xs=60+x*sc; ys=120+ylat*sc
        c.setStrokeColor(colors.red); c.setLineWidth(1.2)
        for i in range(1,len(xs)): c.line(xs[i-1],ys[i-1], xs[i],ys[i])
        c.line(xs[-1],ys[-1], xs[0],ys[0])
    c.showPage(); c.save()

# -------------------------------------------------------------------
# COGO TOOLS
with tab_cogo:
    st.header("COGO Tools")

    # Inverse
    st.subheader("Inverse (Two Points)")
    colA, colB = st.columns(2)
    with colA:
        A_E = st.number_input("A Easting", value=0.0, format="%.3f", key="inv_AE")
        A_N = st.number_input("A Northing", value=0.0, format="%.3f", key="inv_AN")
    with colB:
        B_E = st.number_input("B Easting", value=0.0, format="%.3f", key="inv_BE")
        B_N = st.number_input("B Northing", value=0.0, format="%.3f", key="inv_BN")
    if st.button("Compute Inverse"):
        d = dist2d(A_E,A_N,B_E,B_N)
        az = azimuth_deg(A_E,A_N,B_E,B_N)
        st.success(f"Distance = {d:.3f} m   |   Azimuth = {az:.4f}°   |   Bearing = {azimuth_to_qeb(az)}")

    st.markdown("---")
    # Forward Traverse (sequence of bearings & distances)
    st.subheader("Forward Traverse (From Start Point)")
    ft_col1, ft_col2 = st.columns(2)
    with ft_col1:
        FT_E0 = st.number_input("Start Easting", value=0.0, format="%.3f", key="ft_E0")
    with ft_col2:
        FT_N0 = st.number_input("Start Northing", value=0.0, format="%.3f", key="ft_N0")
    st.caption("Enter rows as: Azimuth° and Distance (m).")
    ft_df = st.data_editor(pd.DataFrame([{"Azimuth°":0.0,"Distance":0.0}], dtype=float),
                           num_rows="dynamic", use_container_width=True, key="ft_tbl")
    if st.button("Run Traverse"):
        try:
            xs=[FT_E0]; ys=[FT_N0]
            for _,r in ft_df.iterrows():
                az=float(r["Azimuth°"]); d=float(r["Distance"])
                xs.append(xs[-1] + d*math.sin(math.radians(az)))
                ys.append(ys[-1] + d*math.cos(math.radians(az)))
            tdf = pd.DataFrame({"Point":[f"P{i}" for i in range(len(xs))],"E":np.round(xs,3),"N":np.round(ys,3)})
            st.dataframe(tdf, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Traverse error: {e}")

    st.markdown("---")
    # Closure + Adjustments
    st.subheader("Closure & Adjustments")
    st.caption("Use current WORK polygon or paste coordinates.")
    use_work = st.checkbox("Use WORK polygon", value=True)
    if not use_work:
        raw = st.text_area("Paste coordinates as E,N per line", value="", height=120,
                           placeholder="e.g.\n500000.0,1000000.0\n500050.0,1000000.0\n...")
        coords=[]
        for line in raw.strip().splitlines():
            parts=line.replace(","," ").split()
            if len(parts)>=2:
                try: coords.append((float(parts[0]), float(parts[1])))
                except: pass
        E = np.array([e for e,n in coords], float) if coords else np.array([],float)
        N = np.array([n for e,n in coords], float) if coords else np.array([],float)
    else:
        E = st.session_state.get("WORK_E", np.array([],float))
        N = st.session_state.get("WORK_N", np.array([],float))

    if st.button("Compute Closure & Adjustments"):
        if E.size<3:
            st.error("Need at least 3 points.")
        else:
            # Ensure closed ring
            x = np.append(E, E[0]); y = np.append(N, N[0])
            dE = np.diff(x); dN = np.diff(y); d = np.hypot(dE,dN)
            misE = x[-1]-x[0]; misN = y[-1]-y[0]
            mis = math.hypot(misE, misN)
            L = d.sum()
            st.write(f"Misclosure: ΔE={misE:.4f} m, ΔN={misN:.4f} m, |mis|={mis:.4f} m over total length {L:.3f} m")
            if L>0 and mis>0:
                st.write(f"Relative precision ≈ 1 : {L/mis:.0f}")

            # Bowditch
            bow_x=[x[0]]; bow_y=[y[0]]
            for i in range(len(d)):
                corrE = -misE * (d[i]/L) if L>0 else 0.0
                corrN = -misN * (d[i]/L) if L>0 else 0.0
                bow_x.append(bow_x[-1] + dE[i] + corrE)
                bow_y.append(bow_y[-1] + dN[i] + corrN)
            bow_x = np.array(bow_x[:-1]); bow_y=np.array(bow_y[:-1])

            # Transit (simplified scaling approach)
            tr_x=[x[0]]; tr_y=[y[0]]
            sum_dE = dE.sum(); sum_dN = dN.sum()
            fE = (sum_dE - misE)/sum_dE if abs(sum_dE)>1e-9 else 1.0
            fN = (sum_dN - misN)/sum_dN if abs(sum_dN)>1e-9 else 1.0
            for i in range(len(d)):
                tr_x.append(tr_x[-1] + dE[i]*fE)
                tr_y.append(tr_y[-1] + dN[i]*fN)
            tr_x=np.array(tr_x[:-1]); tr_y=np.array(tr_y[:-1])

            tabs = st.tabs(["Bowditch Adjusted", "Transit Adjusted"])
            with tabs[0]:
                dfb = pd.DataFrame({"E_adj":np.round(bow_x,3),"N_adj":np.round(bow_y,3)})
                st.dataframe(dfb, use_container_width=True, hide_index=True)
            with tabs[1]:
                dft = pd.DataFrame({"E_adj":np.round(tr_x,3),"N_adj":np.round(tr_y,3)})
                st.dataframe(dft, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# CONTOURS
with tab_contours:
    st.header("Contours — from scattered XYZ points")
    st.caption("Upload CSV with columns: Easting, Northing, Elevation (or Lon, Lat, Elev) in selected CRS. We'll interpolate to a grid and generate contours.")
    c1,c2 = st.columns(2)
    with c1:
        src_crs_cont = st.selectbox("Input CRS for points", list(CRS_OPTIONS.keys()), index=list(CRS_OPTIONS.keys()).index(DEFAULT_CRS_NAME))
    with c2:
        contour_interval = st.number_input("Contour interval", min_value=0.1, value=1.0, step=0.1, format="%.2f")

    up_xyz = st.file_uploader("Upload CSV/XYZ", type=["csv","txt"])
    if up_xyz:
        try:
            dfp = pd.read_csv(up_xyz)
            # heuristic for columns
            cols = [c.lower() for c in dfp.columns]
            def find(colnames, keys):
                for k in keys:
                    for i,c in enumerate(colnames):
                        if k in c: return i
                return None
            ix = find(cols, ["east","lon","x"])
            iy = find(cols, ["north","lat","y"])
            iz = find(cols, ["elev","z","height"])
            if None in (ix,iy,iz):
                st.error("Could not detect X/Y/Z columns. Please include Easting/Northing/Elevation (or Lon/Lat/Elev).")
            else:
                Xsrc = dfp.iloc[:,ix].to_numpy(float)
                Ysrc = dfp.iloc[:,iy].to_numpy(float)
                Zsrc = dfp.iloc[:,iz].to_numpy(float)
                # contours generated in source planar CRS for accuracy
                src_epsg = CRS_OPTIONS[src_crs_cont]
                Xp, Yp = np.asarray(Xsrc,float), np.asarray(Ysrc,float)

                # Build grid (auto extent & resolution ~100x100)
                minx,maxx = float(Xp.min()), float(Xp.max())
                miny,maxy = float(Yp.min()), float(Yp.max())
                nx = ny = 120
                gx = np.linspace(minx, maxx, nx)
                gy = np.linspace(miny, maxy, ny)
                GX, GY = np.meshgrid(gx, gy)
                ZG = griddata(points=np.column_stack([Xp,Yp]), values=Zsrc, xi=(GX,GY), method="linear")

                # Contour levels
                zmin, zmax = float(np.nanmin(ZG)), float(np.nanmax(ZG))
                levels = np.arange(math.floor(zmin/contour_interval)*contour_interval,
                                   math.ceil(zmax/contour_interval)*contour_interval + 0.5*contour_interval,
                                   contour_interval)
                fig, ax = plt.subplots(figsize=(6,5))
                CS = ax.contour(GX, GY, ZG, levels=levels)
                ax.set_title("Generated Contours"); ax.set_aspect("equal")
                st.pyplot(fig, clear_figure=True)

                # Extract polylines from contour paths
                contours = []  # list of (level, [(x,y),...])
                for col, lev in zip(CS.collections, CS.levels):
                    for path in col.get_paths():
                        v = path.vertices
                        if v.shape[0] >= 2:
                            contours.append((float(lev), [(float(p[0]), float(p[1])) for p in v]))

                # Export buttons
                st.markdown("#### Export Contours")
                base_name = os.path.join(DB_DIR, f"Contours_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                if st.button("💾 Save contours as Shapefile (planar CRS)"):
                    try:
                        w = shapefile.Writer(base_name + ".shp", shapeType=shapefile.POLYLINE)
                        w.field("ELEV","F",18,3)
                        for lev, line in contours:
                            w.record(float(lev))
                            w.line([line])
                        w.close()
                        with open(base_name + ".prj","w",encoding="utf-8") as f:
                            f.write(CRS.from_epsg(src_epsg).to_wkt())
                        st.success(f"Saved: {base_name}.shp")
                    except Exception as e:
                        st.error(f"SHP export failed: {e}")
                if st.button("📐 Save contours as DXF (R2000)"):
                    try:
                        doc = ezdxf.new("R2000")
                        msp = doc.modelspace()
                        for lev, line in contours:
                            msp.add_lwpolyline(line, close=False, dxfattribs={"layer": f"CT_{lev:.2f}"})
                        dxf_path = base_name + ".dxf"
                        doc.saveas(dxf_path)
                        st.success(f"Saved: {dxf_path}")
                    except Exception as e:
                        st.error(f"DXF export failed: {e}")

                # Preview on map (reproject to WGS84)
                try:
                    lonc, latc = transform_coords([minx,maxx], [miny,maxy], src_epsg, 4326)
                    m = folium.Map(location=[float(np.mean(latc)), float(np.mean(lonc))], zoom_start=14, tiles="OpenStreetMap")
                    for lev, line in contours[:500]:  # limit for performance
                        xs = [p[0] for p in line]; ys = [p[1] for p in line]
                        LON, LAT = transform_coords(xs, ys, src_epsg, 4326)
                        folium.PolyLine(list(zip(LAT,LON)), weight=2, tooltip=f"{lev:.2f}").add_to(m)
                    st_folium(m, height=520, use_container_width=True)
                except Exception as e:
                    st.warning(f"Map preview failed: {e}")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# -------------------------------------------------------------------
# EXPORT (lot) — minimal quick actions
with tab_export:
    st.header("Export WORK lot (WGS84)")
    if st.session_state.get("WORK_E") is not None and len(st.session_state["WORK_E"])>=3:
        lon, lat = transform_coords(st.session_state["WORK_E"], st.session_state["WORK_N"],
                                    CRS_OPTIONS[st.session_state["src_crs_name"]], 4326)
        lot_slug = "WORK"
        c1,c2 = st.columns(2)
        with c1:
            if st.button("📐 DXF (R2000)"):
                try:
                    path=os.path.join(DB_DIR, f"{lot_slug}.dxf")
                    export_dxf_r2000(path, lon, lat)
                    st.success(f"Saved: {path}")
                except Exception as e:
                    st.error(f"DXF export failed: {e}")
        with c2:
            if st.button("📄 PDF"):
                try:
                    path=os.path.join(DB_DIR, f"{lot_slug}.pdf")
                    export_pdf(path, lon, lat, {"DATE":dt.date.today().isoformat()})
                    st.success(f"Saved: {path}")
                except Exception as e:
                    st.error(f"PDF export failed: {e}")
    else:
        st.info("No WORK polygon to export. Build one in the Input tab.")
