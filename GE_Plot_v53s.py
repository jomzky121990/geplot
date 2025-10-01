# GE_Plot_v53s.py
# Program: GE Plot    Author: Jomar B. Frogoso
# Full build with:
# - Shapefile DB (WGS84) + Edit/Update/Delete
# - CRS options: PRS92 Z3/Z4, Luzon1911 Z3/Z4, UTM 51N
# - Selection -> Results (bearings/distances) & Map (WGS84)
# - Export: DXF (R2000), PDF, KML, KMZ
# - NEW input modes:
#    1) Bearing & Distance by PROMPTS (TP -> corner 1, then corner-to-corner)
#    2) Input by COORDINATES (#corners, Easting/Northing prompts)
#    3) (kept) Draw on Map (WGS84) optional scaffold

import os, io, zipfile, tempfile, math, datetime as dt, shutil, base64
from typing import Tuple, List, Optional

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

import simplekml  # for KMZ

# ------------------------------------------------------------------------------
APP_NAME = "GE Plot v5.3s"
DB_DIR   = "GEPlotDB"
DB_NAME  = "GE_Plots"
BASE     = os.path.join(DB_DIR, DB_NAME)

# CRS menu
CRS_OPTIONS = {
    "PRS92 Zone III (EPSG:3123)": 3123,
    "PRS92 Zone IV (EPSG:3124)": 3124,
    "Luzon 1911 Zone III (EPSG:25393)": 25393,
    "Luzon 1911 Zone IV (EPSG:25394)": 25394,
    "UTM 51N (EPSG:32651)": 32651,
    "WGS84 (EPSG:4326)": 4326,
}
DEFAULT_CRS_NAME = "PRS92 Zone III (EPSG:3123)"  # working CRS for TP/BD/coords
STORAGE_EPSG = 4326  # Shapefile & KML/KMZ geometry always saved in WGS84

os.makedirs(DB_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def prj_epsg_or_default(base: str, default_epsg: int = STORAGE_EPSG) -> int:
    prj = base + ".prj"
    if os.path.exists(prj):
        try:
            c = CRS.from_wkt(open(prj).read())
            if c.to_epsg():
                return int(c.to_epsg())
        except Exception:
            pass
    return default_epsg

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

# ------------------------------------------------------------------------------
# Bearing helpers (PROMPTS)
# ------------------------------------------------------------------------------
def parse_angle_ddmm(txt: str) -> float:
    """'DD-MM' -> decimal degrees (no seconds)."""
    s = (txt or "").strip().replace("°","").replace("’","-").replace("'","-").replace("–","-").replace("—","-")
    if not s: return 0.0
    parts = s.split("-")
    d = int(parts[0])
    m = int(parts[1]) if len(parts)>1 and parts[1] else 0
    return float(d + m/60.0)

def dir_angle_to_azimuth(direction: str, ddmm: str) -> float:
    """Direction in {N,S,E,W,NE,NW,SE,SW} + angle 'DD-MM' -> azimuth deg."""
    direction = (direction or "").upper().strip()
    a = parse_angle_ddmm(ddmm)
    # Cardinal directions (angle treated from axis)
    if direction == "N": return 0.0 + a  # along North turning to East
    if direction == "E": return 90.0 + a # along East turning to South
    if direction == "S": return 180.0 + a
    if direction == "W": return 270.0 + a
    # Quadrants
    if direction == "NE": return a
    if direction == "SE": return 180.0 - a
    if direction == "SW": return 180.0 + a
    if direction == "NW": return 360.0 - a
    # Fallback
    return a % 360.0

# ------------------------------------------------------------------------------
# Shapefile I/O (WGS84 storage)
# ------------------------------------------------------------------------------
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
    shp = base + ".shp"
    if not os.path.exists(shp):
        return pd.DataFrame(columns=[f[0] for f in FIELDS]), []
    sf = shapefile.Reader(shp)
    fields = [f[0] for f in sf.fields[1:]]
    recs = [dict(zip(fields, r)) for r in sf.records()]
    shps = sf.shapes()
    df = pd.DataFrame(recs)
    if "ID" not in df.columns:
        df.insert(0, "ID", list(range(1, len(df)+1)))
    return df, shps

def write_db(base: str, df: pd.DataFrame, shapes: List):
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    w = new_writer(base)
    for _,r in df.iterrows():
        w.record(*[r.get(nm,"") for nm, *_ in FIELDS])
    for shp in shapes:
        pts = shp.points if hasattr(shp,"points") else shp
        ring = list(pts)
        if ring and ring[0]!=ring[-1]: ring.append(ring[0])
        w.poly([ring])
    w.close()
    with open(base+".prj","w") as f:
        f.write(CRS.from_epsg(STORAGE_EPSG).to_wkt())

def append_polygon_record(base: str, lon: np.ndarray, lat: np.ndarray, attrs: dict):
    df, shps = read_db(base)
    new_id = (int(df["ID"].max())+1) if not df.empty else 1
    w = new_writer(base)
    # old
    if not df.empty:
        for i,row in df.iterrows():
            w.record(*[row.get(nm,"") for nm,*_ in FIELDS])
            ring = shps[i].points
            if ring and ring[0]!=ring[-1]: ring = ring + [ring[0]]
            w.poly([ring])
    # new
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
    with open(base+".prj","w") as f:
        f.write(CRS.from_epsg(STORAGE_EPSG).to_wkt())

def unzip_shp_to_base(file_bytes: bytes, dest_base: str) -> str:
    with tempfile.TemporaryDirectory() as tmpd:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            zf.extractall(tmpd)
        shp = None
        for root,_,files in os.walk(tmpd):
            for f in files:
                if f.lower().endswith(".shp"):
                    shp = os.path.join(root,f); break
            if shp: break
        if not shp: raise RuntimeError("ZIP has no .shp")
        src = os.path.splitext(shp)[0]
        for ext in (".shp",".shx",".dbf",".prj"):
            s = src+ext
            if os.path.exists(s):
                shutil.copy2(s, dest_base+ext)
    return dest_base

# ------------------------------------------------------------------------------
# Selection sync → session_state
# ------------------------------------------------------------------------------
def update_selection_from_id():
    try:
        base = st.session_state["db_base"]
        df, shps = read_db(base)
        if df.empty:
            for k in ("SEL_LINES","SEL_MAP_LON","SEL_MAP_LAT","SEL_EPSG","SEL_ATTR"):
                st.session_state.pop(k, None); return
        sel_id = st.session_state.get("sel_id", None)
        if sel_id is None or sel_id not in df["ID"].tolist():
            for k in ("SEL_LINES","SEL_MAP_LON","SEL_MAP_LAT","SEL_EPSG","SEL_ATTR"):
                st.session_state.pop(k, None); return
        idx = df.index[df["ID"]==sel_id][0]
        row = df.loc[idx]
        shp = shps[idx]
        lon = np.array([p[0] for p in shp.points], float)
        lat = np.array([p[1] for p in shp.points], float)
        rec_epsg = int(row.get("SRC_EPSG", prj_epsg_or_default(base, STORAGE_EPSG))) \
                   if str(row.get("SRC_EPSG","")).strip().isdigit() else prj_epsg_or_default(base, STORAGE_EPSG)
        E, N = transform_coords(lon, lat, STORAGE_EPSG, rec_epsg)
        st.session_state["SEL_LINES"] = ring_to_lines(E,N)
        st.session_state["SEL_EPSG"]  = rec_epsg
        st.session_state["SEL_MAP_LON"] = lon
        st.session_state["SEL_MAP_LAT"] = lat
        st.session_state["SEL_ATTR"] = row.to_dict()
    except Exception as ex:
        for k in ("SEL_LINES","SEL_MAP_LON","SEL_MAP_LAT","SEL_EPSG","SEL_ATTR"):
            st.session_state.pop(k, None)
        st.warning(f"Selection update failed: {ex}")

# ------------------------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------------------------
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)

if "db_base" not in st.session_state:
    st.session_state["db_base"] = BASE

# keep working CRS & temp polygon
st.session_state.setdefault("src_crs_name", DEFAULT_CRS_NAME)
st.session_state.setdefault("WORK_E", None)   # working polygon E (in working CRS)
st.session_state.setdefault("WORK_N", None)   # working polygon N
st.session_state.setdefault("WORK_LINES", None)

# Sidebar: shapefile & tie points
with st.sidebar:
    st.subheader("Data")
    up = st.file_uploader("Upload shapefile (.zip)", type=["zip"])
    if up:
        try:
            unzip_shp_to_base(up.read(), BASE)
            st.session_state["db_base"] = BASE
            st.success("Shapefile deployed.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
    st.subheader("Tie point CSV/XLSX (optional)")
    tp_file = st.file_uploader("Upload tiepoints", type=["csv","xlsx","xls"])

df_all, shapes_all = read_db(st.session_state["db_base"])

tab_input, tab_results, tab_map, tab_export = st.tabs(["Input", "Results", "Map Viewer", "Export"])

# ==============================================================================
# INPUT TAB
# ==============================================================================
def _init_prompt_state():
    st.session_state.setdefault("tpE", 0.0)
    st.session_state.setdefault("tpN", 0.0)
    st.session_state.setdefault("bd_prompt_segments", [])  # list of dicts for PROMPT mode
    st.session_state.setdefault("coord_rows", [])          # list of (E,N)

def _coord_set_n(n: int):
    n = max(3, int(n))
    rows = st.session_state.get("coord_rows", [])
    if len(rows) < n:
        rows = rows + [(np.nan, np.nan)]*(n - len(rows))
    else:
        rows = rows[:n]
    st.session_state["coord_rows"] = rows

with tab_input:
    _init_prompt_state()
    st.header("Create / Update Lots")

    # Working CRS (used for BD & coordinates computations)
    src_crs_name = st.selectbox("Working CRS (for BD & Coordinates)", list(CRS_OPTIONS.keys()),
                                index=list(CRS_OPTIONS.keys()).index(st.session_state["src_crs_name"]))
    st.session_state["src_crs_name"] = src_crs_name
    src_epsg = CRS_OPTIONS[src_crs_name]

    st.markdown("### A) Bearing & Distance — PROMPTS")
    c1,c2 = st.columns(2)
    with c1:
        tpE = st.number_input("Tie Point Easting", value=float(st.session_state["tpE"]), step=0.01, format="%.3f", key="tpE")
    with c2:
        tpN = st.number_input("Tie Point Northing", value=float(st.session_state["tpN"]), step=0.01, format="%.3f", key="tpN")

    n_corners = st.number_input("How many corners?", min_value=3, max_value=60, value=4, step=1, key="bdp_n")

    st.caption("Enter segment data. **Segment 1 is TP → Corner 1**. Segment 2 is Corner 1 → Corner 2, and so on.")
    seg_rows=[]
    for i in range(int(n_corners)):
        st.markdown(f"**Segment {i+1}**")
        cols = st.columns([1.0, 1.0, 1.0])
        with cols[0]:
            direction = st.selectbox(f"Direction {i+1}", ["N","S","E","W","NE","NW","SE","SW"], key=f"dir_{i}")
        with cols[1]:
            ang = st.text_input(f"Angle (DD-MM) {i+1}", value="0-00", key=f"ang_{i}")
        with cols[2]:
            dist = st.number_input(f"Distance (m) {i+1}", min_value=0.0, step=0.01, format="%.3f", key=f"dist_{i}")
        seg_rows.append({"dir":direction, "ang":ang, "dist":dist})

    if st.button("Build polygon from PROMPTS"):
        try:
            # Build vertices in working CRS
            E=[tpE]; N=[tpN]
            prevE, prevN = tpE, tpN
            for k,seg in enumerate(seg_rows):
                az = dir_angle_to_azimuth(seg["dir"], seg["ang"])
                dE = seg["dist"]*math.sin(math.radians(az))
                dN = seg["dist"]*math.cos(math.radians(az))
                if k==0:
                    # TP -> Corner 1
                    E1 = tpE + dE; N1 = tpN + dN
                    E.append(E1); N.append(N1)
                    prevE, prevN = E1, N1
                else:
                    E1 = prevE + dE; N1 = prevN + dN
                    E.append(E1); N.append(N1)
                    prevE, prevN = E1, N1

            E = np.asarray(E,float); N=np.asarray(N,float)
            st.session_state["WORK_E"] = E
            st.session_state["WORK_N"] = N
            st.session_state["WORK_LINES"] = ring_to_lines(E,N)
            st.success("PROMPT polygon built. See Results & Map Viewer. You can also save it in the Attributes section below.")
        except Exception as e:
            st.error(f"Build failed: {e}")

    st.markdown("---")
    st.markdown("### B) Input by Coordinates")
    num_corners = st.number_input("How many corners? (coordinates)", min_value=3, max_value=60, value=4, step=1,
                                  on_change=lambda: _coord_set_n(st.session_state.get('num_corners',4)), key="num_corners")

    if not st.session_state["coord_rows"]:
        _coord_set_n(int(num_corners))

    # Grid editor
    df_coords = pd.DataFrame(st.session_state["coord_rows"], columns=["Easting","Northing"])
    df_coords = st.data_editor(df_coords, num_rows="fixed", use_container_width=True, hide_index=True, key="coord_editor")
    # write back
    new_rows=[]
    for _, row in df_coords.iterrows():
        try:
            e = float(row["Easting"]); n = float(row["Northing"])
        except:
            e, n = np.nan, np.nan
        new_rows.append((e,n))
    st.session_state["coord_rows"] = new_rows

    if st.button("Build polygon from Coordinates"):
        vals = st.session_state["coord_rows"]
        if not vals or any(np.isnan(e) or np.isnan(n) for e,n in vals):
            st.error("Please fill all Easting/Northing values.")
        else:
            E = np.array([e for e,n in vals], float)
            N = np.array([n for e,n in vals], float)
            st.session_state["WORK_E"] = E
            st.session_state["WORK_N"] = N
            st.session_state["WORK_LINES"] = ring_to_lines(E,N)
            st.success("Coordinate polygon built. See Results & Map Viewer.")

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
                st.error("No working polygon. Build it first (PROMPTS or Coordinates).")
            else:
                # compute area & perim in working CRS
                area, perim = polygon_area_perimeter(st.session_state["WORK_E"], st.session_state["WORK_N"])
                # store geometry in WGS84
                lon, lat = transform_coords(st.session_state["WORK_E"], st.session_state["WORK_N"], src_epsg, STORAGE_EPSG)
                append_polygon_record(st.session_state["db_base"], lon, lat, {
                    "CLAIMANT": claim, "ADDRESS": addr, "LOT_NO": lotno,
                    "SURVEY_NO": surv, "PATENT_NO": pat, "LOT_TYPE": ltype,
                    "AREA": area, "PERIM": perim, "SRC_EPSG": src_epsg, "DATE": today
                })
                st.success("Saved to shapefile.")
                # refresh selection
                df_all,_ = read_db(st.session_state["db_base"])
                st.session_state["sel_id"] = int(df_all["ID"].max())
                update_selection_from_id()
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.markdown("---")
    st.header("Existing Lots (DataGrid) • Edit / Replace Geometry / Delete")
    df_all, shapes_all = read_db(st.session_state["db_base"])
    if df_all.empty:
        st.info("No shapefile yet. Add a lot above.")
    else:
        st.dataframe(df_all, use_container_width=True)
        ids = df_all["ID"].tolist()
        if "sel_id" not in st.session_state:
            st.session_state["sel_id"] = ids[0]
        st.selectbox("Select record by ID", ids, key="sel_id", on_change=update_selection_from_id)
        if "SEL_ATTR" not in st.session_state:
            update_selection_from_id()

        if "SEL_ATTR" in st.session_state:
            sel = st.session_state["SEL_ATTR"]
            st.subheader("✏️ Edit attributes")
            cols = st.columns(3)
            with cols[0]:
                e_claim = st.text_input("CLAIMANT", sel.get("CLAIMANT",""))
                e_addr  = st.text_input("ADDRESS",  sel.get("ADDRESS",""))
                e_lot   = st.text_input("LOT_NO",   sel.get("LOT_NO",""))
            with cols[1]:
                e_surv  = st.text_input("SURVEY_NO", sel.get("SURVEY_NO",""))
                e_pat   = st.text_input("PATENT_NO", sel.get("PATENT_NO",""))
                e_type  = st.text_input("LOT_TYPE",  sel.get("LOT_TYPE",""))
            with cols[2]:
                e_area  = st.number_input("AREA", value=float(sel.get("AREA",0.0)), format="%.3f")
                e_perim = st.number_input("PERIM", value=float(sel.get("PERIM",0.0)), format="%.3f")
                e_src   = st.number_input("SRC_EPSG", value=int(sel.get("SRC_EPSG", STORAGE_EPSG)), step=1)

            if st.button("💾 Update attributes"):
                try:
                    df, shps = read_db(st.session_state["db_base"])
                    idx = df.index[df["ID"]==st.session_state["sel_id"]][0]
                    df.loc[idx,"CLAIMANT"]=e_claim
                    df.loc[idx,"ADDRESS"]=e_addr
                    df.loc[idx,"LOT_NO"]=e_lot
                    df.loc[idx,"SURVEY_NO"]=e_surv
                    df.loc[idx,"PATENT_NO"]=e_pat
                    df.loc[idx,"LOT_TYPE"]=e_type
                    df.loc[idx,"AREA"]=float(e_area)
                    df.loc[idx,"PERIM"]=float(e_perim)
                    df.loc[idx,"SRC_EPSG"]=int(e_src)
                    write_db(st.session_state["db_base"], df, shps)
                    st.success("Attributes updated.")
                    update_selection_from_id()
                except Exception as e:
                    st.error(f"Update failed: {e}")

            st.subheader("🔁 Replace geometry with CURRENT WORK polygon")
            st.caption("Uses the polygon from PROMPTS or Coordinates (working CRS).")
            if st.button("Replace geometry now"):
                try:
                    if st.session_state["WORK_E"] is None or len(st.session_state["WORK_E"])<3:
                        st.error("No working polygon.")
                    else:
                        lon, lat = transform_coords(st.session_state["WORK_E"], st.session_state["WORK_N"], src_epsg, STORAGE_EPSG)
                        df, shps = read_db(st.session_state["db_base"])
                        idx = df.index[df["ID"]==st.session_state["sel_id"]][0]
                        new_ring = list(zip(lon,lat))
                        if new_ring and new_ring[0]!=new_ring[-1]: new_ring.append(new_ring[0])
                        shps[idx] = type("Tmp", (), {"points": new_ring})()
                        area, perim = polygon_area_perimeter(st.session_state["WORK_E"], st.session_state["WORK_N"])
                        df.loc[idx,"AREA"]=area; df.loc[idx,"PERIM"]=perim; df.loc[idx,"SRC_EPSG"]=int(src_epsg)
                        write_db(st.session_state["db_base"], df, shps)
                        st.success("Geometry replaced.")
                        update_selection_from_id()
                except Exception as e:
                    st.error(f"Replace failed: {e}")

            st.subheader("🗑️ Delete record")
            if st.button("Delete selected record"):
                try:
                    df, shps = read_db(st.session_state["db_base"])
                    idx = df.index[df["ID"]==st.session_state["sel_id"]][0]
                    df2 = df.drop(index=idx).reset_index(drop=True)
                    shps2 = [s for i,s in enumerate(shps) if i!=idx]
                    write_db(st.session_state["db_base"], df2, shps2)
                    st.success("Record deleted.")
                    st.session_state.pop("SEL_ATTR",None)
                    df_all, shapes_all = read_db(st.session_state["db_base"])
                    if not df_all.empty:
                        st.session_state["sel_id"]=int(df_all.iloc[0]["ID"])
                        update_selection_from_id()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

# ==============================================================================
# RESULTS TAB
# ==============================================================================
with tab_results:
    st.header("Results")

    # Priority: selected lot lines; otherwise working polygon lines
    if "SEL_LINES" in st.session_state and isinstance(st.session_state["SEL_LINES"], pd.DataFrame) and not st.session_state["SEL_LINES"].empty:
        st.subheader("Lines — selected lot")
        st.dataframe(st.session_state["SEL_LINES"], use_container_width=True, hide_index=True)
    elif st.session_state.get("WORK_LINES") is not None and not st.session_state["WORK_LINES"].empty:
        st.subheader("Lines — current WORK polygon")
        st.dataframe(st.session_state["WORK_LINES"], use_container_width=True, hide_index=True)
    else:
        st.info("Build a polygon (PROMPTS/Coordinates) or select a lot in Input tab.")

# ==============================================================================
# MAP VIEWER TAB
# ==============================================================================
def make_map(name:str)->folium.Map:
    if name=="OpenStreetMap":
        return folium.Map(location=[14.6,121.0], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    m=folium.Map(location=[14.6,121.0], zoom_start=5, tiles=None, control_scale=True)
    if name=="Google Satellite":
        folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                         attr="Google", name="Google Satellite").add_to(m)
    else:
        folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                         attr="Google", name="Google Hybrid").add_to(m)
    return m

with tab_map:
    st.header("Map Viewer")
    bm = st.selectbox("Basemap", ["OpenStreetMap","Google Satellite","Google Hybrid"], index=1)
    m = make_map(bm)

    # 1) Prefer selected lot geometry (WGS84)
    if "SEL_MAP_LON" in st.session_state and st.session_state["SEL_MAP_LON"] is not None:
        lon = np.asarray(st.session_state["SEL_MAP_LON"], float)
        lat = np.asarray(st.session_state["SEL_MAP_LAT"], float)
    # 2) Else show WORK polygon (transform to WGS84)
    elif st.session_state.get("WORK_E") is not None and len(st.session_state["WORK_E"])>=3:
        src_epsg = CRS_OPTIONS[st.session_state["src_crs_name"]]
        lon, lat = transform_coords(st.session_state["WORK_E"], st.session_state["WORK_N"], src_epsg, STORAGE_EPSG)
    else:
        lon, lat = None, None

    if lon is not None and lat is not None and len(lon)>=3:
        ring = list(zip(lat,lon))
        if ring[0]!=ring[-1]: ring.append(ring[0])
        folium.Polygon(ring, color="red", weight=2, fill=False,
                       popup=f"ID {st.session_state.get('sel_id','WORK')}").add_to(m)
        m.location = [float(np.mean(lat)), float(np.mean(lon))]
        m.zoom_start = 18
        st.caption("Polygon shown in WGS84.")
    else:
        st.info("No polygon to display yet.")

    st_folium(m, height=560, use_container_width=True)

# ==============================================================================
# EXPORT TAB
# ==============================================================================
def export_dxf_r2000(filepath: str, lon: np.ndarray, lat: np.ndarray):
    doc = ezdxf.new("R2000")  # AC1015
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
    x = np.array(lon,float); ylat=np.array(lat,float)
    if x.size>=3:
        x = x - x.min(); ylat = ylat - ylat.min()
        sc = min((w-120)/(x.max()+1e-9), (h-200)/(ylat.max()+1e-9))*0.9
        xs = 60 + x*sc; ys = 120 + ylat*sc
        c.setStrokeColor(colors.red); c.setLineWidth(1.2)
        for i in range(1,len(xs)): c.line(xs[i-1],ys[i-1], xs[i],ys[i])
        c.line(xs[-1],ys[-1], xs[0],ys[0])
    c.showPage(); c.save()

def export_kml(filepath:str, lon:np.ndarray, lat:np.ndarray, name:str, desc:str=""):
    lon = np.asarray(lon,float); lat=np.asarray(lat,float)
    if lon[0]!=lon[-1] or lat[0]!=lat[-1]:
        lon=np.append(lon,lon[0]); lat=np.append(lat,lat[0])
    coords = " ".join([f"{x:.8f},{y:.8f},0" for x,y in zip(lon,lat)])
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <name>{slugify(name)}</name>
  <Placemark>
    <name>{slugify(name)}</name>
    <Style><LineStyle><color>ff3333ff</color><width>2</width></LineStyle>
           <PolyStyle><color>553333ff</color></PolyStyle></Style>
    <Polygon><outerBoundaryIs><LinearRing><coordinates>
      {coords}
    </coordinates></LinearRing></outerBoundaryIs></Polygon>
  </Placemark>
</Document></kml>"""
    with open(filepath,"w",encoding="utf-8") as f:
        f.write(kml)

def export_kmz(filepath:str, lon:np.ndarray, lat:np.ndarray, name:str, desc:str=""):
    kml = simplekml.Kml()
    ring = list(zip(lon,lat))
    if ring and ring[0]!=ring[-1]: ring.append(ring[0])
    pol = kml.newpolygon(name=name, outerboundaryis=ring)
    pol.style.linestyle.width = 2
    pol.style.linestyle.color = simplekml.Color.red
    pol.style.polystyle.color = simplekml.Color.changealphaint(80, simplekml.Color.red)
    kml.savekmz(filepath)

with tab_export:
    st.header("Export selected lot (WGS84)")
    # Prefer selected lot; else working polygon
    if "SEL_MAP_LON" in st.session_state and st.session_state["SEL_MAP_LON"] is not None:
        lon = np.asarray(st.session_state["SEL_MAP_LON"], float)
        lat = np.asarray(st.session_state["SEL_MAP_LAT"], float)
        df_cur,_ = read_db(st.session_state["db_base"])
        row = df_cur[df_cur["ID"]==st.session_state.get("sel_id",-999)]
        attrs = row.iloc[0].to_dict() if not row.empty else {}
        label = attrs.get("LOT_NO","lot")
    elif st.session_state.get("WORK_E") is not None and len(st.session_state["WORK_E"])>=3:
        src_epsg = CRS_OPTIONS[st.session_state["src_crs_name"]]
        lon, lat = transform_coords(st.session_state["WORK_E"], st.session_state["WORK_N"], src_epsg, STORAGE_EPSG)
        attrs = {"LOT_NO":"WORK","CLAIMANT":"","AREA":0,"PERIM":0,"SRC_EPSG":src_epsg,"DATE":dt.date.today().isoformat()}
        label = "WORK"
    else:
        lon, lat, attrs, label = None, None, None, None

    if lon is None:
        st.info("Select a lot in Input tab or build a WORK polygon to export.")
    else:
        lot_slug = slugify(label)
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            if st.button("📐 DXF (R2000)"):
                dxf_path = os.path.join(DB_DIR, f"{lot_slug}.dxf")
                try:
                    export_dxf_r2000(dxf_path, lon, lat)
                    st.success(f"Saved: {dxf_path}")
                except Exception as e:
                    st.error(f"DXF export failed: {e}")
        with c2:
            if st.button("📄 PDF"):
                pdf_path = os.path.join(DB_DIR, f"{lot_slug}.pdf")
                try:
                    export_pdf(pdf_path, lon, lat, attrs or {})
                    st.success(f"Saved: {pdf_path}")
                except Exception as e:
                    st.error(f"PDF export failed: {e}")
        with c3:
            if st.button("🌍 KML"):
                kml_path = os.path.join(DB_DIR, f"{lot_slug}.kml")
                try:
                    export_kml(kml_path, lon, lat, label, "")
                    st.success(f"Saved: {kml_path}")
                except Exception as e:
                    st.error(f"KML export failed: {e}")
        with c4:
            if st.button("🌐 KMZ"):
                kmz_path = os.path.join(DB_DIR, f"{lot_slug}.kmz")
                try:
                    export_kmz(kmz_path, lon, lat, label, "")
                    st.success(f"Saved: {kmz_path}")
                except Exception as e:
                    st.error(f"KMZ export failed: {e}")
