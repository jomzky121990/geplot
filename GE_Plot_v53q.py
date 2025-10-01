# GE_Plot_v53q.py
# GE Plot — Full build with edit/update/delete, PRS92 Z3/Z4 + Luzon1911 Z3/Z4 + UTM 51N,
# selection-driven Results/Map, and PDF/DXF/KML/KMZ export (WGS84 storage).
# Program: GE Plot    Author: Jomar B. Frogoso

import os, io, zipfile, tempfile, math, datetime as dt, shutil
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

# ----------------------------------------------------------------------
APP_NAME = "GE Plot v5.3q"
DB_DIR   = "GEPlotDB"
DB_NAME  = "GE_Plots"
BASE     = os.path.join(DB_DIR, DB_NAME)

# CRS menu (add/remove as needed)
CRS_OPTIONS = {
    "PRS92 Zone III (EPSG:3123)": 3123,
    "PRS92 Zone IV (EPSG:3124)": 3124,
    "Luzon 1911 Zone III (EPSG:25393)": 25393,
    "Luzon 1911 Zone IV (EPSG:25394)": 25394,
    "UTM 51N (EPSG:32651)": 32651,
}
DEFAULT_CRS_NAME = "PRS92 Zone III (EPSG:3123)"  # default input CRS
STORAGE_EPSG = 4326  # WGS84 for shapefile geometry and KML/KMZ

os.makedirs(DB_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
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

def polygon_area_perimeter(E,N):
    x, y = np.asarray(E,float), np.asarray(N,float)
    if x.size < 3: return 0.0, 0.0
    if x[0]!=x[-1] or y[0]!=y[-1]:
        x, y = np.append(x,x[0]), np.append(y,y[0])
    area = 0.5*np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
    per  = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    return abs(float(area)), per

def parse_bearing_qeb(b: str):
    if not isinstance(b,str): return None
    s = b.strip().upper()
    parts = s.split()
    if len(parts)!=3: return None
    q1, ang, q2 = parts
    if q1 not in ("N","S") or q2 not in ("E","W"): return None
    ang = ang.replace("°","").replace("’","-").replace("'","-").replace('"',"").replace("–","-").replace("—","-")
    seg = ang.split("-")
    try:
        d = int(seg[0]); m = int(seg[1]) if len(seg)>1 and seg[1] else 0
        return q1, d, m, q2
    except Exception:
        return None

def bearing_to_azimuth_deg(b: str) -> float:
    p = parse_bearing_qeb(b)
    if not p: raise ValueError(f"Bearing error: '{b}' → use 'N dd-mm E'")
    q1,d,m,q2 = p
    a = d + m/60.0
    if   q1=="N" and q2=="E": az=a
    elif q1=="S" and q2=="E": az=180-a
    elif q1=="S" and q2=="W": az=180+a
    else: az=360-a
    return float((az+360)%360)

def bd_to_ring(E0: float, N0: float, lines_df: pd.DataFrame):
    E=[E0]; N=[N0]
    for _,r in lines_df.iterrows():
        if pd.isna(r.get("Bearing")) or pd.isna(r.get("Distance")): continue
        az = bearing_to_azimuth_deg(str(r["Bearing"]))
        dist = float(r["Distance"])
        dE = dist*math.sin(math.radians(az))
        dN = dist*math.cos(math.radians(az))
        E.append(E[-1]+dE); N.append(N[-1]+dN)
    return np.asarray(E,float), np.asarray(N,float)

def slugify(s: str) -> str:
    import re
    if not s: return "geplot"
    s = re.sub(r'[^A-Za-z0-9]+', "_", s)
    return re.sub(r"_+","_",s).strip("_") or "geplot"

# ----------------------------------------------------------------------
# Shapefile I/O (WGS84 storage)
# ----------------------------------------------------------------------
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
    # write all records
    for _,r in df.iterrows():
        w.record(*[r.get(nm,"") for nm, *_ in FIELDS])
    # write shapes (WGS84 lon/lat)
    for shp in shapes:
        pts = shp.points if hasattr(shp, "points") else shp
        ring = list(pts)
        if ring and ring[0]!=ring[-1]: ring.append(ring[0])
        w.poly([ring])
    w.close()
    with open(base+".prj","w") as f:
        f.write(CRS.from_epsg(STORAGE_EPSG).to_wkt())

def append_polygon_record(base: str, lon: np.ndarray, lat: np.ndarray, attrs: dict):
    df, shps = read_db(base)
    new_id = (int(df["ID"].max())+1) if not df.empty else 1
    # Rebuild with an extra record
    w = new_writer(base)
    # old
    if not df.empty:
        for i, row in df.iterrows():
            w.record(*[row.get(nm,"") for nm, *_ in FIELDS])
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
    w.record(*[attrs_out.get(nm,"") for nm, *_ in FIELDS])
    ring = list(zip(lon, lat))
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

# ----------------------------------------------------------------------
# Selection sync → session_state
# ----------------------------------------------------------------------
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
        # geometry in WGS84
        lon = np.array([p[0] for p in shp.points], float)
        lat = np.array([p[1] for p in shp.points], float)
        # distances in record EPSG if available; else PRJ; else WGS84
        rec_epsg = int(row.get("SRC_EPSG", prj_epsg_or_default(base, STORAGE_EPSG))) \
                   if str(row.get("SRC_EPSG","")).strip().isdigit() else prj_epsg_or_default(base, STORAGE_EPSG)
        # transform to rec_epsg for lines
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

# ----------------------------------------------------------------------
# Streamlit Layout
# ----------------------------------------------------------------------
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)

if "db_base" not in st.session_state:
    st.session_state["db_base"] = BASE

# Sidebar: load shapefile, tie points CSV/XLSX
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
    st.subheader("Tie point file")
    tp_file = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx","xls"])

df_all, shapes_all = read_db(st.session_state["db_base"])

tab_input, tab_results, tab_map, tab_export = st.tabs(["Input", "Results", "Map Viewer", "Export"])

# ============================== INPUT =================================
with tab_input:
    st.header("Plot NEW Lot (Tie Point + Bearing/Distance)")

    # Tie point from file (optional)
    tpE = st.session_state.get("tpE", 0.0)
    tpN = st.session_state.get("tpN", 0.0)
    src_crs_name = st.session_state.get("src_crs_name", DEFAULT_CRS_NAME)
    src_epsg = CRS_OPTIONS[src_crs_name]

    if tp_file:
        try:
            if tp_file.name.lower().endswith(".csv"):
                tpdf = pd.read_csv(tp_file)
            else:
                tpdf = pd.read_excel(tp_file)
            st.success(f"Tie point file loaded: {len(tpdf)} rows")
            name_col = st.selectbox("Name column", list(tpdf.columns), index=0)
            ppcsE = st.selectbox("PPCS Easting col (optional)", ["(none)"]+list(tpdf.columns))
            ppcsN = st.selectbox("PPCS Northing col (optional)", ["(none)"]+list(tpdf.columns))
            prsE  = st.selectbox("PRS Easting col (optional)",  ["(none)"]+list(tpdf.columns))
            prsN  = st.selectbox("PRS Northing col (optional)", ["(none)"]+list(tpdf.columns))
            tpname = st.selectbox("Tie Point name", tpdf[name_col].astype(str).tolist())
            row_tp = tpdf[tpdf[name_col].astype(str)==tpname].iloc[0]
            st.caption("Pick which coordinates to use:")
            use = st.radio("Use", ["PRS92","PPCS Luzon 1911"], horizontal=True)
            if use=="PRS92" and prsE!="(none)" and prsN!="(none)":
                tpE = float(row_tp[prsE]); tpN = float(row_tp[prsN])
                src_crs_name = "PRS92 Zone III (EPSG:3123)"  # default; allow change below
            elif use=="PPCS Luzon 1911" and ppcsE!="(none)" and ppcsN!="(none)":
                tpE = float(row_tp[ppcsE]); tpN = float(row_tp[ppcsN])
                src_crs_name = "Luzon 1911 Zone III (EPSG:25393)"  # default; allow change below
            st.info(f"TP set to E={tpE:.3f}, N={tpN:.3f}")
        except Exception as e:
            st.error(f"Tie point parse failed: {e}")

    # Manual / confirm TP + CRS
    c1,c2,c3 = st.columns(3)
    with c1:
        tpE = st.number_input("Tie Point Easting", value=float(tpE), step=0.01, format="%.3f", key="tpE")
    with c2:
        tpN = st.number_input("Tie Point Northing", value=float(tpN), step=0.01, format="%.3f", key="tpN")
    with c3:
        src_crs_name = st.selectbox("Input CRS (for TP & BD)", list(CRS_OPTIONS.keys()),
                                    index=list(CRS_OPTIONS.keys()).index(src_crs_name))
        src_epsg = CRS_OPTIONS[src_crs_name]
        st.session_state["src_crs_name"] = src_crs_name

    # BD table
    st.subheader("Lines (Bearing & Distance)")
    st.caption("Format: N dd-mm E (seconds optional)")
    default_bd = pd.DataFrame({
        "Bearing":["N 57-59 W","S 79-46 E","S 08-44 W","N 80-14 W","N 06-47 E","S 79-35 E","S 79-40 E"],
        "Distance":[4631.98,20.77,79.67,97.64,80.78,29.93,49.71],
    })
    bd_tbl = st.data_editor(default_bd, num_rows="dynamic", use_container_width=True, key="bd_tbl")

    # Attributes for new lot
    st.subheader("Attributes")
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

    if st.button("🟩 Plot NEW Lot (append to shapefile)"):
        try:
            lines_df = bd_tbl.dropna(subset=["Bearing","Distance"])
            if lines_df.empty: st.error("Enter at least one BD row."); st.stop()
            E, N = bd_to_ring(tpE, tpN, lines_df)
            area, perim = polygon_area_perimeter(E,N)
            lon, lat = transform_coords(E, N, src_epsg, STORAGE_EPSG)
            append_polygon_record(st.session_state["db_base"], lon, lat, {
                "CLAIMANT": claim, "ADDRESS": addr, "LOT_NO": lotno,
                "SURVEY_NO": surv, "PATENT_NO": pat, "LOT_TYPE": ltype,
                "AREA": area, "PERIM": perim, "SRC_EPSG": src_epsg, "DATE": today
            })
            st.success("Lot appended to shapefile.")
            df_all, shapes_all = read_db(st.session_state["db_base"])
            st.session_state["sel_id"] = int(df_all["ID"].max())
            update_selection_from_id()
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.markdown("---")
    st.header("Existing Lots (DataGrid) • Edit / Replace Geometry / Delete")
    df_all, shapes_all = read_db(st.session_state["db_base"])
    if df_all.empty:
        st.info("No shapefile yet. Add your first lot above.")
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

            st.subheader("🔁 Replace geometry with CURRENT BD polygon")
            st.caption("Uses the polygon computed from the Tie Point & BD table above (input CRS).")
            if st.button("Replace geometry now"):
                try:
                    lines_df = bd_tbl.dropna(subset=["Bearing","Distance"])
                    if lines_df.empty: st.error("Enter at least one BD row."); st.stop()
                    E, N = bd_to_ring(tpE, tpN, lines_df)
                    # store in WGS84
                    lon, lat = transform_coords(E,N, src_epsg, STORAGE_EPSG)
                    df, shps = read_db(st.session_state["db_base"])
                    idx = df.index[df["ID"]==st.session_state["sel_id"]][0]
                    new_ring = list(zip(lon,lat))
                    if new_ring and new_ring[0]!=new_ring[-1]: new_ring.append(new_ring[0])
                    # replace
                    shps[idx] = type("Tmp", (), {"points": new_ring})()
                    # recompute area/perim in input CRS
                    area, perim = polygon_area_perimeter(E,N)
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

# ============================== RESULTS ===============================
with tab_results:
    st.header("Results")
    if "SEL_LINES" in st.session_state and isinstance(st.session_state["SEL_LINES"], pd.DataFrame) and not st.session_state["SEL_LINES"].empty:
        st.subheader("Lines — selected lot")
        st.dataframe(st.session_state["SEL_LINES"], use_container_width=True, hide_index=True)
    else:
        st.info("Select a lot in the Input tab or plot a new one.")

# ============================== MAP VIEWER ============================
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

    if "SEL_MAP_LON" in st.session_state and st.session_state["SEL_MAP_LON"] is not None:
        lon = np.asarray(st.session_state["SEL_MAP_LON"], float)
        lat = np.asarray(st.session_state["SEL_MAP_LAT"], float)
        if lon.size>=3:
            ring = list(zip(lat,lon))
            if ring[0]!=ring[-1]: ring.append(ring[0])
            folium.Polygon(ring, color="red", weight=2, fill=False,
                           popup=f"ID {st.session_state.get('sel_id','?')}").add_to(m)
            m.location = [float(lat.mean()), float(lon.mean())]
            m.zoom_start = 18
            st.caption("Selected lot displayed in WGS84.")
        else:
            st.info("Selected lot has no polygon geometry.")
    else:
        st.info("Select a lot in the Input tab.")

    st_folium(m, height=560, use_container_width=True)

# ============================== EXPORT ================================
def export_dxf_r2000(filepath: str, lon: np.ndarray, lat: np.ndarray):
    doc = ezdxf.new("R2000")  # AC1015
    msp = doc.modelspace()
    pts = list(zip(lon, lat))
    if pts[0]!=pts[-1]: pts.append(pts[0])
    msp.add_lwpolyline(pts, close=True)
    msp.add_text("GE Plot", dxfattribs={"height":2.5}).set_pos((pts[0][0], pts[0][1]), align="LEFT")
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
    # quick-fit polygon
    x = np.array(lon,float); ylat=np.array(lat,float)
    x = x - x.min(); ylat = ylat - ylat.min()
    sc = min((w-120)/(x.max()+1e-9), (h-200)/(ylat.max()+1e-9))
    xs = 60 + x*sc; ys = 120 + ylat*sc
    c.setStrokeColor(colors.red); c.setLineWidth(1.2)
    if xs.size>=3:
        for i in range(1,len(xs)):
            c.line(xs[i-1],ys[i-1], xs[i],ys[i])
        c.line(xs[-1],ys[-1], xs[0],ys[0])
    c.showPage(); c.save()

def export_kml(filepath:str, lon:np.ndarray, lat:np.ndarray, name:str, desc:str=""):
    from xml.sax.saxutils import escape
    lon = np.asarray(lon,float); lat=np.asarray(lat,float)
    if lon[0]!=lon[-1] or lat[0]!=lat[-1]:
        lon=np.append(lon,lon[0]); lat=np.append(lat,lat[0])
    coords = " ".join([f"{x:.8f},{y:.8f},0" for x,y in zip(lon,lat)])
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <name>{escape(name)}</name>
  <Placemark>
    <name>{escape(name)}</name>
    <description><![CDATA[{desc}]]></description>
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
    kml_tmp = filepath.replace(".kmz",".kml")
    export_kml(kml_tmp, lon, lat, name, desc)
    kml = simplekml.Kml()
    pol = kml.newpolygon(name=name, outerboundaryis=list(zip(lon,lat)))
    pol.style.linestyle.width = 2
    pol.style.linestyle.color = simplekml.Color.red
    pol.style.polystyle.color = simplekml.Color.changealphaint(80, simplekml.Color.red)
    kml.savekmz(filepath)
    try: os.remove(kml_tmp)
    except: pass

with tab_export:
    st.header("Export selected lot (WGS84)")
    if "SEL_MAP_LON" not in st.session_state or st.session_state["SEL_MAP_LON"] is None:
        st.info("Select a lot first.")
    else:
        lon = np.asarray(st.session_state["SEL_MAP_LON"], float)
        lat = np.asarray(st.session_state["SEL_MAP_LAT"], float)
        df_cur, _ = read_db(st.session_state["db_base"])
        row = df_cur[df_cur["ID"]==st.session_state.get("sel_id",-999)]
        attrs = row.iloc[0].to_dict() if not row.empty else {}
        lot_slug = slugify(attrs.get("LOT_NO","lot"))
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
                    export_pdf(pdf_path, lon, lat, attrs)
                    st.success(f"Saved: {pdf_path}")
                except Exception as e:
                    st.error(f"PDF export failed: {e}")
        with c3:
            if st.button("🌍 KML (WGS84)"):
                kml_path = os.path.join(DB_DIR, f"{lot_slug}.kml")
                try:
                    desc = f"Claimant: {attrs.get('CLAIMANT','')}<br/>Area: {attrs.get('AREA',0)} sq.m"
                    export_kml(kml_path, lon, lat, attrs.get("LOT_NO","GE Plot"), desc)
                    st.success(f"Saved: {kml_path}")
                except Exception as e:
                    st.error(f"KML export failed: {e}")
        with c4:
            if st.button("🌐 KMZ (WGS84)"):
                kmz_path = os.path.join(DB_DIR, f"{lot_slug}.kmz")
                try:
                    desc = f"Claimant: {attrs.get('CLAIMANT','')} | Area: {attrs.get('AREA',0)} sq.m"
                    export_kmz(kmz_path, lon, lat, attrs.get("LOT_NO","GE Plot"), desc)
                    st.success(f"Saved: {kmz_path}")
                except Exception as e:
                    st.error(f"KMZ export failed: {e}")
