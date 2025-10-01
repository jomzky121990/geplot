# GE_Plot_v53p.py
# GE Plot — full build with: tie points (PPCS/PRS), BD input, shapefile append,
# selection-driven Results + WGS84 Map, PDF+DXF export (R2000).
# Program: GE Plot   Author: Jomar B. Frogoso

import os, io, zipfile, tempfile, math, datetime as dt
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

# ------------------------------ Config --------------------------------

APP_NAME = "GE Plot v5.3p"
DB_DIR   = "GEPlotDB"
DB_NAME  = "GE_Plots"
BASE     = os.path.join(DB_DIR, DB_NAME)

DEFAULT_EPSG = 4326  # WGS84 for storage & map
ZONE_TO_EPSG = {      # Your earlier mapping may be richer; keep core choices here
    "PRS92 (EPSG:3123)": 3123,            # your PRS92 group used previously
    "PPCS Luzon 1911 Zone III": 25393,    # Luzon 1911 / Philippines zone III
    "PPCS Luzon 1911 Zone IV": 25394,     # Luzon 1911 / Philippines zone IV
}
DEFAULT_ZONE = "PRS92 (EPSG:3123)"  # default input CRS for BD calc

os.makedirs(DB_DIR, exist_ok=True)

# ----------------------------- Helpers --------------------------------

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
                with open(s,"rb") as r, open(dest_base+ext,"wb") as w:
                    w.write(r.read())
    return dest_base

def read_db(base: str) -> Tuple[pd.DataFrame, List]:
    if not os.path.exists(base+".shp"):
        return pd.DataFrame(), []
    sf = shapefile.Reader(base+".shp")
    fields = [f[0] for f in sf.fields[1:]]
    recs = [dict(zip(fields, r)) for r in sf.records()]
    shps = sf.shapes()
    df = pd.DataFrame(recs)
    if "ID" not in df.columns:
        df.insert(0,"ID",list(range(1,len(df)+1)))
    return df, shps

def prj_epsg_or_default(base: str, default_epsg: int = DEFAULT_EPSG) -> int:
    prj = base + ".prj"
    if os.path.exists(prj):
        try:
            c = CRS.from_wkt(open(prj).read())
            if c.to_epsg(): return int(c.to_epsg())
        except Exception:
            pass
    return default_epsg

def transform_coords(x,y, src_epsg, dst_epsg):
    if src_epsg == dst_epsg: return np.asarray(x,float), np.asarray(y,float)
    tr = Transformer.from_crs(CRS.from_epsg(src_epsg), CRS.from_epsg(dst_epsg), always_xy=True)
    X,Y = tr.transform(x,y)
    return np.asarray(X,float), np.asarray(Y,float)

def dist2d(ax,ay,bx,by): return float(math.hypot(bx-ax, by-ay))
def azimuth_deg(ax,ay,bx,by):
    dx,dy = bx-ax, by-ay
    return float((math.degrees(math.atan2(dx,dy))+360)%360)

def azimuth_to_qeb(az: float) -> str:
    if   0<=az<90:   q=("N","E"); a=az
    elif 90<=az<180:q=("S","E"); a=180-az
    elif 180<=az<270:q=("S","W");a=az-180
    else:           q=("N","W"); a=360-az
    d=int(a); m=int(round((a-d)*60))
    if m==60: d+=1; m=0
    return f"{q[0]} {d:02d}-{m:02d} {q[1]}"

def ring_to_lines(E,N)->pd.DataFrame:
    if len(E)<3: return pd.DataFrame(columns=["From","To","Bearing","Azimuth°","Distance (m)"])
    if E[0]!=E[-1] or N[0]!=N[-1]: E=list(E)+[E[0]]; N=list(N)+[N[0]]
    rows=[]
    for i in range(len(E)-1):
        aE,aN,bE,bN = E[i],N[i],E[i+1],N[i+1]
        d = dist2d(aE,aN,bE,bN)
        az= azimuth_deg(aE,aN,bE,bN)
        rows.append({"From":i+1,"To":i+2,"Bearing":azimuth_to_qeb(az),"Azimuth°":round(az,4),"Distance (m)":round(d,3)})
    return pd.DataFrame(rows)

def polygon_area_perimeter(E,N)->Tuple[float,float]:
    if len(E)<3: return 0.0,0.0
    x = np.array(E,float); y=np.array(N,float)
    if x[0]!=x[-1] or y[0]!=y[-1]:
        x=np.append(x,x[0]); y=np.append(y,y[0])
    area = 0.5*np.sum(x[:-1]*y[1:]-x[1:]*y[:-1])
    per  = np.sum(np.hypot(np.diff(x), np.diff(y)))
    return abs(float(area)), float(per)

# -------- Bearing parsing & polygon from BD + tie point ----------------

def parse_bearing_qeb(b: str) -> Optional[Tuple[str,int,int,str]]:
    # "N 57-59 W" | "S 08-44 W" | seconds optional or with 00
    if not isinstance(b,str) or len(b.strip())<3: return None
    s = b.strip().upper().replace("  ", " ")
    # allow "N 57-59-00 W" or "N 57-59 W"
    try:
        parts = s.split()
        if len(parts)!=3: return None
        q1, ang, q2 = parts[0], parts[1], parts[2]
        if q1 not in ("N","S") or q2 not in ("E","W"): return None
        ang=ang.replace("°","").replace("'", "-").replace("’","-").replace('"',"")
        seg=ang.split("-")
        d=int(seg[0]); m=int(seg[1]) if len(seg)>1 and seg[1] else 0
        return (q1,d,m,q2)
    except Exception:
        return None

def bearing_to_azimuth_deg(b: str) -> float:
    parsed = parse_bearing_qeb(b)
    if not parsed: raise ValueError(f"Bearing error: '{b}' → Use 'N dd-mm E'")
    q1,d,m,q2 = parsed
    a = d + m/60
    if q1=="N" and q2=="E": az = a
    elif q1=="S" and q2=="E": az = 180 - a
    elif q1=="S" and q2=="W": az = 180 + a
    else: az = 360 - a
    return float((az+360)%360)

def bd_to_ring(E0: float, N0: float, lines_df: pd.DataFrame) -> Tuple[np.ndarray,np.ndarray]:
    E=[E0]; N=[N0]
    for _,r in lines_df.iterrows():
        if pd.isna(r.get("Bearing")) or pd.isna(r.get("Distance")): continue
        az = bearing_to_azimuth_deg(str(r["Bearing"]))
        dist = float(r["Distance"])
        dE = dist*math.sin(math.radians(az))
        dN = dist*math.cos(math.radians(az))
        E.append(E[-1]+dE); N.append(N[-1]+dN)
    return np.asarray(E,float), np.asarray(N,float)

# ------------------------- Shapefile write/append ----------------------

def _open_writer(base: str, fields_schema: List[Tuple[str,str,int,int]]):
    # Create if not exists
    if not os.path.exists(base+".shp"):
        w = shapefile.Writer(base+".shp", shapeType=shapefile.POLYGON)
        for f in fields_schema:
            w.field(*f)
        return w, True
    # else open existing for append (pyshp style)
    w = shapefile.Writer(base+".shp", shapeType=shapefile.POLYGON)
    w.fields = shapefile.Reader(base+".shp").fields[1:]
    return w, False

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

def append_polygon_record(base: str, lon: np.ndarray, lat: np.ndarray, attrs: dict):
    # We store geometry in WGS84 (lon,lat)
    ensure_fields = FIELDS
    # Load existing to copy records then add one; pyshp has no in-place append+schema mgmt
    df, shps = read_db(base)
    # figure ID
    new_id = (int(df["ID"].max())+1) if not df.empty else 1
    attrs_out = {
        "ID": new_id,
        "CLAIMANT": attrs.get("CLAIMANT",""),
        "ADDRESS": attrs.get("ADDRESS",""),
        "LOT_NO": attrs.get("LOT_NO",""),
        "SURVEY_NO": attrs.get("SURVEY_NO",""),
        "PATENT_NO": attrs.get("PATENT_NO",""),
        "LOT_TYPE": attrs.get("LOT_TYPE",""),
        "AREA": float(attrs.get("AREA",0.0)),
        "PERIM": float(attrs.get("PERIM",0.0)),
        "SRC_EPSG": int(attrs.get("SRC_EPSG", DEFAULT_EPSG)),
        "DATE": attrs.get("DATE", dt.date.today().isoformat()),
    }

    # Rebuild the shapefile with all old records + new one
    w = shapefile.Writer(base+".shp", shapeType=shapefile.POLYGON)
    for f in ensure_fields: w.field(*f)

    # write old
    if not df.empty:
        for i, row in df.iterrows():
            shp = shps[i]
            pts = shp.points
            # polygon parts → pyshp expects rings in list-of-lists
            w.poly([pts])
            rec = []
            for nm, *_ in ensure_fields:
                if nm in row: rec.append(row[nm])
                else:
                    # computed for compatibility
                    if nm=="ID": rec.append(int(row["ID"]))
                    else: rec.append("")
            w.record(*rec)

    # write new
    ring = list(zip(lon, lat))
    if ring[0]!=ring[-1]: ring.append(ring[0])
    w.poly([ring])
    w.record(*[attrs_out.get(nm,"") for nm, *_ in ensure_fields])
    w.close()

    # write/refresh PRJ (WGS84)
    with open(base+".prj","w") as f:
        f.write(CRS.from_epsg(4326).to_wkt())

# --------------------- Selection → session_state ----------------------

def update_selection_from_id():
    try:
        base = st.session_state["db_base"]
        df, shps = read_db(base)
        if df.empty:
            for k in ("SEL_LINES","SEL_MAP_LON","SEL_MAP_LAT","SEL_EPSG"):
                st.session_state.pop(k, None)
            return
        sel_id = st.session_state.get("sel_id", None)
        if sel_id is None or sel_id not in df["ID"].tolist():
            for k in ("SEL_LINES","SEL_MAP_LON","SEL_MAP_LAT","SEL_EPSG"):
                st.session_state.pop(k, None)
            return
        idx = df.index[df["ID"]==sel_id][0]
        row = df.loc[idx]
        shp = shps[idx]
        src_epsg = prj_epsg_or_default(base, DEFAULT_EPSG)  # CRS of geometry
        rec_epsg = int(row.get("SRC_EPSG", src_epsg)) if str(row.get("SRC_EPSG","")).isdigit() else src_epsg

        lon = np.array([p[0] for p in shp.points], float)
        lat = np.array([p[1] for p in shp.points], float)

        # map
        lon_wgs, lat_wgs = transform_coords(lon, lat, src_epsg, 4326)

        # distance in rec_epsg → need projected coords
        E, N = transform_coords(lon, lat, src_epsg, rec_epsg)
        lines = ring_to_lines(E,N)

        st.session_state["SEL_MAP_LON"]=lon_wgs
        st.session_state["SEL_MAP_LAT"]=lat_wgs
        st.session_state["SEL_LINES"]=lines
        st.session_state["SEL_EPSG"]=rec_epsg
    except Exception as ex:
        for k in ("SEL_LINES","SEL_MAP_LON","SEL_MAP_LAT","SEL_EPSG"):
            st.session_state.pop(k, None)
        st.warning(f"Selection update failed: {ex}")

# -------------------------- Streamlit layout --------------------------

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(f"{APP_NAME}")

if "db_base" not in st.session_state:
    st.session_state["db_base"] = BASE

# ---- Sidebar: load shapefile
with st.sidebar:
    st.subheader("Shapefile")
    up = st.file_uploader("Upload shapefile ZIP", type=["zip"])
    if up:
        try:
            unzip_shp_to_base(up.read(), BASE)
            st.session_state["db_base"]=BASE
            st.success("Shapefile deployed.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
    # tiepoint file
    st.subheader("Tie point file")
    tp_file = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])

# Read DB
df_all, shapes_all = read_db(st.session_state["db_base"])

tab_input, tab_results, tab_map, tab_export = st.tabs(["Input", "Results", "Map Viewer", "Export"])

# ----------------------------- INPUT ---------------------------------

with tab_input:
    st.subheader("A) Tie point & BD input → Plot New Lot")

    # Parse tiepoint file
    tie_df = pd.DataFrame()
    if tp_file:
        try:
            if tp_file.name.lower().endswith(".csv"):
                tie_df = pd.read_csv(tp_file)
            else:
                tie_df = pd.read_excel(tp_file)
        except Exception as e:
            st.warning(f"Could not read file: {e}")
    # Normalize expected columns, based on your screenshot
    # Expected: TiePointName, PPCSEasting, PPCSNorthing, PRSEasting, PRSNorthing
    tp_cols = {c.lower():c for c in tie_df.columns}
    def getcol(name):
        for k,v in tp_cols.items():
            if k==name.lower(): return v
        return None

    if not tie_df.empty:
        st.success(f"Loaded {len(tie_df)} tie points.")
        name_col = getcol("TiePointName") or getcol("TiePoint") or list(tie_df.columns)[0]
        ppcs_e = getcol("PPCSEasting"); ppcs_n = getcol("PPCSNorthing")
        prs_e = getcol("PRSEasting") ; prs_n = getcol("PRSNorthing")

        tpn = st.selectbox("Tie point name", tie_df[name_col].tolist())
        row_tp = tie_df[tie_df[name_col]==tpn].iloc[0]

        st.markdown("**PPCS (Luzon 1911)**")
        pe = float(row_tp[ppcs_e]) if ppcs_e in row_tp and not pd.isna(row_tp[ppcs_e]) else 0.0
        pn = float(row_tp[ppcs_n]) if ppcs_n in row_tp and not pd.isna(row_tp[ppcs_n]) else 0.0
        st.write(f"Easting: `{pe}`  • Northing: `{pn}`")

        st.markdown("**PRS (PRS92)**")
        re_ = float(row_tp[prs_e]) if prs_e in row_tp and not pd.isna(row_tp[prs_e]) else 0.0
        rn_ = float(row_tp[prs_n]) if prs_n in row_tp and not pd.isna(row_tp[prs_n]) else 0.0
        st.write(f"Easting: `{re_}`  • Northing: `{rn_}`")

        crs_choice = st.radio("Use coordinates from", ["PRS (PRS92)", "PPCS (Luzon 1911)"], index=0, horizontal=True)
        zone_label = st.selectbox("If PPCS is chosen, pick Luzon 1911 zone", ["Luzon 1911 Zone III","Luzon 1911 Zone IV"], index=0)

        if crs_choice.startswith("PRS"):
            tpE, tpN = re_, rn_
            src_epsg = 3123  # PRS92 as used earlier
        else:
            tpE, tpN = pe, pn
            src_epsg = 25393 if "III" in zone_label else 25394
    else:
        st.info("Upload tie point CSV/XLSX to auto-fill TP. Otherwise you can type tie point & BD by hand.")
        tpE = st.number_input("Tie point Easting", value=0.0, step=0.01)
        tpN = st.number_input("Tie point Northing", value=0.0, step=0.01)
        src_epsg = ZONE_TO_EPSG.get(DEFAULT_ZONE, 3123)

    # Bearing-Distance table
    st.markdown("**Lines (Bearing & Distance)** — format: `N dd-mm E` (seconds optional)")
    default_bd = pd.DataFrame({
        "Bearing":["N 57-59 W","S 79-46 E","S 08-44 W","N 80-14 W","N 06-47 E","S 79-35 E","S 79-40 E"],
        "Distance":[4631.98,20.77,79.67,97.64,80.78,29.93,49.71],
    })
    lines_df = st.data_editor(default_bd, num_rows="dynamic", use_container_width=True, key="bd_tbl")

    # Attributes for new lot
    st.markdown("**Attributes**")
    colA,colB,colC = st.columns(3)
    with colA:
        claim = st.text_input("Claimant","")
        addr  = st.text_input("Address","")
        lotno = st.text_input("Lot No","")
    with colB:
        surv  = st.text_input("Survey No","")
        pat   = st.text_input("Patent No","")
        ltype = st.radio("Lot Type", ["RF","FP","SP SCHOOL","SP NGA"], index=0, horizontal=False)
    with colC:
        src_name = [k for k,v in ZONE_TO_EPSG.items() if v==src_epsg]
        st.text_input("Input CRS (EPSG of tie/lines)", f"{src_epsg}", disabled=True)
        today = dt.date.today().isoformat()

    col1,col2 = st.columns(2)
    with col1:
        if st.button("Plot NEW lot (append to shapefile)"):
            try:
                # Build polygon in input CRS from TP + BD
                lines_df_clean = lines_df.dropna(subset=["Bearing","Distance"])
                if lines_df_clean.empty: st.error("No BD rows."); st.stop()
                E_ring, N_ring = bd_to_ring(tpE, tpN, lines_df_clean)

                # compute area/perim in input CRS
                area, perim = polygon_area_perimeter(E_ring, N_ring)

                # Save geometry to shapefile in WGS84
                lon_in, lat_in = transform_coords(E_ring, N_ring, src_epsg, 4326)
                append_polygon_record(st.session_state["db_base"], lon_in, lat_in, {
                    "CLAIMANT": claim, "ADDRESS": addr, "LOT_NO": lotno, "SURVEY_NO": surv,
                    "PATENT_NO": pat, "LOT_TYPE": ltype, "AREA": area, "PERIM": perim,
                    "SRC_EPSG": src_epsg, "DATE": today
                })
                st.success("New lot saved to shapefile.")
                # refresh selection to the last ID
                df_ref,_ = read_db(st.session_state["db_base"])
                st.session_state["sel_id"] = int(df_ref["ID"].max())
                update_selection_from_id()
            except Exception as e:
                st.error(f"Plot/Save failed: {e}")

    st.markdown("---")
    st.subheader("B) Existing lots (from shapefile)")
    if df_all.empty:
        st.info("No shapefile features yet. Plot a new lot above.")
    else:
        st.dataframe(df_all, use_container_width=True)
        ids = df_all["ID"].tolist()
        if "sel_id" not in st.session_state: st.session_state["sel_id"] = ids[0]
        st.selectbox("Select record by ID", ids, key="sel_id", on_change=update_selection_from_id)
        if "SEL_LINES" not in st.session_state: update_selection_from_id()

# ---------------------------- RESULTS --------------------------------

with tab_results:
    st.header("Results")
    if "SEL_LINES" in st.session_state and isinstance(st.session_state["SEL_LINES"], pd.DataFrame) and not st.session_state["SEL_LINES"].empty:
        st.subheader("Lines — selected record")
        st.dataframe(st.session_state["SEL_LINES"], use_container_width=True, hide_index=True)
    else:
        st.info("Select a record in **Input** tab or plot a new lot.")

# ---------------------------- MAP VIEWER ------------------------------

def base_map(name:str)->folium.Map:
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
    m = base_map(bm)

    if "SEL_MAP_LON" in st.session_state and st.session_state["SEL_MAP_LON"] is not None:
        lon = np.asarray(st.session_state["SEL_MAP_LON"],float)
        lat = np.asarray(st.session_state["SEL_MAP_LAT"],float)
        if lon.size>=3:
            ring = list(zip(lat, lon))
            if ring[0]!=ring[-1]: ring.append(ring[0])
            folium.Polygon(ring, color="red", weight=2, fill=False,
                           popup=f"ID {st.session_state.get('sel_id','?')}").add_to(m)
            m.location = [float(lat.mean()), float(lon.mean())]
            m.zoom_start = 18
            st.caption("Selected record displayed in WGS84.")
    else:
        st.info("No selection yet.")

    st_folium(m, height=560, use_container_width=True)

# ------------------------------ EXPORT --------------------------------

def export_dxf_r2000(filepath: str, lon: np.ndarray, lat: np.ndarray):
    # Create simple DXF R2000 polyline in modelspace (coordinates in meters-like; we will write in WGS84 degrees though)
    doc = ezdxf.new("R2000")  # AC1015
    msp = doc.modelspace()
    pts = list(zip(lon, lat))
    if pts[0]!=pts[-1]: pts.append(pts[0])
    msp.add_lwpolyline(pts, close=True)
    msp.add_text("GE Plot", dxfattribs={"height":2.5}).set_pos((lon[0], lat[0]), align="LEFT")
    doc.saveas(filepath)

def export_pdf_simple(filepath: str, lon: np.ndarray, lat: np.ndarray, attrs: dict):
    c = canvas.Canvas(filepath, pagesize=A4)
    c.setTitle("GE Plot")
    w,h = A4
    # frame
    c.rect(30,30,w-60,h-60)
    # meta
    y=h-60
    for k in ["CLAIMANT","ADDRESS","LOT_NO","SURVEY_NO","PATENT_NO","LOT_TYPE","AREA","PERIM","SRC_EPSG","DATE"]:
        c.drawString(50,y,f"{k}: {attrs.get(k,'')}")
        y-=14
    # polygon quick-draw
    # scale to fit
    x = np.array(lon,float); ylat = np.array(lat,float)
    x = (x-x.min()); ylat=(ylat-ylat.min())
    sc = min((w-120)/(x.max()+1e-9), (h-200)/(ylat.max()+1e-9))
    xs = 60 + x*sc; ys = 120 + ylat*sc
    c.setStrokeColor(colors.red)
    c.setLineWidth(1.2)
    if xs.size>=3:
        c.moveTo(xs[0],ys[0])
        for i in range(1,len(xs)): c.line(xs[i-1],ys[i-1], xs[i],ys[i])
        c.line(xs[-1],ys[-1], xs[0],ys[0])
    c.showPage(); c.save()

with tab_export:
    st.header("Export selected lot")
    if ("SEL_MAP_LON" not in st.session_state) or (st.session_state["SEL_MAP_LON"] is None):
        st.info("Select a lot first.")
    else:
        lon = np.asarray(st.session_state["SEL_MAP_LON"],float)
        lat = np.asarray(st.session_state["SEL_MAP_LAT"],float)
        df_cur,_ = read_db(st.session_state["db_base"])
        row = df_cur[df_cur["ID"]==st.session_state.get("sel_id",-999)]
        attrs = row.iloc[0].to_dict() if not row.empty else {}
        col1,col2 = st.columns(2)
        with col1:
            if st.button("Export DXF (R2000)"):
                dxf_path = os.path.join(DB_DIR, f"lot_{st.session_state.get('sel_id','x')}.dxf")
                try:
                    export_dxf_r2000(dxf_path, lon, lat)
                    st.success(f"DXF saved: {dxf_path}")
                except Exception as e:
                    st.error(f"DXF export failed: {e}")
        with col2:
            if st.button("Export PDF"):
                pdf_path = os.path.join(DB_DIR, f"lot_{st.session_state.get('sel_id','x')}.pdf")
                try:
                    export_pdf_simple(pdf_path, lon, lat, attrs)
                    st.success(f"PDF saved: {pdf_path}")
                except Exception as e:
                    st.error(f"PDF export failed: {e}")
