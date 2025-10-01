# ============================================================
# GE Plot v5.3m — Input • Results • Map Viewer • Export
# Author: Jomar B. Frogoso
# ============================================================

import os, re, math, time, shutil, datetime
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Transformer, CRS
import shapefile  # pyshp
import folium
from streamlit_folium import st_folium
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import ezdxf
from ezdxf.addons import Importer
import matplotlib.pyplot as plt

APP_NAME = "GE Plot"
VERSION  = "v5.3m"

st.set_page_config(page_title=f"{APP_NAME} {VERSION}", layout="wide")

# ---------------- CRS options ----------------
ZONE_TO_EPSG = {
    "PRS92 Zone III": 3123,
    "PRS92 Zone IV": 3124,
    "Luzon 1911 Zone III": 25393,
    "Luzon 1911 Zone IV": 25394,
}
DEFAULT_ZONE = "PRS92 Zone III"
CRS_WGS84 = 4326

# ---------------- Strict schema (WGS84 shapefile) ----------------
CANON_SCHEMA = [
    ("ID","N",10,0),
    ("CLAIMANT","C",80,0),
    ("ADDRESS","C",120,0),
    ("LOT_NO","C",40,0),
    ("SURVEY_NO","C",40,0),
    ("PATENT_NO","C",40,0),
    ("LOT_TYPE","C",20,0),
    ("AREA","N",19,3),
    ("PERIM","N",19,3),
    ("SRC_EPSG","N",10,0),
    ("DATE","C",20,0),
]

# ---------------- Utils ----------------
def slugify(s: str) -> str:
    if not s: return "geplot"
    s = re.sub(r'[^A-Za-z0-9]+', "_", s)
    return re.sub(r"_+","_",s).strip("_") or "geplot"

def ensure_dir(path_wo_ext: str):
    os.makedirs(os.path.dirname(path_wo_ext) or ".", exist_ok=True)

def ensure_prj(base_wo_ext: str):
    ensure_dir(base_wo_ext)
    with open(base_wo_ext + ".prj", "w", encoding="utf-8") as f:
        f.write(CRS.from_epsg(4326).to_wkt())

def _retry(fn, tries=6, delay=0.6):
    last=None
    for _ in range(tries):
        try: return fn()
        except PermissionError as e:
            last=e; time.sleep(delay)
    if last: raise last

def transform_coords(xs, ys, epsg_from, epsg_to):
    """Always returns arrays; source can be lon/lat or projected — we don't assume units."""
    t = Transformer.from_crs(CRS.from_epsg(epsg_from), CRS.from_epsg(epsg_to), always_xy=True)
    X, Y = t.transform(xs, ys)
    return np.array(X), np.array(Y)

# ---- bearings
def bearing_to_azimuth_deg(bearing: str) -> float:
    s = str(bearing).upper().strip()
    s = (s.replace("º"," ").replace("°"," ")
           .replace("’","'").replace("′","'")
           .replace("“",'"').replace("”",'"')
           .replace("—","-").replace("–","-").replace("−","-"))
    m = re.match(r'^\s*([NS])\s*([0-9]{1,3})(?:[-\s:]?([0-9]{1,2}))?(?:[-\s:]?([0-9]{1,2}))?\s*([EW])\s*$', s)
    if not m:
        raise ValueError("Use quadrant style (e.g., N 30-00 E).")
    ns, dd, mm, ss, ew = m.groups()
    deg = int(dd) + (int(mm) if mm else 0)/60 + (int(ss) if ss else 0)/3600
    if   ns=="N" and ew=="E": az=deg
    elif ns=="S" and ew=="E": az=180-deg
    elif ns=="S" and ew=="W": az=180+deg
    else: az=360-deg
    return az%360.0

def azimuth_to_qeb(az: float) -> str:
    az%=360
    if az<90: ns,ew,a="N","E",az
    elif az<180: ns,ew,a="S","E",180-az
    elif az<270: ns,ew,a="S","W",az-180
    else: ns,ew,a="N","W",360-az
    d=int(a); m=int((a-d)*60); s=int(round((a-d-m/60)*3600))
    return f"{ns} {d:02d}-{m:02d}-{s:02d} {ew}"

def traverse_lines(E0,N0,rows):
    E=[E0]; N=[N0]
    for r in rows:
        az=bearing_to_azimuth_deg(str(r["Bearing"]))
        d=float(r["Distance"])
        rad=math.radians(az)
        E.append(E[-1]+d*math.sin(rad))
        N.append(N[-1]+d*math.cos(rad))
    return np.array(E),np.array(N)

def polygon_area(E,N):
    x=np.asarray(E); y=np.asarray(N)
    return 0.5*abs(np.dot(x[:-1],y[1:])-np.dot(y[:-1],x[1:]))

# ---------- Tie Point loader (CSV/XLSX) with diagnostics ----------
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())

def _to_num_series(s):
    """Robust numeric coercion: handles commas & scientific notation."""
    if s is None:
        return pd.Series(dtype=float)
    ser = pd.Series(s).astype(str)
    ser = ser.str.replace(r"[^0-9eE\-\+\,\.]", "", regex=True)
    mask = ser.str.count(",").fillna(0).eq(1) & ~ser.str.contains(r"\.")
    ser = ser.where(~mask, ser.str.replace(",", ".", regex=False))
    ser = ser.str.replace(",", "", regex=False)
    ser = ser.str.replace(r"^\+", "", regex=True)
    return pd.to_numeric(ser, errors="coerce")

def _read_any_table(file):
    name = getattr(file, "name", "")
    ext = os.path.splitext(name)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(file, dtype=str)
        except Exception:
            file.seek(0)
            return pd.read_excel(file, dtype=str, engine="openpyxl")
    file.seek(0)
    try:
        return pd.read_csv(file, dtype=str, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, dtype=str)

def parse_tiepoint_table(file) -> pd.DataFrame:
    df_raw = _read_any_table(file)
    if df_raw is None or df_raw.empty:
        raise ValueError("Empty or unreadable file.")

    cols = { _norm(c): c for c in df_raw.columns }
    name_col = next((cols[k] for k in cols if k in ("tiepointname","tpname","name","station","pointname")), None)
    ppcs_e   = next((cols[k] for k in cols if k in ("ppcseasting","ppcse","ppcseast","lpcseasting","lpcse")), None)
    ppcs_n   = next((cols[k] for k in cols if k in ("ppcsnorthing","ppcsn","ppcsnorth","lpcsnorthing","lpcsn")), None)
    prs_e    = next((cols[k] for k in cols if k in ("prseasting","prse","prseast")), None)
    prs_n    = next((cols[k] for k in cols if k in ("prsnorthing","prsn","prsnorth")), None)

    if name_col is None:
        raise ValueError(f"Couldn't find TiePoint name column in: {list(df_raw.columns)}")

    out = pd.DataFrame({
        "NAME":  df_raw[name_col].astype(str).str.strip(),
        "PPCS_E": _to_num_series(df_raw[ppcs_e]) if ppcs_e in df_raw else np.nan,
        "PPCS_N": _to_num_series(df_raw[ppcs_n]) if ppcs_n in df_raw else np.nan,
        "PRS_E":  _to_num_series(df_raw[prs_e])  if prs_e  in df_raw else np.nan,
        "PRS_N":  _to_num_series(df_raw[prs_n])  if prs_n  in df_raw else np.nan,
    })

    out["HAS_PPCS"] = out["PPCS_E"].notna() & out["PPCS_N"].notna()
    out["HAS_PRS"]  = out["PRS_E"].notna()  & out["PRS_N"].notna()
    return out

# ---------------- Shapefile I/O (strict schema, WGS84) ----------------
def _new_writer(base):
    w=shapefile.Writer(base, shapeType=shapefile.POLYGON, autoBalance=1)
    for n,t,s,d in CANON_SCHEMA: w.field(n,t,size=s,decimal=d)
    ensure_prj(base); return w

def _backup_all(base):
    ts=time.strftime("%Y%m%d_%H%M%S")
    for ext in (".shp",".shx",".dbf",".prj"):
        f=base+ext
        if os.path.exists(f): shutil.copy2(f, f+f".bak_{ts}")

def read_db(base)->tuple[pd.DataFrame,list]:
    if not os.path.exists(base+".shp"):
        return pd.DataFrame(columns=[f[0] for f in CANON_SCHEMA]), []
    r=shapefile.Reader(base)
    fields=[f[0] for f in r.fields if f[0]!="DeletionFlag"]
    df=pd.DataFrame(r.records(), columns=fields)
    shapes=r.shapes()
    r.close()
    return df, shapes

def write_db(base, df_attrs: pd.DataFrame, shapes):
    ensure_dir(base)
    w = _new_writer(base)

    cols = [f[0] for f in CANON_SCHEMA]
    for _, row in df_attrs[cols].iterrows():
        w.record(*list(row.values))

    for shp in shapes:
        if hasattr(shp, "points"):
            pts = list(getattr(shp, "points", []))
            parts = list(getattr(shp, "parts", []))
            if parts and len(parts) > 1:
                idxs = parts + [len(pts)]
                rings = [pts[idxs[j]:idxs[j+1]] for j in range(len(idxs)-1)]
                rings = [ring + ([ring[0]] if ring and ring[0] != ring[-1] else []) for ring in rings]
                w.poly(rings)
            else:
                ring = pts[:]
                if ring and ring[0] != ring[-1]:
                    ring.append(ring[0])
                w.poly([ring])

        elif isinstance(shp, (list, tuple)) and shp and isinstance(shp[0], (list, tuple, tuple)):
            ring = [(float(p[0]), float(p[1])) for p in shp]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            w.poly([ring])
        else:
            continue

    w.close()

def append_record(base, ring_proj_xy, attrs:dict, src_epsg:int):
    def _append():
        ensure_dir(base)
        df, shapes = read_db(base)
        next_id = 1 if df.empty else int(df["ID"].max())+1
        xs=np.array([p[0] for p in ring_proj_xy]); ys=np.array([p[1] for p in ring_proj_xy])
        lon,lat=transform_coords(xs,ys,src_epsg,CRS_WGS84)
        ring=[(float(x),float(y)) for x,y in zip(lon,lat)]
        if ring[0]!=ring[-1]: ring.append(ring[0])
        shapes.append(ring)
        row = {
            "ID":next_id,
            "CLAIMANT":attrs.get("CLAIMANT",""),
            "ADDRESS":attrs.get("ADDRESS",""),
            "LOT_NO":attrs.get("LOT_NO",""),
            "SURVEY_NO":attrs.get("SURVEY_NO",""),
            "PATENT_NO":attrs.get("PATENT_NO",""),
            "LOT_TYPE":attrs.get("LOT_TYPE",""),
            "AREA":float(attrs.get("AREA",0.0)),
            "PERIM":float(attrs.get("PERIM",0.0)),
            "SRC_EPSG":int(src_epsg),
            "DATE":attrs.get("DATE",datetime.date.today().isoformat()),
        }
        new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        write_db(base, new_df, shapes)
        return True
    return _retry(_append)

def rebuild_with_change(base, modifier):
    ensure_dir(base)
    _backup_all(base)
    df, shapes = read_db(base)
    df2, shapes2 = modifier(df.copy(), list(shapes))
    write_db(base, df2, shapes2)

# ---------------- Map helper (always shows WGS84) ----------------
def add_polygon_to_map(E_closed, N_closed, epsg, basemap="OpenStreetMap"):
    lon,lat=transform_coords(E_closed, N_closed, epsg, CRS_WGS84)
    cen=(float(np.mean(lat)), float(np.mean(lon)))
    m=folium.Map(location=cen, zoom_start=18, tiles=None)
    if basemap=="OpenStreetMap":
        folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
    else:
        sub=['mt0','mt1','mt2','mt3']
        lyrs={"Google Satellite":"s","Google Hybrid":"s,h"}
        folium.TileLayer(
            tiles=f"https://{'{s}'}.google.com/vt/lyrs={lyrs.get(basemap,'s')}&x={'{x}'}&y={'{y}'}&z={'{z}'}",
            attr="Google", name=basemap, max_zoom=20, subdomains=sub
        ).add_to(m)
    folium.PolyLine(list(zip(lat,lon)), color="#ff3333", weight=3).add_to(m)
    return m

def _update_selection_from_id():
    """Compute SEL_* from the current 'sel_id' and store in session_state."""
    try:
        base = st.session_state["db_base"]
        if not os.path.exists(base + ".shp"):
            for k in ("SEL_POLY_E","SEL_POLY_N","SEL_EPSG","SEL_LINES"):
                st.session_state.pop(k, None)
            return

        # read full shapefile (attributes + shapes), keep index aligned
        df_all, shapes_all = read_db(base)

        sel_id = st.session_state.get("sel_id", None)
        if sel_id is None or df_all.empty or sel_id not in df_all["ID"].tolist():
            for k in ("SEL_POLY_E","SEL_POLY_N","SEL_EPSG","SEL_LINES"):
                st.session_state.pop(k, None)
            return

        # exact row index in full dataframe
        idx = df_all.index[df_all["ID"] == sel_id][0]
        rec = df_all.loc[idx]

        # shape points (stored in WGS84)
        shp = shapes_all[idx]
        if hasattr(shp, "points"):
            lon = np.array([p[0] for p in shp.points], dtype=float)
            lat = np.array([p[1] for p in shp.points], dtype=float)
        else:
            lon = np.array([p[0] for p in shp], dtype=float)
            lat = np.array([p[1] for p in shp], dtype=float)

        # original source EPSG saved in attribute (fallback 4326 if missing)
        rec_epsg = int(rec.get("SRC_EPSG", 4326)) if str(rec.get("SRC_EPSG","")).strip().isdigit() else 4326

        # derive lines in the original projected CRS
        E_sel, N_sel, df_lines_sel = ring_to_lines(lon, lat, rec_epsg)

        # store for other tabs
        st.session_state["SEL_POLY_E"] = E_sel
        st.session_state["SEL_POLY_N"] = N_sel
        st.session_state["SEL_EPSG"]   = rec_epsg
        st.session_state["SEL_LINES"]  = df_lines_sel

    except Exception as ex:
        # clear any partial state on error
        for k in ("SEL_POLY_E","SEL_POLY_N","SEL_EPSG","SEL_LINES"):
            st.session_state.pop(k, None)
        # Optional: uncomment to see exact issue in the UI
        # st.warning(f"Selection update failed: {ex}")


# ---------------- PDF, KML, DXF (with template) ----------------
def export_pdf(filename, xs, ys, meta):
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4
    margin = 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, h - margin, f"GE Plot - {meta.get('LOT_NO','')}")
    c.setFont("Helvetica", 10)
    c.drawString(margin, h - margin - 16, f"Claimant: {meta.get('CLAIMANT','')}")
    c.drawString(margin, h - margin - 32, f"Area: {meta.get('AREA',0):,.3f} sq.m   Perimeter: {meta.get('PERIM',0):,.3f} m")
    c.drawString(margin, h - margin - 48, f"CRS: EPSG:{meta.get('EPSG','')}")

    fx0, fy0, fw, fh = margin, margin+80, w-2*margin, h- (margin+120)
    xs=np.asarray(xs); ys=np.asarray(ys)
    minx,maxx = float(xs.min()), float(xs.max())
    miny,maxy = float(ys.min()), float(ys.max())
    sx = fw / max(1e-9, (maxx-minx)); sy = fh / max(1e-9, (maxy-miny))
    s = min(sx, sy)*0.9
    cx, cy = fx0+fw/2, fy0+fh/2
    px = (xs - (minx+maxx)/2)*s + cx
    py = (ys - (miny+maxy)/2)*s + cy

    c.setStrokeColor(colors.black); c.rect(fx0, fy0, fw, fh)
    c.setStrokeColor(colors.red); c.setLineWidth(1.2)
    for i in range(len(px)-1):
        c.line(px[i], py[i], px[i+1], py[i+1])
    c.line(px[-1], py[-1], px[0], py[0])
    c.showPage(); c.save()

def export_kml(filename, xs, ys, epsg, attrs):
    lon, lat = transform_coords(np.asarray(xs), np.asarray(ys), epsg, 4326)
    if lon[0] != lon[-1] or lat[0] != lat[-1]:
        lon = np.append(lon, lon[0]); lat = np.append(lat, lat[0])
    name = attrs.get("LOT_NO","GE Plot")
    desc = (
        f"Claimant: {attrs.get('CLAIMANT','')}<br/>"
        f"Address: {attrs.get('ADDRESS','')}<br/>"
        f"Area: {attrs.get('AREA',0):,.3f} sq.m<br/>"
        f"Perimeter: {attrs.get('PERIM',0):,.3f} m<br/>"
        f"Source EPSG: {epsg}"
    )
    coords_str = " ".join([f"{x:.8f},{y:.8f},0" for x,y in zip(lon,lat)])
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <name>{name}</name>
  <Placemark>
    <name>{name}</name>
    <description><![CDATA[{desc}]]></description>
    <Style>
      <LineStyle><color>ff3333ff</color><width>2</width></LineStyle>
      <PolyStyle><color>553333ff</color></PolyStyle>
    </Style>
    <Polygon><outerBoundaryIs><LinearRing><coordinates>
      {coords_str}
    </coordinates></LinearRing></outerBoundaryIs></Polygon>
  </Placemark>
</Document></kml>"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(kml)

def _poly_area(pts):
    a=0.0
    for i in range(len(pts)-1):
        x1,y1=pts[i]; x2,y2=pts[i+1]
        a += x1*y2 - y1*x2
    return abs(a)*0.5

def _lwpoly_points(lw):
    pts=[(p[0],p[1]) for p in lw.get_points()]
    if len(pts)>=2 and pts[0]!=pts[-1] and lw.closed:
        pts.append(pts[0])
    return pts

def _find_frame_polyline(msp):
    preferred = {"lotplan_form","lotplan_frame","frame"}
    candidates=[]
    for e in msp.query("LWPOLYLINE"):
        try:
            if not e.closed: continue
        except Exception:
            continue
        layer = str(e.dxf.layer or "").lower()
        pts = _lwpoly_points(e)
        if len(pts) < 4: continue
        area = _poly_area(pts)
        score = (1 if layer in preferred else 0, area)
        candidates.append((score, pts))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0][0], t[0][1]))
    return candidates[-1][1]

def _draw_polygon_and_title(doc_or_none, xs, ys, meta, out_name=None):
    doc = doc_or_none or ezdxf.new("R2000")
    msp = doc.modelspace()
    if 'LOT' not in doc.layers:  doc.layers.add('LOT')
    if 'TITLE' not in doc.layers: doc.layers.add('TITLE')

    pts = [(float(x), float(y)) for x,y in zip(xs, ys)]
    if pts[0] != pts[-1]: pts.append(pts[0])
    msp.add_lwpolyline(pts, close=True, dxfattribs={'layer': 'LOT'})

    cx = float(np.mean(xs)); cy = float(np.mean(ys))
    text_lines = [
        f"LOT: {meta.get('LOT_NO','')}",
        f"CLAIMANT: {meta.get('CLAIMANT','')}",
        f"AREA: {meta.get('AREA',0):,.3f} SQ.M",
        f"EPSG: {meta.get('EPSG','')}",
    ]
    yoff = (max(1.0, (max(ys)-min(ys))*0.03))
    start_y = cy + 5*yoff
    for i, line in enumerate(text_lines):
        t = msp.add_text(line, dxfattribs={'height': yoff, 'layer': 'TITLE'})
        t.dxf.insert = (cx + 3*yoff, start_y - i*yoff)

    if out_name:
        doc.saveas(out_name)

def export_dxf_with_template(out_name, xs, ys, meta, template_path="BL-LOCA.dxf"):
    xs=np.asarray(xs); ys=np.asarray(ys)
    try:
        if os.path.exists(template_path):
            src = ezdxf.readfile(template_path)
            doc = ezdxf.new("R2000")
            imp = Importer(src, doc)
            imp.import_modelspace()
            for l in src.layers:
                if l.dxf.name not in doc.layers:
                    doc.layers.add(l.dxf.name)
            msp = doc.modelspace()
            frame_pts = _find_frame_polyline(msp)
            if frame_pts is None:
                _draw_polygon_and_title(doc, xs, ys, meta, out_name); return

            fx = np.array([p[0] for p in frame_pts]); fy = np.array([p[1] for p in frame_pts])
            fminx,fmaxx = float(fx.min()), float(fx.max())
            fminy,fmaxy = float(fy.min()), float(fy.max())
            fw, fh = (fmaxx - fminx), (fmaxy - fminy)
            fcx, fcy = (fminx+fmaxx)/2.0, (fminy+fmaxy)/2.0

            minx,maxx = float(xs.min()), float(xs.max())
            miny,maxy = float(ys.min()), float(ys.max())
            pw, ph = max(1e-9, (maxx-minx)), max(1e-9, (maxy-miny))
            pcx, pcy = (minx+maxx)/2.0, (miny+maxy)/2.0

            margin = 0.08
            s = min( (fw*(1-margin))/pw, (fh*(1-margin))/ph )
            tx = fcx + (xs - pcx) * s
            ty = fcy + (ys - pcy) * s

            if 'LOT' not in doc.layers:  doc.layers.add('LOT')
            pts = [(float(x), float(y)) for x,y in zip(tx, ty)]
            if pts[0] != pts[-1]: pts.append(pts[0])
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer":"LOT"})

            if 'TITLE' not in doc.layers: doc.layers.add('TITLE')
            tlx, tly = fminx + 0.02*fw, fminy + 0.02*fh
            text_lines = [
                f"LOT: {meta.get('LOT_NO','')}",
                f"CLAIMANT: {meta.get('CLAIMANT','')}",
                f"AREA: {meta.get('AREA',0):,.3f} SQ.M",
                f"EPSG: {meta.get('EPSG','')}",
            ]
            dy = 0.03*fh
            for i, line in enumerate(text_lines):
                t = msp.add_text(line, dxfattribs={'height': 0.025*fh, 'layer': 'TITLE'})
                t.dxf.insert = (tlx, tly + (len(text_lines)-1-i)*dy)

            doc.saveas(out_name)
        else:
            _draw_polygon_and_title(None, xs, ys, meta, out_name)
    except Exception:
        _draw_polygon_and_title(None, xs, ys, meta, out_name)

# --- Derive bearings & distances from a stored polygon (selected grid row) ---
def ring_to_lines(lon, lat, src_epsg: int):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size < 3 or lat.size < 3:
        return np.array([]), np.array([]), pd.DataFrame(columns=["From","To","Bearing","Azimuth°","Distance (m)"])
    if lon[0] != lon[-1] or lat[0] != lat[-1]:
        lon = np.append(lon, lon[0]); lat = np.append(lat, lat[0])
    # Transform to the record's original projected CRS
    E, N = transform_coords(lon, lat, 4326, src_epsg)
    rows = []
    for i in range(len(E)-1):
        dE = float(E[i+1] - E[i]); dN = float(N[i+1] - N[i])
        az = (math.degrees(math.atan2(dE, dN)) + 360.0) % 360.0
        qeb = azimuth_to_qeb(az)
        dist = float(math.hypot(dE, dN))
        rows.append({"From": i+1, "To": i+2, "Bearing": qeb, "Azimuth°": round(az, 4), "Distance (m)": round(dist, 3)})
    return np.asarray(E), np.asarray(N), pd.DataFrame(rows)

# ============================================================
#                           UI (4 tabs)
# ============================================================
tab_input, tab_results, tab_map, tab_export = st.tabs(["📥 Input", "📊 Results", "🗺️ Map Viewer", "📤 Export"])

# ========= INPUT =========
with tab_input:
    st.header("Input")

    # --- Tie Point CSV/XLSX picker ---
    st.subheader("Tie Point (from CSV/XLSX)")
    tp_file = st.file_uploader("Upload tie point table (CSV or XLSX)", type=["csv","xlsx","xls"], key="tp_csv")

    df_tp = None
    if tp_file:
        try:
            df_tp = parse_tiepoint_table(tp_file)
            total = len(df_tp)
            n_ppcs = int(df_tp["HAS_PPCS"].sum())
            n_prs  = int(df_tp["HAS_PRS"].sum())
            n_both = int((df_tp["HAS_PPCS"] & df_tp["HAS_PRS"]).sum())
            n_none = int((~df_tp["HAS_PPCS"] & ~df_tp["HAS_PRS"]).sum())
            st.success(f"Loaded {total} tie points. PPCS: {n_ppcs}, PRS: {n_prs}, Both: {n_both}, None: {n_none}")

            with st.expander("Preview / diagnostics", expanded=False):
                st.dataframe(df_tp.head(50), use_container_width=True)

            # ---- Column mapping override (optional) ----
            with st.expander("Column mapping (override if needed)", expanded=False):
                st.caption("If any coordinates show 0.000, select the correct columns below and press **Apply mapping**.")
                raw = _read_any_table(tp_file)
                headers = list(raw.columns)

                def pick(*cands):
                    for c in cands:
                        if c in headers: return c
                    for h in headers:
                        if "name" in h.lower(): return h
                    return headers[0] if headers else None

                name_sel  = st.selectbox("TiePoint Name column", headers, index=headers.index(pick("TiePointName","Station","Name")) if headers else 0, key="map_name")
                ppcs_e_sel= st.selectbox("PPCS Easting column", headers, index=headers.index(pick("PPCSEasting","LPCSEasting","Easting PPCS","Easting LPCS")) if headers else 0, key="map_ppcs_e")
                ppcs_n_sel= st.selectbox("PPCS Northing column", headers, index=headers.index(pick("PPCSNorthing","LPCSNorthing","Northing PPCS","Northing LPCS")) if headers else 0, key="map_ppcs_n")
                prs_e_sel = st.selectbox("PRS Easting column",  headers, index=headers.index(pick("PRSEasting","Easting PRS")) if headers else 0, key="map_prs_e")
                prs_n_sel = st.selectbox("PRS Northing column", headers, index=headers.index(pick("PRSNorthing","Northing PRS")) if headers else 0, key="map_prs_n")

                if st.button("Apply mapping"):
                    try:
                        df_tp = pd.DataFrame({
                            "NAME":  raw[name_sel].astype(str).str.strip(),
                            "PPCS_E": _to_num_series(raw[ppcs_e_sel]),
                            "PPCS_N": _to_num_series(raw[ppcs_n_sel]),
                            "PRS_E":  _to_num_series(raw[prs_e_sel]),
                            "PRS_N":  _to_num_series(raw[prs_n_sel]),
                        })
                        df_tp["HAS_PPCS"] = df_tp["PPCS_E"].notna() & df_tp["PPCS_N"].notna()
                        df_tp["HAS_PRS"]  = df_tp["PRS_E"].notna()  & df_tp["PRS_N"].notna()
                        st.success("Mapping applied.")
                    except Exception as ex:
                        st.error(f"Mapping failed: {ex}")

            use_all = st.checkbox("Show ALL names (even without coordinates)", value=False)
            if not use_all:
                df_tp = df_tp[df_tp["HAS_PPCS"] | df_tp["HAS_PRS"]]

            tp_name = st.selectbox("Tie Point Name", sorted(df_tp["NAME"].unique()), key="tp_name")
            rec = df_tp[df_tp["NAME"]==tp_name].iloc[0]

            has_prs  = bool(rec["HAS_PRS"])
            has_ppcs = bool(rec["HAS_PPCS"])

            cA,cB = st.columns(2)
            with cA:
                st.markdown("**PPCS (Luzon 1911)**")
                if has_ppcs:
                    st.write(f"Easting: `{rec.PPCS_E:.3f}`  •  Northing: `{rec.PPCS_N:.3f}`")
                else:
                    st.caption("— Not available —")
            with cB:
                st.markdown("**PRS (PRS92)**")
                if has_prs:
                    st.write(f"Easting: `{rec.PRS_E:.3f}`  •  Northing: `{rec.PRS_N:.3f}`")
                else:
                    st.caption("— Not available —")

            options=[]
            if has_prs:  options.append("PRS (PRS92)")
            if has_ppcs: options.append("PPCS (Luzon 1911)")
            if not options:
                st.error("Selected name has no usable coordinates."); st.stop()

            if "tp_choice" not in st.session_state:
                st.session_state["tp_choice"] = options[0]
            choice = st.radio("Use coordinates from", options, horizontal=True, key="tp_choice")

            if choice.startswith("PRS"):
                zone_sel = st.selectbox("PRS92 Zone", ["PRS92 Zone III","PRS92 Zone IV"], index=0, key="tp_zone_prs")
                useE, useN = float(rec.PRS_E), float(rec.PRS_N); chosen_zone = zone_sel
            else:
                zone_sel = st.selectbox("Luzon 1911 Zone", ["Luzon 1911 Zone III","Luzon 1911 Zone IV"], index=0, key="tp_zone_ppcs")
                useE, useN = float(rec.PPCS_E), float(rec.PPCS_N); chosen_zone = zone_sel

            if st.button("Use these coordinates", key="btn_use_tp"):
                st.session_state["tpE"]=useE; st.session_state["tpN"]=useN; st.session_state["zone"]=chosen_zone
                st.success(f"Applied: E={useE:.3f}, N={useN:.3f}  •  {chosen_zone}")
        except Exception as e:
            st.error(f"Failed to load tie point table: {e}")
    else:
        st.info("Headers can be any of: TiePointName / PPCS(Easting/Northing) / PRS(Easting/Northing). CSV or XLSX are fine.")

    # --- Tie point (manual/autofilled) + lines ---
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Tie Point (Projected)")
        tpE = st.number_input("Easting (m)", value=float(st.session_state.get("tpE", 500000.0)), format="%.3f")
        tpN = st.number_input("Northing (m)", value=float(st.session_state.get("tpN", 1000000.0)), format="%.3f")
        zone = st.selectbox("CRS / Zone", list(ZONE_TO_EPSG.keys()),
                             index=list(ZONE_TO_EPSG.keys()).index(st.session_state.get("zone", DEFAULT_ZONE)),
                             key="zone")
        epsg = ZONE_TO_EPSG[zone]
        st.caption(f"Using EPSG:{epsg}")

    with c2:
        st.subheader("Lines (Bearing & Distance)")
        st.caption("Format: N dd-mm ([-ss]) E")

        # Default demo lines
        _default_lines = pd.DataFrame([
            {"Bearing":"N 57-59 W","Distance":4631.98},
            {"Bearing":"S 79-46 E","Distance":20.77},
            {"Bearing":"S 08-44 W","Distance":79.67},
            {"Bearing":"N 80-14 W","Distance":97.64},
            {"Bearing":"N 06-47 E","Distance":80.78},
            {"Bearing":"S 79-35 E","Distance":29.93},
            {"Bearing":"S 79-40 E","Distance":49.71},
        ])

        # If a lot was selected in the grid, use its derived lines once
        if "lines_override" in st.session_state and isinstance(st.session_state["lines_override"], pd.DataFrame):
            default_lines = st.session_state.pop("lines_override")
            default_lines = default_lines[["Bearing","Distance"]].copy()
        else:
            default_lines = _default_lines.copy()

        lines_value = st.data_editor(
            default_lines,
            num_rows="dynamic",
            use_container_width=True,
            key="lines_tbl"
        )

    tie_is_first = st.checkbox("First row is TIE (exclude from area & perimeter)", value=True)

    # Attributes
    st.subheader("Lot Attributes")
    a1,a2 = st.columns(2)
    with a1:
        st.text_input("Claimant", key="attr_claimant")
        st.text_input("Lot Number", key="attr_lot_no")
        st.text_input("Survey Number", key="attr_survey_no")
    with a2:
        st.text_input("Address", key="attr_address")
        st.text_input("Patent Number", key="attr_patent_no")
        st.radio("Lot Classification", ["RF","FP","SP SCHOOL","SP NGA"], index=0, key="attr_lot_type")

    # Save path
    if "db_base" not in st.session_state:
        st.session_state["db_base"] = os.path.join("GEPlotDB","GE_Plots")
    st.text_input("Shapefile base path (no extension)", key="db_base",
                  help="Example: C:/GEPlotDB/GE_Plots  (WGS84)")

    # --- Convert editor to DF & validate ---
    def editor_to_df(obj, expected):
        if isinstance(obj, pd.DataFrame): df=obj.copy()
        elif isinstance(obj, dict) and "data" in obj: df=pd.DataFrame(obj["data"])
        else: df=pd.DataFrame(columns=expected)
        df=df.rename(columns={c:str(c).strip() for c in df.columns})
        for col in expected:
            if col not in df.columns: df[col]=np.nan
        return df[expected]

    lines_df = editor_to_df(lines_value, ["Bearing","Distance"])
    lines_df["Bearing"]=lines_df["Bearing"].astype(str).str.strip()
    ds = lines_df["Distance"].astype(str).str.strip()
    ds = ds.str.replace(r"[^\d\-,\.]","",regex=True)
    mask = ds.str.count(",").fillna(0).eq(1) & ~ds.str.contains(r"\.")
    ds = ds.where(~mask, ds.str.replace(",",".",regex=False))
    ds = ds.str.replace(",", "", regex=False)
    lines_df["Distance"]=pd.to_numeric(ds,errors="coerce")
    lines_df=lines_df.dropna(subset=["Distance"])
    lines_df=lines_df[lines_df["Bearing"]!=""]
    if lines_df.empty:
        st.warning("Enter at least one Bearing & Distance row to compute.")
        st.stop()
    for i,r in lines_df.iterrows():
        try: _=bearing_to_azimuth_deg(r["Bearing"])
        except Exception as e:
            st.error(f"Bearing error row {i+1}: {e}"); st.stop()

    # Traverse
    E_line,N_line=traverse_lines(tpE,tpN,lines_df.to_dict(orient="records"))
    if tie_is_first and len(E_line)>=2: E_bnd,N_bnd=E_line[1:],N_line[1:]
    else: E_bnd,N_bnd=E_line,N_line
    E_closed=np.append(E_bnd,E_bnd[0]); N_closed=np.append(N_bnd,N_bnd[0])
    area=float(polygon_area(E_closed,N_closed))
    perimeter=float(np.sum(np.hypot(np.diff(E_bnd), np.diff(N_bnd))))

    st.info(f"Area: {area:.3f} sq.m • Perimeter: {perimeter:.3f} m")

    # Save (append)
    attrs_for_db = {
        "CLAIMANT": st.session_state.get("attr_claimant",""),
        "ADDRESS":  st.session_state.get("attr_address",""),
        "LOT_NO":   st.session_state.get("attr_lot_no",""),
        "SURVEY_NO":st.session_state.get("attr_survey_no",""),
        "PATENT_NO":st.session_state.get("attr_patent_no",""),
        "LOT_TYPE": st.session_state.get("attr_lot_type",""),
        "AREA":     area,
        "PERIM":    perimeter,
        "DATE":     datetime.date.today().isoformat(),
    }

    if st.button("🟩 Plot New Lot (Save)", key="btn_add_save"):
        try:
            ring_proj=list(zip(E_closed.tolist(), N_closed.tolist()))
            append_record(st.session_state["db_base"], ring_proj, attrs_for_db, epsg)
            st.success("Saved to shapefile (WGS84). DataGrid refreshed below.")
        except Exception as e:
            st.error(f"Save failed: {e}")

    # ---------------- DataGrid ----------------
    st.markdown("---")
    st.subheader("Database Records")

    base = st.session_state["db_base"]
    shp = base + ".shp"
    if not os.path.exists(shp):
        st.info("No shapefile yet. Add your first lot to create it.")
    else:
        df, shapes = read_db(base)
        if df.empty:
            st.info("Shapefile has no records yet.")
        else:
            search = st.text_input("🔎 Search (any field)", "")
            df_view = df.copy()
            if search:
                m = df_view.apply(lambda r: r.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
                df_view = df_view[m]

            st.dataframe(df_view, use_container_width=True, hide_index=True)

            ids = df_view["ID"].tolist()
default_idx = 0 if len(ids) == 0 else ids.index(ids[0])

# Keep the selected ID in session_state and compute on change
st.selectbox(
    "Select record by ID",
    ids,
    index=default_idx,
    key="sel_id",
    on_change=_update_selection_from_id,  # <<— triggers immediate update
)

# Show selected record details
sel_id = st.session_state.get("sel_id", ids[0] if ids else None)
if sel_id is not None:
    idx_full = df.index[df["ID"] == sel_id][0]  # index in the FULL df
    sel_row = df.loc[idx_full]
    st.markdown("**Selected record:**")
    st.json(sel_row.to_dict())

    # (Optional) also push tie point to the editor using the first vertex
    try:
        _, shapes_full = read_db(base)
        shp_sel = shapes_full[idx_full]
        if hasattr(shp_sel, "points"):
            lon = np.array([p[0] for p in shp_sel.points], dtype=float)
            lat = np.array([p[1] for p in shp_sel.points], dtype=float)
        else:
            lon = np.array([p[0] for p in shp_sel], dtype=float)
            lat = np.array([p[1] for p in shp_sel], dtype=float)

        rec_epsg = int(sel_row.get("SRC_EPSG", 4326)) if str(sel_row.get("SRC_EPSG","")).strip().isdigit() else 4326
        E_sel, N_sel = transform_coords(lon, lat, 4326, rec_epsg)
        if E_sel.size >= 1 and N_sel.size >= 1:
            st.session_state["tpE"] = float(E_sel[0])
            st.session_state["tpN"] = float(N_sel[0])
            inv = {v:k for k,v in ZONE_TO_EPSG.items()}
            st.session_state["zone"] = inv.get(rec_epsg, st.session_state.get("zone", DEFAULT_ZONE))
    except Exception:
        pass


            # ---- derive lines from selected record & store for other tabs; also push to editor ----
            try:
                _, shapes_all = read_db(base)
                shp_sel = shapes_all[idx]
                if hasattr(shp_sel, "points"):
                    lon = np.array([p[0] for p in shp_sel.points], dtype=float)
                    lat = np.array([p[1] for p in shp_sel.points], dtype=float)
                else:
                    lon = np.array([p[0] for p in shp_sel], dtype=float)
                    lat = np.array([p[1] for p in shp_sel], dtype=float)
                rec_epsg = int(sel_row.get("SRC_EPSG", 4326)) if str(sel_row.get("SRC_EPSG","")).strip().isdigit() else 4326
                E_sel, N_sel, df_lines_sel = ring_to_lines(lon, lat, rec_epsg)

                # expose to other tabs
                st.session_state["SEL_POLY_E"] = E_sel
                st.session_state["SEL_POLY_N"] = N_sel
                st.session_state["SEL_EPSG"]   = rec_epsg
                st.session_state["SEL_LINES"]  = df_lines_sel

                # push lines into the Lines editor (once)
                if not df_lines_sel.empty:
                    lines_override = df_lines_sel[["Bearing","Distance (m)"]].rename(columns={"Distance (m)":"Distance"}).copy()
                    st.session_state["lines_override"] = lines_override

                    # refresh the tie point to first vertex in projected CRS
                    if E_sel.size >= 1 and N_sel.size >= 1:
                        st.session_state["tpE"] = float(E_sel[0])
                        st.session_state["tpN"] = float(N_sel[0])
                        inv = {v:k for k,v in ZONE_TO_EPSG.items()}
                        st.session_state["zone"] = inv.get(rec_epsg, st.session_state.get("zone", DEFAULT_ZONE))

            except Exception as ex:
                st.warning(f"Could not derive lines from selected shape: {ex}")
                for k in ("SEL_POLY_E","SEL_POLY_N","SEL_EPSG","SEL_LINES"):
                    st.session_state.pop(k, None)

            with st.expander("✏️ Edit attributes"):
                edit_vals={}
                form_cols = st.columns(2)
                left_fields = ["CLAIMANT","LOT_NO","SURVEY_NO","LOT_TYPE","AREA","PERIM"]
                right_fields= ["ADDRESS","PATENT_NO","SRC_EPSG","DATE"]
                with form_cols[0]:
                    for c in left_fields:
                        v = sel_row.get(c,"")
                        if c in ("AREA","PERIM"): v = st.number_input(c, value=float(v), format="%.3f")
                        else: v = st.text_input(c, value=str(v))
                        edit_vals[c]=v
                with form_cols[1]:
                    for c in right_fields:
                        v = sel_row.get(c,"")
                        if c=="SRC_EPSG":
                            v = st.number_input(c, value=int(v) if str(v).isdigit() else 4326, step=1)
                        else:
                            v = st.text_input(c, value=str(v))
                        edit_vals[c]=v

                if st.button("💾 Update attributes"):
                    def modifier(dfa, shp_list):
                        dfa.loc[dfa["ID"]==sel_id, left_fields+right_fields] = [
                            edit_vals["CLAIMANT"], edit_vals["LOT_NO"], edit_vals["SURVEY_NO"], edit_vals["LOT_TYPE"],
                            float(edit_vals["AREA"]), float(edit_vals["PERIM"]),
                            edit_vals["ADDRESS"], edit_vals["PATENT_NO"], int(edit_vals["SRC_EPSG"]), edit_vals["DATE"]
                        ]
                        return dfa, shp_list
                    try:
                        rebuild_with_change(base, modifier)
                        st.success("Attributes updated. Click 'Rerun' (top-right) to refresh.")
                    except Exception as e:
                        st.error(f"Update failed: {e}")

            with st.expander("🧭 Replace geometry with CURRENT polygon"):
                st.caption("Uses the polygon computed from tie point & lines above.")
                if st.button("Replace geometry now"):
                    def modifier(dfa, shp_list):
                        xs=np.array(E_closed); ys=np.array(N_closed)
                        lon,lat=transform_coords(xs,ys,epsg,CRS_WGS84)
                        ring=[(float(x),float(y)) for x,y in zip(lon,lat)]
                        if ring[0]!=ring[-1]: ring.append(ring[0])
                        shp_list[idx]=ring
                        dfa.loc[dfa["ID"]==sel_id,"AREA"]=float(area)
                        dfa.loc[dfa["ID"]==sel_id,"PERIM"]=float(perimeter)
                        dfa.loc[dfa["ID"]==sel_id,"SRC_EPSG"]=int(epsg)
                        dfa.loc[dfa["ID"]==sel_id,"DATE"]=datetime.date.today().isoformat()
                        return dfa, shp_list
                    try:
                        rebuild_with_change(base, modifier)
                        st.success("Geometry replaced. Click 'Rerun' to refresh.")
                    except Exception as e:
                        st.error(f"Replace failed: {e}")

            with st.expander("🗑️ Delete record"):
                if st.button("Delete selected record"):
                    def modifier(dfa, shp_list):
                        keep = dfa["ID"]!=sel_id
                        dfa2 = dfa[keep].copy()
                        shp2 = [s for k,s in enumerate(shp_list) if k!=idx]
                        return dfa2, shp2
                    try:
                        rebuild_with_change(base, modifier)
                        st.success("Record deleted. Click 'Rerun' to refresh.")
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

# ========= RESULTS =========
with tab_results:
    st.header("Results")

    # If a record is selected, show its lines FIRST
    if (
        "SEL_LINES" in st.session_state
        and isinstance(st.session_state["SEL_LINES"], pd.DataFrame)
        and not st.session_state["SEL_LINES"].empty
    ):
        st.markdown("### Selected record (from shapefile)")
        st.dataframe(st.session_state["SEL_LINES"], use_container_width=True, hide_index=True)
    else:
        st.info("Select a record in the Input tab to view its bearings & distances here.")

    # Also show the current input polygon lines (optional, as a reference)
    try:
        rows = lines_df.to_dict(orient="records")
        out_rows=[]
        for i,r in enumerate(rows, start=1):
            az = bearing_to_azimuth_deg(r["Bearing"])
            out_rows.append({
                "From": i, "To": i+1,
                "Bearing": azimuth_to_qeb(az),
                "Azimuth°": round(az,4),
                "Distance (m)": round(float(r["Distance"]),3)
            })
        st.markdown("### Current input polygon (reference)")
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)
    except Exception:
        pass


# ========= MAP VIEWER =========
with tab_map:
    st.header("Map Viewer")
    base_map = st.selectbox("Basemap", ["OpenStreetMap","Google Satellite","Google Hybrid"], index=0)

    # If a selection exists, always show it; otherwise show current input polygon
    sel_available = (
        "SEL_POLY_E" in st.session_state and
        isinstance(st.session_state["SEL_POLY_E"], np.ndarray) and
        st.session_state["SEL_POLY_E"].size >= 3
    )

    try:
        if sel_available:
            Emap = st.session_state["SEL_POLY_E"]
            Nmap = st.session_state["SEL_POLY_N"]
            epsg_map = st.session_state["SEL_EPSG"]
            st.caption("Showing selected record from shapefile.")
        else:
            Emap = E_closed
            Nmap = N_closed
            epsg_map = ZONE_TO_EPSG.get(st.session_state.get("zone", DEFAULT_ZONE), 3123)
            st.caption("No selection. Showing current input polygon.")

        # Map always renders in WGS84 internally
        m = add_polygon_to_map(np.asarray(Emap), np.asarray(Nmap), epsg_map, base_map)
        st_folium(m, height=520, use_container_width=True)
    except Exception as e:
        st.error(f"Map error: {e}")


# ========= EXPORT =========
with tab_export:
    st.header("Export")
    try:
        lot_no = st.session_state.get("attr_lot_no","")
        lot_slug = slugify(lot_no) or "lot"
        xs = np.array(E_closed); ys = np.array(N_closed)
        epsg_current = ZONE_TO_EPSG.get(st.session_state.get("zone", DEFAULT_ZONE), 3123)

        meta_common = {
            "LOT_NO": lot_no,
            "CLAIMANT": st.session_state.get("attr_claimant",""),
            "ADDRESS":  st.session_state.get("attr_address",""),
            "AREA":     float(polygon_area(xs,ys)),
            "PERIM":    float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))),
            "EPSG":     epsg_current,
        }

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("📄 Export PDF"):
                try:
                    pdf_name = f"{lot_slug}_plan.pdf"
                    export_pdf(pdf_name, xs, ys, meta_common)
                    with open(pdf_name,"rb") as f:
                        st.download_button("Download PDF", f, file_name=pdf_name, mime="application/pdf", key="dl_pdf")
                except Exception as e:
                    st.error(f"PDF export failed: {e}")

        with c2:
            if st.button("📐 Export DXF (R2000, Template if present)"):
                try:
                    dxf_name = f"{lot_slug}.dxf"
                    export_dxf_with_template(dxf_name, xs, ys, meta_common, template_path="BL-LOCA.dxf")
                    with open(dxf_name,"rb") as f:
                        st.download_button("Download DXF", f, file_name=dxf_name, mime="application/dxf", key="dl_dxf")
                except Exception as e:
                    st.error(f"DXF export failed: {e}")

        with c3:
            if st.button("🌍 Export KML (WGS84)"):
                try:
                    kml_name = f"{lot_slug}.kml"
                    export_kml(kml_name, xs, ys, epsg_current, meta_common)
                    with open(kml_name,"rb") as f:
                        st.download_button("Download KML", f, file_name=kml_name, mime="application/vnd.google-earth.kml+xml", key="dl_kml")
                except Exception as e:
                    st.error(f"KML export failed: {e}")
    except Exception:
        st.info("Enter inputs in the Input tab to enable exports.")
