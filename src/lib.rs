//! Land Cover Classification for RustGIS: training samples drawn on the map as points,
//! rectangles and polygons, a composite of band files of any cell size, supervised
//! classification (Random Forest, SVM, CART, k-NN, naive Bayes, minimum distance, maximum
//! likelihood) with a training/test split, and an accuracy assessment (confusion matrix, overall
//! accuracy, kappa, producer's and user's accuracy, F1).
//!
//! The heavy work runs in the plugin's tools (a tool has 60 times a command's budget). A tool
//! can't add a raster to the map, so it hands its GeoTIFF over in a one-feature layer, and the
//! `add-results` command (which the panel's Classify runs itself) turns it into a raster layer.
// Loops index several arrays by one counter (classes, bands), which reads more plainly than
// zipped iterators; and `chunks_exact` is kept over `as_chunks` for older compilers.
#![allow(clippy::needless_range_loop, clippy::chunks_exact_to_as_chunks)]

pub mod classify;
pub mod metrics;
pub mod ml;
pub mod proj;
pub mod sample;
pub mod stack;
pub mod tif;

use rustgis_plugin::{Kind, Request, ToolOutput, Value, geojson, host, json};
use sample::{ClassInfo, PALETTE, hex, parse_color};
use stack::{GridChoice, Resampling, Stack};

const ID: &str = "land-cover-classification";
pub const POINTS: &str = "Training Points";
pub const POLYGONS: &str = "Training Polygons";
const SKETCH: &str = "Polygon Sketch";
/// The field of a result layer holding a GeoTIFF in base64.
const PAYLOAD: &str = "lcc_tif";
/// Passes of a part-done classification one click of Add Results to Map runs.
const PASSES_PER_CLICK: usize = 3;

fn handle(mut req: Request) -> Result<Value, String> {
    match (req.kind, req.id.as_str()) {
        (Kind::Tool, "composite") => composite_tool(&mut req),
        (Kind::Tool, "classify") => classify::classify_tool(&mut req),
        (Kind::Tool, "assess") => assess_tool(&mut req),
        (Kind::Command, "panel-state") => panel_state(&req, None),
        (Kind::Command, "new-class") => new_class(&req),
        (Kind::Command, "choose-class") => choose_class(&req),
        (Kind::Command, "draw-points") => draw(&req, false),
        (Kind::Command, "draw-polygons") => draw(&req, true),
        (Kind::Command, "view-box") => view_box(&req),
        (Kind::Command, "add-point") => add_point(&req),
        (Kind::Command, "add-rectangle") => add_rectangle(&req),
        (Kind::Command, "add-vertex") => add_vertex(&req),
        (Kind::Command, "close-polygon") => close_polygon(&req),
        (Kind::Command, "run") => run(&req),
        (Kind::Command, "add-results") => add_results(&req, PASSES_PER_CLICK).map(|(report, _)| report),
        (_, other) => Err(format!("unknown '{other}'")),
    }
}

// The module's entry points: the SDK's, with large strings (GeoTIFFs and rasters in base64)
// spliced into the reply after it is written, so they are never held twice (in the answer and
// in its JSON text) within the module's 512 MB.

thread_local! {
    static SPLICES: std::cell::RefCell<Vec<(String, String)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A token standing for `data` in an answer; the reply gets `data` in its place.
pub(crate) fn splice(data: String) -> String {
    SPLICES.with(|s| {
        let mut s = s.borrow_mut();
        let token = format!("@@lcc-splice-{}@@", s.len());
        s.push((token.clone(), data));
        token
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rustgis_alloc(len: i32) -> i32 {
    rustgis_plugin::abi::alloc(len)
}

#[unsafe(no_mangle)]
pub extern "C" fn rustgis_call(ptr: i32, len: i32) -> i64 {
    SPLICES.with(|s| s.borrow_mut().clear());
    let packed = rustgis_plugin::abi::call(ptr, len, handle);
    let splices = SPLICES.with(|s| std::mem::take(&mut *s.borrow_mut()));
    if splices.is_empty() {
        return packed;
    }
    let (p, n) = ((packed as u64 >> 32) as usize, packed as u64 as u32 as usize);
    // SAFETY: `abi::call` answers a reply it leaked as a boxed slice of `n` bytes at `p`; it is
    // read here, then taken back and freed.
    let reply: Box<[u8]> = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(p as *mut u8, n)) };
    let extra: usize = splices.iter().map(|(_, d)| d.len()).sum();
    let mut out: Vec<u8> = Vec::with_capacity(reply.len() + extra);
    let mut at = 0;
    for (token, data) in splices {
        match find(&reply[at..], token.as_bytes()) {
            Some(i) => {
                out.extend_from_slice(&reply[at..at + i]);
                out.extend_from_slice(data.as_bytes());
                at += i + token.len();
            }
            None => continue,
        }
        drop(data);
    }
    out.extend_from_slice(&reply[at..]);
    drop(reply);
    let out = out.into_boxed_slice();
    let n = out.len() as i64;
    let p = Box::into_raw(out) as *mut u8 as i64;
    (p << 32) | n
}

// ---- Shared helpers --------------------------------------------------------------------------

/// Standard padded base64.
pub fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        out.push(T[(n >> 18) as usize & 63]);
        out.push(T[(n >> 12) as usize & 63]);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] } else { b'=' });
        out.push(if c.len() > 2 { T[n as usize & 63] } else { b'=' });
    }
    String::from_utf8(out).unwrap_or_default()
}

pub(crate) fn num(req: &Request, name: &str, default: f64) -> f64 {
    match req.params.get(name) {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(default),
        _ => default,
    }
}

pub(crate) fn text<'a>(req: &'a Request, name: &str) -> Option<&'a str> {
    req.str(name).map(str::trim).filter(|s| !s.is_empty())
}

/// A file parameter's contents, taken out of the request and decoded in place (the SDK's
/// decoder is general; this one is several times cheaper in RustGIS's metered interpreter).
fn take_file(req: &mut Request, param: &str) -> Result<Option<(String, Vec<u8>)>, String> {
    let Some(mut f) = req.files.remove(param) else { return Ok(None) };
    let name = f.get("name").and_then(Value::as_str).unwrap_or("file").to_string();
    let Some(Value::String(data)) = f.get_mut("data").map(Value::take) else {
        return Err(format!("{param}: the file has no contents"));
    };
    let bytes = decode_base64(data.into_bytes()).ok_or_else(|| format!("{name}: its contents aren't base64"))?;
    Ok(Some((name, bytes)))
}

/// Standard base64 (written without bounds checks in the inner loop: every instruction counts
/// in RustGIS's metered interpreter).
pub fn decode_base64(b: Vec<u8>) -> Option<Vec<u8>> {
    let mut t = [255u8; 256];
    for (i, c) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".iter().enumerate() {
        t[*c as usize] = i as u8;
    }
    let pad = b.iter().rev().take(2).take_while(|&&c| c == b'=').count();
    let len = b.len() - pad;
    let full = len / 4;
    let tail = len - full * 4;
    if tail == 1 {
        return None;
    }
    let mut out = vec![0u8; full * 3 + tail.saturating_sub(1)];
    let mut bad = 0u8;
    for (q, o) in b[..full * 4].as_chunks::<4>().0.iter().zip(out.chunks_exact_mut(3)) {
        let (a, c, d, e) = (t[q[0] as usize], t[q[1] as usize], t[q[2] as usize], t[q[3] as usize]);
        bad |= a | c | d | e;
        let n = (u32::from(a) << 18) | (u32::from(c) << 12) | (u32::from(d) << 6) | u32::from(e);
        o[0] = (n >> 16) as u8;
        o[1] = (n >> 8) as u8;
        o[2] = n as u8;
    }
    if tail > 0 {
        let mut n = 0u32;
        for (j, c) in b[full * 4..len].iter().enumerate() {
            let v = t[*c as usize];
            bad |= v;
            n |= u32::from(v) << (18 - 6 * j);
        }
        for j in 0..tail - 1 {
            out[full * 3 + j] = (n >> (16 - 8 * j)) as u8;
        }
    }
    drop(b);
    if bad & 0xC0 != 0 { None } else { Some(out) }
}

/// The rasters given in `raster1`, `raster2`, … read as images, in that order.
fn read_rasters(req: &mut Request, most: usize) -> Result<Vec<tif::Image>, String> {
    let mut images = Vec::new();
    for i in 1..=most {
        if let Some((name, bytes)) = take_file(req, &format!("raster{i}"))? {
            images.push(tif::read(&name, &bytes)?);
        }
    }
    if images.is_empty() {
        return Err("give at least one raster (a GeoTIFF of one or more bands)".into());
    }
    Ok(images)
}

pub(crate) fn pct(v: f64) -> String {
    if v.is_nan() { "–".into() } else { format!("{:.2}", v * 100.0) }
}

pub(crate) fn round(v: f64, places: i32) -> Value {
    if v.is_nan() {
        Value::Null
    } else {
        let p = 10f64.powi(places);
        json!((v * p).round() / p)
    }
}

/// A table: rows of `(field, value)` pairs, as a FeatureCollection without geometry.
pub(crate) fn table(rows: Vec<Vec<(String, Value)>>) -> Value {
    let features: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            let props: rustgis_plugin::Map<String, Value> = r.into_iter().collect();
            json!({"type": "Feature", "geometry": null, "properties": props})
        })
        .collect();
    geojson::collection(features)
}

/// A layer of one point holding a GeoTIFF for `add-results` to put on the map.
pub(crate) fn payload_b64(at: [f64; 2], name: &str, file: &str, kind: &str, data: String, report: &str) -> Value {
    geojson::collection(vec![geojson::feature(
        geojson::point(at[0], at[1]),
        json!({"lcc_name": name, "lcc_file": file, "lcc_kind": kind, "lcc_report": report, PAYLOAD: splice(data)}),
    )])
}

pub(crate) fn payload(at: [f64; 2], name: &str, file: &str, kind: &str, tif: &[u8], report: &str) -> Value {
    payload_b64(at, name, file, kind, base64(tif), report)
}

// ---- Composite Bands --------------------------------------------------------------------------

fn composite_tool(req: &mut Request) -> Result<Value, String> {
    let images = read_rasters(req, 24)?;
    let stack = stack::composite(images, GridChoice::parse(req.str("grid")), Resampling::parse(req.str("resampling")))?;
    let name = text(req, "name").unwrap_or("Composite").to_string();
    let at = centre_degrees(&stack).unwrap_or([0.0, 0.0]);
    let Stack { grid, geo, crs, bands, names, sample, notes } = stack;
    let n_bands = bands.len();
    let tif = tif::write_bands(&grid, &geo, bands, &names, sample, if sample == tif::Sample::F32 { -9999.0 } else { 0.0 })?;
    let order: Vec<String> = names.iter().enumerate().map(|(i, n)| format!("{}={n}", i + 1)).collect();
    let mut out = ToolOutput::new();
    for n in &notes {
        out = out.message(n.clone());
    }
    let report = format!(
        "{name}: {n_bands} bands, {} × {} cells of {}, {}. Band order: {}",
        grid.w,
        grid.h,
        stack::cell(&grid, &crs),
        crs.describe(),
        order.join(", ")
    );
    out = out.message(report.clone());
    out = out.message(format!("Run Land Cover › Add Results to Map to put the composite on the map ({} MB).", tif.len() / 1_000_000));
    // Memory: the module has 512 MB, and the GeoTIFF goes out again as base64.
    let data = base64(&tif);
    drop(tif);
    Ok(out
        .layer(&format!("{name} (raster data)"), payload_b64(at, &name, &format!("{}.tif", file_stem(&name)), "composite", data, &report))
        .into())
}

pub(crate) fn file_stem(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    if s.is_empty() { "result".into() } else { s }
}

/// Somewhere inside the raster, in degrees, for the result layer's point: found by searching the
/// forward projection (there is no inverse here), good to well under a cell.
pub(crate) fn centre_degrees(stack: &Stack) -> Option<[f64; 2]> {
    let (cx, cy) = stack.grid.center(stack.grid.w / 2, stack.grid.h / 2);
    if !stack.crs.metric() {
        return Some([cx, cy]);
    }
    let (mut lon, mut lat) = match stack.crs {
        proj::Crs::Utm { zone, south } => (-183.0 + 6.0 * zone as f64, if south { -10.0 } else { 10.0 }),
        proj::Crs::Tm { lon0, .. } => (lon0, 0.0),
        _ => (0.0, 0.0),
    };
    for _ in 0..50 {
        let (x, y) = stack.crs.forward(lon, lat);
        let (x1, _) = stack.crs.forward(lon + 1e-4, lat);
        let (_, y1) = stack.crs.forward(lon, lat + 1e-4);
        let (dxl, dyl) = ((x1 - x) / 1e-4, (y1 - y) / 1e-4);
        if dxl.abs() < 1e-9 || dyl.abs() < 1e-9 {
            return None;
        }
        lon -= (x - cx) / dxl;
        lat = (lat - (y - cy) / dyl).clamp(-89.0, 89.0);
    }
    Some([lon, lat])
}

// ---- Accuracy assessment of a classified map --------------------------------------------------

fn assess_tool(req: &mut Request) -> Result<Value, String> {
    let (fname, bytes) = take_file(req, "classified")?.ok_or("give the classified raster")?;
    let mut img = tif::read(&fname, &bytes)?;
    drop(bytes);
    img.bands.truncate(1);
    let mut labels: std::collections::BTreeMap<u32, String> = img.class_names.iter().cloned().collect();
    let stack = stack::composite(vec![img], GridChoice::First, Resampling::Nearest)?;
    text(req, "reference").ok_or("give the reference samples (a point or polygon layer)")?;
    let class_field = text(req, "classField").unwrap_or("class").to_string();
    let max_pixels = num(req, "maxPixels", 500.0).max(1.0) as usize;
    let mut refs_cells: Vec<(u32, u32)> = Vec::new();
    let mut skipped = 0;
    let mut rng = ml::Rng::new(7);
    for feat in req.features("reference")? {
        let v = match &feat["properties"][&class_field] {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.trim().parse::<f64>().ok(),
            _ => None,
        };
        let Some(v) = v.filter(|v| v.fract() == 0.0 && *v >= 0.0) else {
            skipped += 1;
            continue;
        };
        let v = v as u32;
        if let Some(l) = feat["properties"].get("label").and_then(Value::as_str) {
            labels.entry(v).or_insert_with(|| l.to_string());
        }
        let mut cells = Vec::new();
        sample::cells(&stack, &feat["geometry"], &mut cells);
        cells.sort_unstable();
        cells.dedup();
        if cells.len() > max_pixels {
            rng.shuffle(&mut cells);
            cells.truncate(max_pixels);
        }
        refs_cells.extend(cells.into_iter().map(|c| (c, v)));
    }
    let mut pairs = Vec::new();
    let mut nodata = 0;
    for (cell, v) in refs_cells {
        let p = stack.bands[0][cell as usize];
        if p.is_nan() || p == 0.0 {
            nodata += 1;
            continue;
        }
        pairs.push((v, p as u32));
    }
    if pairs.is_empty() {
        return Err("none of the reference samples falls on classified cells".into());
    }
    let mut values: Vec<u32> = pairs.iter().flat_map(|(a, b)| [*a, *b]).collect();
    values.sort_unstable();
    values.dedup();
    let k = values.len();
    let mut cm = metrics::Confusion::new(k);
    for (a, p) in &pairs {
        cm.add(values.binary_search(a).unwrap_or(0), values.binary_search(p).unwrap_or(0));
    }
    let classes: Vec<ClassInfo> = values
        .iter()
        .map(|v| ClassInfo {
            value: (*v).min(255) as u8,
            label: labels.get(v).cloned().unwrap_or_else(|| format!("Class {v}")),
            color: PALETTE[*v as usize % PALETTE.len()],
        })
        .collect();
    let name = text(req, "name").unwrap_or("Accuracy Assessment").to_string();
    let mut out = ToolOutput::new();
    if skipped > 0 {
        out = out.message(format!("{skipped} reference sample(s) with no whole-number class in '{class_field}' left out"));
    }
    if nodata > 0 {
        out = out.message(format!("{nodata} reference pixel(s) over no-data left out"));
    }
    let rows = vec![
        vec![("Metric".into(), json!("Reference pixels")), ("Value".into(), json!(cm.total().to_string()))],
        vec![("Metric".into(), json!("Overall accuracy")), ("Value".into(), json!(format!("{}%", pct(cm.overall()))))],
        vec![("Metric".into(), json!("Kappa")), ("Value".into(), json!(format!("{:.4}", cm.kappa())))],
    ];
    out = out.table(&format!("{name} – Accuracy"), table(rows));
    out = out.table(&format!("{name} – Confusion Matrix"), classify::confusion_table(&cm, &classes));
    let per: Vec<Vec<(String, Value)>> = classes
        .iter()
        .enumerate()
        .map(|(c, info)| {
            vec![
                ("Value".into(), json!(values[c])),
                ("Class".into(), json!(info.label)),
                ("Reference pixels".into(), json!(cm.row(c))),
                ("Producer's accuracy (%)".into(), round(cm.producers(c) * 100.0, 2)),
                ("User's accuracy (%)".into(), round(cm.users(c) * 100.0, 2)),
                ("F1 score".into(), round(cm.f1(c), 4)),
            ]
        })
        .collect();
    out = out.table(&format!("{name} – Classes"), table(per));
    out = out.message(format!("Overall accuracy {}%, kappa {:.4}, from {} reference pixels", pct(cm.overall()), cm.kappa(), cm.total()));
    Ok(out.into())
}

// ---- Panel: classes and samples ---------------------------------------------------------------

/// The classes a new project starts with. The panel for RustGIS 0.6.0, which has no dialog to
/// name a new class, offers more.
fn default_classes(more: bool) -> Vec<ClassInfo> {
    let base: &[(&str, [u8; 3])] =
        &[("Water", [31, 120, 180]), ("Vegetation", [51, 160, 44]), ("Built-up", [227, 26, 28]), ("Bare land", [253, 191, 111])];
    let extra: &[(&str, [u8; 3])] = &[
        ("Mangrove", [0, 109, 44]),
        ("Cropland", [230, 171, 2]),
        ("Rice field", [166, 217, 106]),
        ("Plantation", [102, 166, 30]),
        ("Shrubland", [178, 223, 138]),
        ("Grassland", [217, 240, 163]),
        ("Wetland", [128, 177, 211]),
        ("Cloud", [240, 240, 240]),
        ("Shadow", [64, 64, 64]),
    ];
    base.iter()
        .chain(if more { extra.iter() } else { [].iter() })
        .enumerate()
        .map(|(i, (l, c))| ClassInfo { value: (i + 1) as u8, label: l.to_string(), color: *c })
        .collect()
}

/// Whether the request comes from the panel for RustGIS 0.6.0 (trees, sliders and checkboxes
/// instead of dropdowns, dialogs and picks; samples drawn with RustGIS's own Create tool).
fn compat(req: &Request) -> bool {
    req.panel.contains_key("drawing")
}

/// A class as a dropdown option's id: `value|label|#rrggbb`, so the option chosen says all of it.
fn class_id(c: &ClassInfo) -> String {
    format!("{}|{}|{}", c.value, c.label, hex(c.color))
}

fn class_from_id(id: &str) -> Option<ClassInfo> {
    let mut parts = id.splitn(3, '|');
    let value = parts.next()?.trim().parse::<u8>().ok().filter(|v| *v > 0)?;
    let label = parts.next()?.to_string();
    let color = parts.next().and_then(parse_color).unwrap_or(PALETTE[(value as usize + PALETTE.len() - 1) % PALETTE.len()]);
    Some(ClassInfo { value, label, color })
}

fn layer_names() -> Result<Vec<(String, bool)>, String> {
    let layers = host::layers()?;
    Ok(layers
        .as_array()
        .into_iter()
        .flatten()
        .map(|l| (l["name"].as_str().unwrap_or_default().to_string(), l["raster"].as_bool().unwrap_or(false)))
        .collect())
}

fn has_layer(layers: &[(String, bool)], name: &str) -> bool {
    layers.iter().any(|l| l.0 == name)
}

/// The training samples' classes, and how many points and polygons each has.
fn layer_classes(layers: &[(String, bool)]) -> Vec<(ClassInfo, usize, usize)> {
    let mut out: Vec<(ClassInfo, usize, usize)> = Vec::new();
    for (layer, polygons) in [(POINTS, false), (POLYGONS, true)] {
        if !has_layer(layers, layer) {
            continue;
        }
        let Ok(fc) = host::features(layer, false) else { continue };
        for f in fc["features"].as_array().into_iter().flatten() {
            let p = &f["properties"];
            let Some(v) = p["class"].as_f64().filter(|v| (1.0..=255.0).contains(v)).map(|v| v as u8) else { continue };
            let i = match out.iter().position(|c| c.0.value == v) {
                Some(i) => i,
                None => {
                    let label = p["label"].as_str().filter(|s| !s.is_empty()).map(str::to_string).unwrap_or_else(|| format!("Class {v}"));
                    let color =
                        p["color"].as_str().and_then(parse_color).unwrap_or(PALETTE[(v as usize + PALETTE.len() - 1) % PALETTE.len()]);
                    out.push((ClassInfo { value: v, label, color }, 0, 0));
                    out.len() - 1
                }
            };
            if polygons {
                out[i].2 += 1;
            } else {
                out[i].1 += 1;
            }
        }
    }
    out.sort_by_key(|c| c.0.value);
    out
}

/// The classes the panel offers: those of the samples, the starting classes whose numbers
/// they don't use, and the one chosen (a new class, until it has samples).
fn panel_classes(req: &Request, sampled: &[(ClassInfo, usize, usize)]) -> Vec<ClassInfo> {
    let mut classes: Vec<ClassInfo> = sampled.iter().map(|c| c.0.clone()).collect();
    for d in default_classes(compat(req)) {
        if !classes.iter().any(|c| c.value == d.value || c.label.eq_ignore_ascii_case(&d.label)) {
            classes.push(d);
        }
    }
    if let Some(c) = req.panel.get("class").and_then(Value::as_str).and_then(class_from_id)
        && !classes.iter().any(|k| k.value == c.value)
    {
        classes.push(c);
    }
    classes.sort_by_key(|c| c.value);
    classes
}

fn active_class(req: &Request) -> Result<ClassInfo, String> {
    if let Some(c) = req.panel.get("class").and_then(Value::as_str).and_then(class_from_id) {
        return Ok(c);
    }
    let layers = layer_names()?;
    panel_classes(req, &layer_classes(&layers)).into_iter().next().ok_or_else(|| "add a class first (New Class)".into())
}

fn sample_summary(sampled: &[(ClassInfo, usize, usize)], old: bool) -> String {
    if sampled.is_empty() {
        return if old {
            "No samples yet. Choose a class, then Draw Points or Draw Polygons and draw on the map; choose the next class when done.".into()
        } else {
            "No samples yet. Choose a class, then Point, Rectangle or Polygon Vertex, and click the map.".into()
        };
    }
    sampled.iter().map(|(c, p, g)| format!("{} {}: {p} point(s), {g} polygon(s)", c.value, c.label)).collect::<Vec<_>>().join("\n")
}

/// The panel value of a widget as a number.
fn pv(req: &Request, id: &str) -> Option<f64> {
    match req.panel.get(id)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn ps<'a>(req: &'a Request, id: &str) -> Option<&'a str> {
    req.panel.get(id).and_then(Value::as_str)
}

/// Which classifier settings apply to which algorithm.
fn setting_visibility(algorithm: classify::Algorithm, old: bool) -> Vec<(&'static str, bool)> {
    use classify::Algorithm as A;
    let forest = matches!(algorithm, A::RandomForest | A::Cart);
    let svm = algorithm == A::Svm;
    let mut v = vec![
        ("trees", algorithm == A::RandomForest),
        ("variables", algorithm == A::RandomForest),
        ("min-leaf", forest),
        ("max-depth", forest),
        ("neighbours", algorithm == A::Knn),
    ];
    if old {
        v.extend([("linear", svm), ("cost-step", svm), ("gamma-step", svm)]);
    } else {
        v.extend([("kernel", svm), ("cost", svm), ("gamma", svm)]);
    }
    v
}

fn panel_state(req: &Request, report: Option<&str>) -> Result<Value, String> {
    // Samples drawn with RustGIS's own tool since the last call get the class being drawn.
    let mut note = None;
    if compat(req) {
        let n = assign_pending(req)?;
        if n > 0 {
            note = Some(n);
        }
    }
    let layers = layer_names()?;
    let sampled = layer_classes(&layers);
    let classes = panel_classes(req, &sampled);
    let old = compat(req);
    let count = |c: &ClassInfo| sampled.iter().find(|s| s.0.value == c.value).map_or(0, |s| s.1 + s.2);
    let options: Vec<Value> = classes
        .iter()
        .map(|c| {
            if old {
                json!({"id": class_id(c), "label": format!("{} {}", c.value, c.label), "count": count(c)})
            } else {
                json!({"id": class_id(c), "label": format!("{} {}", c.value, c.label)})
            }
        })
        .collect();
    let mut images: Vec<Value> = vec![json!({"id": "*", "label": "All visible raster layers"})];
    images.extend(layers.iter().filter(|l| l.1).map(|l| json!({"id": l.0, "label": l.0})));
    let list = if old { "nodes" } else { "options" };
    let mut items = json!({
        "class": {list: options},
        "samples": {"text": sample_summary(&sampled, old)},
        "image": {list: images},
    });
    let algorithm = classify::Algorithm::parse(ps(req, "algorithm").unwrap_or("Random Forest"));
    for (id, visible) in setting_visibility(algorithm, old) {
        items[id] = json!({"visible": visible});
    }
    if old {
        let names = [
            "Random Forest",
            "Support Vector Machine",
            "CART (Decision Tree)",
            "k-Nearest Neighbors",
            "Naive Bayes",
            "Minimum Distance",
            "Maximum Likelihood",
        ];
        items["algorithm"] = json!({"nodes": names.iter().map(|n| json!({"id": n})).collect::<Vec<_>>()});
        let labels = |v: &[f64]| v.iter().map(|x| if *x == 0.0 { "auto".to_string() } else { x.to_string() }).collect::<Vec<_>>();
        items["grid-step"] = json!({"labels": ["Finest raster", "Coarsest raster", "First raster"]});
        items["cost-step"] = json!({"labels": labels(&COSTS)});
        items["gamma-step"] = json!({"labels": labels(&GAMMAS)});
        items["max-pixels-step"] = json!({"labels": labels(&MAX_PIXELS)});
        items["drawing"] = json!({"visible": false});
    }
    if let Some(r) = report {
        items["report"] = json!({"text": r});
    }
    if let Some(n) = note {
        host::notify(&format!("{n} new sample(s) given their class"));
    }
    let chosen = req.panel.get("class").and_then(Value::as_str).filter(|id| classes.iter().any(|c| class_id(c) == *id)).map(str::to_string);
    let chosen = chosen.unwrap_or_else(|| class_id(&classes[0]));
    let mut values = json!({"class": chosen});
    if old {
        values["drawing"] = json!(class_from_id(&chosen).map_or(0, |c| c.value));
        if req.panel.get("algorithm").is_none_or(Value::is_null) {
            values["algorithm"] = json!("Random Forest");
        }
    }
    Ok(json!({"panel": {"items": items, "values": values}}))
}

/// Styles the training layers by class, with the classes' colours, and gives each a feature
/// template per class (Edit › Create then draws samples of that class).
fn style_samples(layers: &[(String, bool)], extra: Option<&ClassInfo>) {
    let mut classes: Vec<ClassInfo> = layer_classes(layers).into_iter().map(|c| c.0).collect();
    if let Some(e) = extra
        && !classes.iter().any(|c| c.value == e.value)
    {
        classes.push(e.clone());
    }
    for layer in [POINTS, POLYGONS] {
        if !has_layer(layers, layer) {
            continue;
        }
        if host::call("layer.symbology", json!({"layer": layer, "type": "unique", "field": "label"})).is_err() {
            continue;
        }
        for c in &classes {
            let _ = host::call(
                "layer.classStyle",
                json!({"layer": layer, "value": c.label, "color": hex(c.color), "label": format!("{} {}", c.value, c.label)}),
            );
            let _ = host::call(
                "edit.template",
                json!({"layer": layer, "name": c.label, "defaults": {"class": c.value, "label": c.label, "color": hex(c.color)}}),
            );
        }
    }
}

/// New Class: a class named in the command's dialog, numbered after the others.
fn new_class(req: &Request) -> Result<Value, String> {
    let name = text(req, "name").ok_or("give the class a name")?.replace('|', "/");
    let layers = layer_names()?;
    let sampled = layer_classes(&layers);
    let classes = panel_classes(req, &sampled);
    if classes.iter().any(|c| c.label.eq_ignore_ascii_case(&name)) {
        return Err(format!("there is a class '{name}' already"));
    }
    let value = classes.iter().map(|c| c.value).max().unwrap_or(0).checked_add(1).ok_or("at most 255 classes")?;
    let color = text(req, "color").and_then(parse_color).unwrap_or(PALETTE[(value as usize + PALETTE.len() - 1) % PALETTE.len()]);
    let c = ClassInfo { value, label: name, color };
    style_samples(&layers, Some(&c));
    let mut state = panel_state(req, None)?;
    let mut options = state["panel"]["items"]["class"]["options"].as_array().cloned().unwrap_or_default();
    options.push(json!({"id": class_id(&c), "label": format!("{} {}", c.value, c.label)}));
    state["panel"]["items"]["class"]["options"] = Value::Array(options);
    state["panel"]["values"]["class"] = json!(class_id(&c));
    state["report"] = json!(format!("Class {} {} ({}) added: draw its samples now", c.value, c.label, hex(c.color)));
    Ok(state)
}

/// Adds a training sample of the chosen class to `layer`, creating the layer if need be.
fn add_sample(req: &Request, layer: &str, geometry: Value) -> Result<Value, String> {
    let c = active_class(req)?;
    let props = json!({"class": c.value, "label": c.label, "color": hex(c.color)});
    let layers = layer_names()?;
    if has_layer(&layers, layer) {
        let known = layer_classes(&layers).iter().any(|k| k.0.value == c.value);
        host::call("edit.addFeature", json!({"layer": layer, "geometry": geometry, "attributes": props}))?;
        // A class new to the samples gets its colour (Unique Values are worked out again).
        if !known {
            style_samples(&layers, Some(&c));
        }
    } else {
        host::add_layer(layer, geojson::collection(vec![geojson::feature(geometry, props)]))?;
        style_samples(&layer_names()?, Some(&c));
    }
    panel_state(req, None)
}

fn place(req: &Request, id: &str) -> Result<Value, String> {
    req.panel.get(id).filter(|v| !v.is_null()).cloned().ok_or_else(|| "click the button, then the map".to_string())
}

fn add_point(req: &Request) -> Result<Value, String> {
    let p = place(req, "point")?;
    let (lon, lat) = (p["lon"].as_f64().ok_or("no longitude")?, p["lat"].as_f64().ok_or("no latitude")?);
    add_sample(req, POINTS, geojson::point(lon, lat))
}

fn add_rectangle(req: &Request) -> Result<Value, String> {
    let p = place(req, "box")?;
    let b: Vec<f64> = p["bbox"].as_array().ok_or("no box")?.iter().filter_map(Value::as_f64).collect();
    if b.len() != 4 {
        return Err("the box isn't four numbers".into());
    }
    let (w, s, e, n) = (b[0], b[1], if b[2] < b[0] { b[2] + 360.0 } else { b[2] }, b[3]);
    add_sample(req, POLYGONS, geojson::polygon(&[[w, s], [e, s], [e, n], [w, n], [w, s]]))
}

fn add_vertex(req: &Request) -> Result<Value, String> {
    let p = place(req, "vertex")?;
    let (lon, lat) = (p["lon"].as_f64().ok_or("no longitude")?, p["lat"].as_f64().ok_or("no latitude")?);
    let layers = layer_names()?;
    let n = if has_layer(&layers, SKETCH) { host::features(SKETCH, false)?["features"].as_array().map_or(0, Vec::len) } else { 0 };
    let props = json!({"vertex": n + 1});
    if has_layer(&layers, SKETCH) {
        host::call("edit.addFeature", json!({"layer": SKETCH, "geometry": geojson::point(lon, lat), "attributes": props}))?;
    } else {
        host::add_layer(SKETCH, geojson::collection(vec![geojson::feature(geojson::point(lon, lat), props)]))?;
    }
    let mut state = panel_state(req, None)?;
    state["panel"]["items"]["close-polygon"] = json!({"label": format!("Close Polygon ({} corners)", n + 1)});
    Ok(state)
}

fn close_polygon(req: &Request) -> Result<Value, String> {
    let layers = layer_names()?;
    if !has_layer(&layers, SKETCH) {
        return Err("add the polygon's corners first: click Polygon Vertex, then the map, for each corner".into());
    }
    let fc = host::features(SKETCH, false)?;
    let mut pts: Vec<(i64, [f64; 2])> = fc["features"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|f| {
            let c = f["geometry"]["coordinates"].as_array()?;
            Some((f["properties"]["vertex"].as_i64().unwrap_or(0), [c.first()?.as_f64()?, c.get(1)?.as_f64()?]))
        })
        .collect();
    pts.sort_by_key(|p| p.0);
    if pts.len() < 3 {
        return Err(format!("a polygon needs at least 3 corners ({} so far)", pts.len()));
    }
    let mut ring: Vec<[f64; 2]> = pts.iter().map(|p| p.1).collect();
    ring.push(ring[0]);
    host::call("layer.remove", json!({"layer": SKETCH}))?;
    let mut state = add_sample(req, POLYGONS, geojson::polygon(&ring))?;
    state["panel"]["items"]["close-polygon"] = json!({"label": null});
    state["panel"]["values"]["vertex"] = Value::Null;
    Ok(state)
}

// ---- Running and results ----------------------------------------------------------------------

/// The Classify tool's parameters from the panel's settings.
fn settings(req: &Request) -> Value {
    let mut p = json!({
        "algorithm": ps(req, "algorithm").unwrap_or("Random Forest"),
        "splitRatio": pv(req, "split").unwrap_or(70.0) / 100.0,
        "splitBy": ps(req, "split-by").unwrap_or("Pixel"),
        "grid": ps(req, "grid").unwrap_or("Finest raster"),
        "classField": "class",
    });
    for (id, param) in [
        ("trees", "trees"),
        ("variables", "variablesPerSplit"),
        ("min-leaf", "minLeaf"),
        ("max-depth", "maxDepth"),
        ("cost", "svmCost"),
        ("gamma", "svmGamma"),
        ("neighbours", "k"),
        ("max-pixels", "maxPixels"),
        ("seed", "seed"),
    ] {
        if let Some(v) = pv(req, id) {
            p[param] = json!(v);
        }
    }
    if let Some(k) = ps(req, "kernel") {
        p["svmKernel"] = json!(k);
    }
    // The panel for RustGIS 0.6.0: steps of sliders and checkboxes.
    let step = |id: &str, list: &[f64]| pv(req, id).map(|i| list[(i.max(0.0) as usize).min(list.len() - 1)]);
    if let Some(i) = pv(req, "grid-step") {
        p["grid"] = json!(["Finest raster", "Coarsest raster", "First raster"][(i.max(0.0) as usize).min(2)]);
    }
    if req.panel_bool("split-by-sample") == Some(true) {
        p["splitBy"] = json!("Sample (whole polygon)");
    }
    if req.panel_bool("linear") == Some(true) {
        p["svmKernel"] = json!("Linear");
    }
    if let Some(v) = step("cost-step", &COSTS) {
        p["svmCost"] = json!(v);
    }
    if let Some(v) = step("gamma-step", &GAMMAS) {
        p["svmGamma"] = json!(v);
    }
    if let Some(v) = step("max-pixels-step", &MAX_PIXELS) {
        p["maxPixels"] = json!(v);
    }
    p
}

/// The steps of the 0.6.0 panel's SVM cost, gamma and max-pixels sliders.
pub const COSTS: [f64; 5] = [0.1, 1.0, 10.0, 100.0, 1000.0];
pub const GAMMAS: [f64; 7] = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0];
pub const MAX_PIXELS: [f64; 6] = [100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0];

// ---- Drawing with RustGIS's own tools (the panel for RustGIS 0.6.0) ---------------------------

/// Gives the samples drawn with RustGIS's Create tool (which leave `class` empty) the class
/// being drawn. Answers how many.
fn assign_pending(req: &Request) -> Result<usize, String> {
    let v = pv(req, "drawing").unwrap_or(0.0) as i64;
    if !(1..=255).contains(&v) {
        return Ok(0);
    }
    let layers = layer_names()?;
    let classes = panel_classes(req, &layer_classes(&layers));
    let Some(c) = classes.into_iter().find(|c| i64::from(c.value) == v) else { return Ok(0) };
    let mut n = 0;
    for layer in [POINTS, POLYGONS] {
        if !has_layer(&layers, layer) {
            continue;
        }
        let fc = host::features(layer, false)?;
        for f in fc["features"].as_array().into_iter().flatten() {
            if f["properties"]["class"].as_f64().is_some() {
                continue;
            }
            let Some(row) = f["row"].as_u64() else { continue };
            for (field, value) in [("class", json!(c.value)), ("label", json!(c.label)), ("color", json!(hex(c.color)))] {
                host::call("edit.setValue", json!({"layer": layer, "row": row, "field": field, "value": value}))?;
            }
            n += 1;
        }
    }
    if n > 0 {
        style_samples(&layer_names()?, Some(&c));
    }
    Ok(n)
}

/// A class chosen: what was drawn so far gets the class before it, and drawing goes on in this one.
fn choose_class(req: &Request) -> Result<Value, String> {
    // `drawing` still holds the class drawn until now; the state answers the new one.
    panel_state(req, None)
}

/// Makes `layer` (points or polygons) if it is missing: from a stand-in feature, removed at once,
/// so the layer has its fields and geometry type for the Create tool.
fn ensure_layer(layer: &str, polygons: bool) -> Result<(), String> {
    if has_layer(&layer_names()?, layer) {
        return Ok(());
    }
    let v = host::view()?;
    let (lon, lat) = (v["center"][0].as_f64().unwrap_or(0.0), v["center"][1].as_f64().unwrap_or(0.0));
    let g = if polygons {
        let d = 1e-5;
        geojson::polygon(&[[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]])
    } else {
        geojson::point(lon, lat)
    };
    host::add_layer(layer, geojson::collection(vec![geojson::feature(g, json!({"class": 0, "label": "", "color": ""}))]))?;
    let _ = host::call("select.clear", json!({}));
    host::call("select.byAttributes", json!({"layer": layer, "where": "class = 0"}))?;
    host::call("edit.delete", json!({}))?;
    Ok(())
}

/// Draw Points / Draw Polygons: the sample layer made active and RustGIS's Create tool on it.
fn draw(req: &Request, polygons: bool) -> Result<Value, String> {
    let layer = if polygons { POLYGONS } else { POINTS };
    let mut state = panel_state(req, None)?;
    ensure_layer(layer, polygons)?;
    style_samples(&layer_names()?, None);
    host::call("layer.select", json!({"layer": layer}))?;
    host::call("tool.set", json!({"tool": if polygons { "createPolygon" } else { "createPoint" }}))?;
    let c = active_class(req)?;
    let how = if polygons { "click each corner, double-click to finish a polygon" } else { "click the map for each point" };
    host::notify(&format!("Drawing {} samples: {how}. Choose another class in the panel when done.", c.label));
    state["report"] = json!(format!("Drawing {} into {layer}: {how}", c.label));
    Ok(state)
}

/// View as Rectangle: a rectangle sample of the chosen class covering the map's view.
fn view_box(req: &Request) -> Result<Value, String> {
    let e = host::call("map.extent", json!({}))?;
    let b: Vec<f64> = e["extent"].as_array().ok_or("the view has no extent")?.iter().filter_map(Value::as_f64).collect();
    if b.len() != 4 {
        return Err("the view has no extent".into());
    }
    let km = (b[2] - b[0]) * 111.32 * ((b[1] + b[3]) / 2.0).to_radians().cos();
    if km > 5.0 {
        return Err(format!("the view is {km:.0} km wide: zoom in to the patch you want as a sample (under 5 km)"));
    }
    let _ = assign_pending(req)?;
    add_sample(req, POLYGONS, geojson::polygon(&[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]))
}

/// The panel's Classify: on the desktop, runs the Classify tool on the raster layers chosen (by
/// their files) and the training layers, then puts the result on the map.
fn run(req: &Request) -> Result<Value, String> {
    if compat(req) {
        assign_pending(req)?;
    }
    let layers = layer_names()?;
    let image = ps(req, "image").unwrap_or("*");
    let chosen: Vec<String> = if image == "*" {
        let all = host::layers()?;
        let mut v: Vec<String> = all
            .as_array()
            .into_iter()
            .flatten()
            .filter(|l| l["raster"].as_bool() == Some(true) && l["visible"].as_bool() != Some(false))
            .filter_map(|l| l["name"].as_str().map(str::to_string))
            .collect();
        // The Layers pane lists the top first; band files are usually added B2, B3, … so the
        // bottom one first keeps that order.
        v.reverse();
        v
    } else {
        vec![image.to_string()]
    };
    if chosen.is_empty() {
        return Err("add the band rasters to the map (or run Land Cover › Classify and pick the files)".into());
    }
    if chosen.len() > 12 {
        return Err(format!("{} rasters are visible: show at most 12, or stack them first with Composite Bands", chosen.len()));
    }
    let mut params = settings(req);
    for (i, name) in chosen.iter().enumerate() {
        let path = host::call("layer.copyPath", json!({"layer": name}))
            .ok()
            .and_then(|v| v["text"].as_str().map(str::to_string))
            .filter(|p| !p.contains("://"))
            .ok_or_else(|| format!("{name} isn't read from a file on this computer. In the web app, run Land Cover › Classify and pick the band files there, then Add Results to Map"))?;
        params[format!("raster{}", i + 1)] = json!(path);
    }
    if has_layer(&layers, POINTS) {
        params["trainingPoints"] = json!(POINTS);
    }
    if has_layer(&layers, POLYGONS) {
        params["trainingPolygons"] = json!(POLYGONS);
    }
    if params.get("trainingPoints").is_none() && params.get("trainingPolygons").is_none() {
        return Err("draw training samples first (Point, Rectangle or Polygon Vertex)".into());
    }
    host::call("gp.run", json!({"tool": format!("plugin:{ID}/classify"), "params": params}))?;
    // The first run took up to half a minute: fewer passes follow it within the desktop app's
    // two minutes for a command.
    add_results(req, PASSES_PER_CLICK - 1).map(|(v, _)| v)
}

/// Raw host calls, for GeoTIFFs too large to parse as JSON within a command's budget.
#[cfg(target_arch = "wasm32")]
mod raw {
    #[link(wasm_import_module = "rustgis")]
    unsafe extern "C" {
        #[link_name = "call"]
        fn host_call(ptr: *const u8, len: usize) -> i32;
        #[link_name = "result"]
        fn host_result(ptr: *mut u8);
    }

    /// Sends a request (`{"op", "params"}` as bytes) and returns the reply's bytes.
    pub fn call(req: &[u8]) -> Result<Vec<u8>, String> {
        // SAFETY: the host reads `len` bytes at `ptr`, which `req` owns for the call.
        let n = unsafe { host_call(req.as_ptr(), req.len()) };
        let len = usize::try_from(n).map_err(|_| String::from("RustGIS didn't answer"))?;
        let mut buf: Vec<u8> = Vec::with_capacity(len);
        // SAFETY: the host writes exactly `len` bytes into `buf`'s capacity.
        unsafe {
            host_result(buf.as_mut_ptr());
            buf.set_len(len);
        }
        if buf.starts_with(b"{\"error\"") {
            let v: serde_json::Value = serde_json::from_slice(&buf).map_err(|e| e.to_string())?;
            return Err(v["error"].as_str().unwrap_or("failed").to_string());
        }
        Ok(buf)
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod raw {
    pub fn call(_req: &[u8]) -> Result<Vec<u8>, String> {
        Err("host calls only work inside RustGIS".into())
    }
}

/// The string value of `"key":` in `json` (a slice of it, still escaped), from `from`.
fn string_at<'a>(json: &'a [u8], key: &str, from: usize) -> Option<(&'a [u8], usize)> {
    let pat = format!("\"{key}\"");
    let at = from + find(&json[from..], pat.as_bytes())?;
    let mut i = at + pat.len();
    while i < json.len() && (json[i] == b' ' || json[i] == b':' || json[i] == b'\n' || json[i] == b'\t' || json[i] == b'\r') {
        i += 1;
    }
    if json.get(i) != Some(&b'"') {
        return None;
    }
    let start = i + 1;
    let mut j = start;
    while j < json.len() {
        match json[j] {
            b'\\' => j += 2,
            b'"' => return Some((&json[start..j], j)),
            _ => j += 1,
        }
    }
    None
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let first = *needle.first()?;
    let mut i = 0;
    while i + needle.len() <= hay.len() {
        let p = hay[i..hay.len() - needle.len() + 1].iter().position(|&b| b == first)?;
        i += p;
        if &hay[i..i + needle.len()] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn unescape(s: &[u8]) -> String {
    let mut q = Vec::with_capacity(s.len() + 2);
    q.push(b'"');
    q.extend_from_slice(s);
    q.push(b'"');
    serde_json::from_slice::<String>(&q).unwrap_or_default()
}

/// Puts every result the tools left (layers with a GeoTIFF) on the map as a raster layer, and
/// removes the layer that carried it. Answers the panel's report.
fn add_results(req: &Request, passes: usize) -> Result<(Value, usize), String> {
    // Classifications left part-done: run their next passes (a few per click: the desktop app
    // stops a command after two minutes, and a pass takes up to half a minute).
    let mut progress = String::new();
    for _ in 0..passes {
        let jobs = layers_with(classify::JOB)?;
        let Some(job) = jobs.first() else { break };
        // Raster 1 is required in the tool's form; a later pass reads the job's copies instead,
        // so it gets a stand-in of three zero bytes.
        let params = json!({"job": job, "raster1": {"name": "job.tif", "data": "AAAA"}});
        host::call("gp.run", json!({"tool": format!("plugin:{ID}/classify"), "params": params}))?;
        host::call("layer.remove", json!({"layer": job}))?;
        progress.clear();
        if let Some(next) = layers_with(classify::JOB)?.first() {
            progress = next.rsplit_once("(classifying, ").map(|(_, p)| p.trim_end_matches(')').to_string()).unwrap_or_default();
        }
    }
    let carriers = layers_with(PAYLOAD)?;
    if carriers.is_empty() {
        if !layers_with(classify::JOB)?.is_empty() {
            let mut state = panel_state(req, Some(&format!("Still classifying: {progress}. Click Add Results to Map again to go on.")))?;
            state["report"] = json!(format!("Still classifying ({progress}): click Add Results to Map again to go on"));
            return Ok((state, 0));
        }
        return Err("no results to add: run Classify or Composite Bands first".into());
    }
    let mut reports = Vec::new();
    let mut added = Vec::new();
    for carrier in &carriers {
        let mut q = br#"{"op":"host.features","params":{"layer":"#.to_vec();
        q.extend_from_slice(&serde_json::to_vec(carrier).unwrap_or_default());
        q.extend_from_slice(b"}}");
        let reply = raw::call(&q)?;
        let (data, _) = string_at(&reply, PAYLOAD, 0).ok_or_else(|| format!("{carrier} holds no GeoTIFF"))?;
        let name = string_at(&reply, "lcc_name", 0)
            .map(|s| unescape(s.0))
            .unwrap_or_else(|| carrier.trim_end_matches(" (raster data)").to_string());
        let file = string_at(&reply, "lcc_file", 0).map(|s| unescape(s.0)).unwrap_or_else(|| "result.tif".into());
        let kind = string_at(&reply, "lcc_kind", 0).map(|s| unescape(s.0)).unwrap_or_default();
        let report = string_at(&reply, "lcc_report", 0).map(|s| unescape(s.0)).unwrap_or_default();
        let mut a = br#"{"op":"layer.addDataBytes","params":{"fileName":"#.to_vec();
        a.extend_from_slice(&serde_json::to_vec(&file).unwrap_or_default());
        a.extend_from_slice(br#","name":"#);
        a.extend_from_slice(&serde_json::to_vec(&name).unwrap_or_default());
        a.extend_from_slice(br#","data":""#);
        a.extend_from_slice(data);
        a.extend_from_slice(br#""}}"#);
        drop(reply);
        let answer = raw::call(&a)?;
        drop(a);
        let answer: Value = serde_json::from_slice(&answer).unwrap_or(Value::Null);
        let added_name =
            answer["ok"]["name"].as_str().or_else(|| answer["ok"]["layer"].as_str()).map(str::to_string).unwrap_or(name.clone());
        let _ = host::call("layer.remove", json!({"layer": carrier}));
        if kind == "composite" {
            style_composite(&added_name, &report);
        }
        added.push(added_name);
        reports.push(report);
    }
    host::notify(&format!("Added {}", added.join(", ")));
    let report = reports.join("\n\n");
    let mut state = panel_state(req, Some(&report)).unwrap_or_else(|_| json!({}));
    state["added"] = json!(added);
    Ok((state, added.len()))
}

/// The layers that have a field `field`.
fn layers_with(field: &str) -> Result<Vec<String>, String> {
    let layers = host::layers()?;
    Ok(layers
        .as_array()
        .into_iter()
        .flatten()
        .filter(|l| l["fields"].as_array().is_some_and(|f| f.iter().any(|f| f["name"] == field)))
        .filter_map(|l| l["name"].as_str().map(str::to_string))
        .collect())
}

/// Shows a composite in natural colour when its bands are named like Sentinel-2's or Landsat's.
fn style_composite(layer: &str, report: &str) {
    let order: Vec<String> = report
        .split("Band order: ")
        .nth(1)
        .unwrap_or("")
        .split(", ")
        .filter_map(|p| p.split_once('=').map(|(_, n)| n.to_ascii_uppercase()))
        .collect();
    let find = |keys: &[&str]| {
        order.iter().position(|n| keys.iter().any(|k| n.ends_with(k) || n.contains(&format!("{k}_")) || n == k)).map(|i| i + 1)
    };
    if let (Some(r), Some(g), Some(b)) = (find(&["B4", "B04", "RED"]), find(&["B3", "B03", "GREEN"]), find(&["B2", "B02", "BLUE"])) {
        let _ = host::call("raster.bands", json!({"layer": layer, "red": r, "green": g, "blue": b}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips() {
        for n in 0..40 {
            let data: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(decode_base64(base64(&data).into_bytes()).unwrap(), data, "{n}");
        }
        assert!(decode_base64(b"Zm9v!mFy".to_vec()).is_none());
    }

    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn class_ids_round_trip() {
        let c = ClassInfo { value: 7, label: "Mangrove".into(), color: [1, 2, 3] };
        let back = class_from_id(&class_id(&c)).unwrap();
        assert_eq!((back.value, back.label.as_str(), back.color), (7, "Mangrove", [1, 2, 3]));
        assert!(class_from_id("0|x|#000000").is_none());
    }

    #[test]
    fn finds_strings_in_replies() {
        let j = br#"{"ok":{"features":[{"properties":{"lcc_name":"A \"b\"","lcc_tif":"QUJD"}}]}}"#;
        assert_eq!(string_at(j, "lcc_tif", 0).unwrap().0, b"QUJD");
        assert_eq!(unescape(string_at(j, "lcc_name", 0).unwrap().0), "A \"b\"");
    }

    #[test]
    fn algorithm_names() {
        use classify::Algorithm;
        assert_eq!(Algorithm::parse("CART (Decision Tree)"), Algorithm::Cart);
        assert_eq!(Algorithm::parse("Support Vector Machine"), Algorithm::Svm);
        assert_eq!(Algorithm::parse("k-Nearest Neighbors"), Algorithm::Knn);
        assert_eq!(Algorithm::parse("Random Forest"), Algorithm::RandomForest);
    }
}

/// Stand-ins for RustGIS's imports, so the crate links on the computer for `cargo test` (host
/// calls then fail, as they do outside RustGIS).
#[cfg(not(target_arch = "wasm32"))]
mod native_stubs {
    #[unsafe(no_mangle)]
    extern "C" fn call(_ptr: *const u8, _len: usize) -> i32 {
        -1
    }

    #[unsafe(no_mangle)]
    extern "C" fn result(_ptr: *mut u8) {}
}
