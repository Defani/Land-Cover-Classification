//! Training samples: the pixels under points, rectangles and polygons, with their classes, and
//! their split into training and test sets.
use crate::ml::Rng;
use crate::stack::Stack;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct ClassInfo {
    /// The value written in the classified raster (1–255).
    pub value: u8,
    pub label: String,
    pub color: [u8; 3],
}

pub struct Samples {
    pub x: Vec<f32>,
    pub y: Vec<u16>,
    /// The feature (point, rectangle or polygon) each pixel came from.
    pub group: Vec<u32>,
    pub d: usize,
    pub classes: Vec<ClassInfo>,
    /// Features per class.
    pub features: Vec<usize>,
    pub notes: Vec<String>,
}

/// A palette for classes without a colour of their own (ColorBrewer "Paired", then "Set1").
pub const PALETTE: [[u8; 3]; 16] = [
    [31, 120, 180],
    [51, 160, 44],
    [227, 26, 28],
    [255, 127, 0],
    [106, 61, 154],
    [177, 89, 40],
    [166, 206, 227],
    [178, 223, 138],
    [251, 154, 153],
    [253, 191, 111],
    [202, 178, 214],
    [255, 255, 153],
    [228, 26, 28],
    [55, 126, 184],
    [77, 175, 74],
    [152, 78, 163],
];

pub fn parse_color(s: &str) -> Option<[u8; 3]> {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let p = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok();
    Some([p(0)?, p(2)?, p(4)?])
}

pub fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// A feature's class: a whole number, or a name.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Key {
    Num(i64),
    Text(String),
}

fn class_key(v: &Value) -> Option<Key> {
    match v {
        Value::Number(n) => n.as_f64().filter(|f| f.fract() == 0.0).map(|f| Key::Num(f as i64)),
        Value::String(s) if !s.trim().is_empty() => {
            Some(s.trim().parse::<i64>().map(Key::Num).unwrap_or_else(|_| Key::Text(s.trim().to_string())))
        }
        _ => None,
    }
}

/// Collects the pixels of `features` (GeoJSON in degrees) from `stack`, at most `max_per` from
/// each feature (a random choice beyond it).
pub fn collect(stack: &Stack, features: &[&Value], class_field: &str, max_per: usize, seed: u64) -> Result<Samples, String> {
    let d = stack.bands.len();
    let mut rng = Rng::new(seed ^ 0x5A17);
    // The classes first, so their order (and indexes) don't depend on the features' order.
    let mut labels: HashMap<Key, (String, Option<[u8; 3]>)> = HashMap::new();
    let mut unclassed = 0;
    for f in features {
        let props = &f["properties"];
        let Some(key) = class_key(&props[class_field]) else {
            unclassed += 1;
            continue;
        };
        let label = props.get("label").and_then(Value::as_str).filter(|s| !s.trim().is_empty()).map(str::to_string);
        let color = props.get("color").and_then(Value::as_str).and_then(parse_color);
        let e = labels.entry(key.clone()).or_insert_with(|| {
            (
                label.clone().unwrap_or_else(|| match &key {
                    Key::Num(n) => format!("Class {n}"),
                    Key::Text(t) => t.clone(),
                }),
                color,
            )
        });
        if e.1.is_none() {
            e.1 = color;
        }
    }
    if labels.is_empty() {
        return Err(format!(
            "no training sample has a class in the field '{class_field}': draw samples with the Land Cover panel, or name the field that holds their class"
        ));
    }
    let mut keys: Vec<Key> = labels.keys().cloned().collect();
    keys.sort();
    let numeric_ok = keys.iter().all(|k| matches!(k, Key::Num(n) if (1..=255).contains(n)));
    let mut notes = Vec::new();
    if !numeric_ok {
        notes.push(format!(
            "Classes numbered 1 to {} in the order of their {} (the field's values aren't all whole numbers from 1 to 255)",
            keys.len(),
            class_field
        ));
    }
    if keys.len() > 255 {
        return Err("more than 255 classes".into());
    }
    if keys.len() < 2 {
        return Err(format!("the samples have only one class ({}): give at least two", labels[&keys[0]].0));
    }
    let classes: Vec<ClassInfo> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let (label, color) = &labels[k];
            let value = match k {
                Key::Num(n) if numeric_ok => *n as u8,
                _ => (i + 1) as u8,
            };
            ClassInfo { value, label: label.clone(), color: color.unwrap_or(PALETTE[(value as usize + PALETTE.len() - 1) % PALETTE.len()]) }
        })
        .collect();
    let index: HashMap<Key, u16> = keys.iter().enumerate().map(|(i, k)| (k.clone(), i as u16)).collect();

    let mut taken: HashMap<u32, u16> = HashMap::new();
    let mut s = Samples { x: Vec::new(), y: Vec::new(), group: Vec::new(), d, classes, features: vec![0; keys.len()], notes };
    let (mut outside, mut conflicts, mut capped, mut lines, mut empty) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut px = vec![0f32; d];
    for (fi, f) in features.iter().enumerate() {
        let Some(c) = class_key(&f["properties"][class_field]).and_then(|k| index.get(&k).copied()) else { continue };
        let mut cells = Vec::new();
        let kind = geometry_cells(stack, &f["geometry"], &mut cells);
        if kind == Shape::Line {
            lines += 1;
            continue;
        }
        cells.sort_unstable();
        cells.dedup();
        if cells.is_empty() {
            outside += 1;
            continue;
        }
        if cells.len() > max_per && max_per > 0 {
            rng.shuffle(&mut cells);
            cells.truncate(max_per);
            capped += 1;
        }
        let mut got = 0;
        for cell in cells {
            match taken.get(&cell) {
                Some(&o) if o == c => continue,
                Some(_) => {
                    conflicts += 1;
                    continue;
                }
                None => {}
            }
            if !stack.pixel(cell as usize, &mut px) {
                continue;
            }
            taken.insert(cell, c);
            s.x.extend_from_slice(&px);
            s.y.push(c);
            s.group.push(fi as u32);
            got += 1;
        }
        if got > 0 {
            s.features[c as usize] += 1;
        } else {
            empty += 1;
        }
    }
    if unclassed > 0 {
        s.notes.push(format!("{unclassed} sample(s) with no class in '{class_field}' left out"));
    }
    if outside > 0 {
        s.notes.push(format!("{outside} sample(s) outside the raster left out"));
    }
    if empty > 0 {
        s.notes.push(format!("{empty} sample(s) only over no-data pixels left out"));
    }
    if lines > 0 {
        s.notes.push(format!("{lines} line sample(s) left out (draw points, rectangles or polygons)"));
    }
    if conflicts > 0 {
        s.notes.push(format!("{conflicts} pixel(s) under samples of two classes kept for the first sample only"));
    }
    if capped > 0 {
        s.notes.push(format!("{capped} sample(s) larger than {max_per} pixels: a random {max_per} of their pixels used"));
    }
    if s.y.is_empty() {
        return Err("none of the samples falls on valid pixels of the raster: draw them over the image".into());
    }
    Ok(s)
}

/// The cells around each feature (its bounding box and a cell beyond), in `grid`, for reading
/// only those parts of the rasters.
pub fn regions(features: &[&Value], grid: &crate::tif::Grid, crs: &crate::proj::Crs) -> Vec<crate::tif::Rect> {
    fn walk(v: &Value, out: &mut Vec<(f64, f64)>) {
        match v {
            Value::Array(a) if a.len() >= 2 && a[0].is_number() && a[1].is_number() => {
                if let (Some(x), Some(y)) = (a[0].as_f64(), a[1].as_f64()) {
                    out.push((x, y));
                }
            }
            Value::Array(a) => a.iter().for_each(|e| walk(e, out)),
            _ => {}
        }
    }
    let mut rects = Vec::new();
    for f in features {
        let mut pts = Vec::new();
        let g = &f["geometry"];
        walk(&g["coordinates"], &mut pts);
        for part in g["geometries"].as_array().into_iter().flatten() {
            walk(&part["coordinates"], &mut pts);
        }
        if pts.is_empty() {
            continue;
        }
        let (mut c0, mut r0, mut c1, mut r1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (lon, lat) in pts {
            let (x, y) = crs.forward(lon, lat);
            let (c, r) = grid.to_pixel(x, y);
            c0 = c0.min(c);
            r0 = r0.min(r);
            c1 = c1.max(c);
            r1 = r1.max(r);
        }
        let clamp = |v: f64, n: usize| v.clamp(0.0, n as f64) as usize;
        let r = (
            clamp(c0.floor() - 1.0, grid.w),
            clamp(r0.floor() - 1.0, grid.h),
            clamp(c1.ceil() + 1.0, grid.w),
            clamp(r1.ceil() + 1.0, grid.h),
        );
        if r.0 < r.2 && r.1 < r.3 {
            rects.push(r);
        }
    }
    rects
}

/// The cells under a geometry in degrees: a point's cell, or the cells whose centres a polygon
/// holds.
pub fn cells(stack: &Stack, g: &Value, out: &mut Vec<u32>) {
    geometry_cells(stack, g, out);
}

#[derive(PartialEq)]
enum Shape {
    Area,
    Line,
}

/// The cells under a geometry: a point's cell, or the cells whose centres a polygon holds (its
/// central cell when it is smaller than one).
fn geometry_cells(stack: &Stack, g: &Value, out: &mut Vec<u32>) -> Shape {
    let coords = &g["coordinates"];
    let grid = &stack.grid;
    let to_px = |p: &Value| -> Option<(f64, f64)> {
        let (lon, lat) = (p.get(0)?.as_f64()?, p.get(1)?.as_f64()?);
        let (x, y) = stack.crs.forward(lon, lat);
        Some(grid.to_pixel(x, y))
    };
    let point = |p: &Value, out: &mut Vec<u32>| {
        if let Some((c, r)) = to_px(p)
            && c >= 0.0
            && r >= 0.0
            && (c as usize) < grid.w
            && (r as usize) < grid.h
        {
            out.push((r as usize * grid.w + c as usize) as u32);
        }
    };
    match g["type"].as_str().unwrap_or("") {
        "Point" => point(coords, out),
        "MultiPoint" => coords.as_array().into_iter().flatten().for_each(|p| point(p, out)),
        "Polygon" => polygon(coords, &to_px, grid, out),
        "MultiPolygon" => coords.as_array().into_iter().flatten().for_each(|p| polygon(p, &to_px, grid, out)),
        "GeometryCollection" => {
            for part in g["geometries"].as_array().into_iter().flatten() {
                geometry_cells(stack, part, out);
            }
        }
        "LineString" | "MultiLineString" => return Shape::Line,
        _ => {}
    }
    Shape::Area
}

fn polygon(rings: &Value, to_px: &dyn Fn(&Value) -> Option<(f64, f64)>, grid: &crate::tif::Grid, out: &mut Vec<u32>) {
    let rings: Vec<Vec<(f64, f64)>> =
        rings.as_array().into_iter().flatten().map(|r| r.as_array().into_iter().flatten().filter_map(to_px).collect()).collect();
    let Some(outer) = rings.first().filter(|r| r.len() >= 3) else { return };
    let (mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(_, y) in outer {
        miny = miny.min(y);
        maxy = maxy.max(y);
    }
    let before = out.len();
    let r0 = (miny - 0.5).ceil().max(0.0) as usize;
    let r1 = ((maxy - 0.5).floor()).min(grid.h as f64 - 1.0);
    if r1 >= 0.0 {
        let mut xs = Vec::new();
        for r in r0..=r1 as usize {
            let yc = r as f64 + 0.5;
            xs.clear();
            for ring in &rings {
                let n = ring.len();
                for i in 0..n {
                    let (a, b) = (ring[i], ring[(i + 1) % n]);
                    if (a.1 <= yc) != (b.1 <= yc) {
                        xs.push(a.0 + (yc - a.1) / (b.1 - a.1) * (b.0 - a.0));
                    }
                }
            }
            xs.sort_by(f64::total_cmp);
            for pair in xs.chunks(2) {
                if pair.len() < 2 {
                    break;
                }
                let c0 = (pair[0] - 0.5).ceil().max(0.0) as usize;
                let c1 = (pair[1] - 0.5).floor();
                if c1 < 0.0 {
                    continue;
                }
                let c1 = (c1 as usize).min(grid.w.saturating_sub(1));
                for c in c0..=c1 {
                    out.push((r * grid.w + c) as u32);
                }
            }
        }
    }
    if out.len() == before {
        // Smaller than a cell: the cell under its middle.
        let n = outer.len() as f64;
        let (cx, cy) = outer.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0 / n, a.1 + p.1 / n));
        if cx >= 0.0 && cy >= 0.0 && (cx as usize) < grid.w && (cy as usize) < grid.h {
            out.push((cy as usize * grid.w + cx as usize) as u32);
        }
    }
}

/// Splits the samples into training and test sets: `ratio` of each class to training, by pixel
/// or by whole sample (feature), so pixels of one polygon never end up on both sides.
pub fn split(s: &Samples, ratio: f64, by_feature: bool, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<String>) {
    let mut rng = Rng::new(seed ^ 0x7E57);
    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut notes = Vec::new();
    if ratio >= 1.0 {
        return ((0..s.y.len()).collect(), Vec::new(), notes);
    }
    let k = s.classes.len();
    for c in 0..k as u16 {
        let members: Vec<usize> = (0..s.y.len()).filter(|&i| s.y[i] == c).collect();
        if by_feature {
            let mut groups: Vec<u32> = members.iter().map(|&i| s.group[i]).collect();
            groups.sort_unstable();
            groups.dedup();
            if groups.len() < 2 {
                notes.push(format!("{}: only one sample, used for training only (draw more to test it)", s.classes[c as usize].label));
                train.extend(members);
                continue;
            }
            rng.shuffle(&mut groups);
            let total = members.len() as f64;
            let mut in_train: Vec<u32> = Vec::new();
            let mut n = 0usize;
            for (gi, g) in groups.iter().enumerate() {
                // Keep at least one sample for testing.
                if gi > 0 && (n as f64 >= ratio * total || gi == groups.len() - 1) {
                    break;
                }
                in_train.push(*g);
                n += members.iter().filter(|&&i| s.group[i] == *g).count();
            }
            for i in members {
                if in_train.contains(&s.group[i]) { train.push(i) } else { test.push(i) }
            }
        } else {
            let mut m = members;
            rng.shuffle(&mut m);
            let mut nt = ((m.len() as f64) * ratio).round() as usize;
            if m.len() >= 2 {
                nt = nt.clamp(1, m.len() - 1);
            } else {
                nt = m.len();
            }
            train.extend_from_slice(&m[..nt]);
            test.extend_from_slice(&m[nt..]);
        }
    }
    train.sort_unstable();
    test.sort_unstable();
    (train, test, notes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::Crs;
    use crate::tif::{GeoTags, Grid};
    use serde_json::json;

    fn stack() -> Stack {
        let grid = Grid { w: 20, h: 20, x0: 100.0, y0: 0.0, dx: 0.01, dy: 0.01 };
        let b: Vec<f32> = (0..400).map(|i| (i % 20) as f32 + 1.0).collect();
        Stack {
            grid,
            geo: GeoTags::default(),
            crs: Crs::Geographic,
            bands: vec![b.clone(), b],
            names: vec!["a".into(), "b".into()],
            sample: crate::tif::Sample::U16,
            notes: vec![],
        }
    }

    #[test]
    fn points_and_polygons_give_their_pixels() {
        let s = stack();
        let feats = [
            json!({"type": "Feature", "geometry": {"type": "Point", "coordinates": [100.055, -0.025]}, "properties": {"class": 1, "label": "Water"}}),
            // A 4 × 3 cell rectangle: cells 2..6 × 10..13.
            json!({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[100.02, -0.10], [100.06, -0.10], [100.06, -0.13], [100.02, -0.13], [100.02, -0.10]]]}, "properties": {"class": 2, "color": "#00ff00"}}),
        ];
        let refs: Vec<&Value> = feats.iter().collect();
        let got = collect(&s, &refs, "class", 1000, 1).unwrap();
        assert_eq!(got.y.len(), 1 + 12);
        assert_eq!(got.x[0], 6.0); // column 5
        assert_eq!(got.classes[0].label, "Water");
        assert_eq!(got.classes[1].label, "Class 2");
        assert_eq!(got.classes[1].color, [0, 255, 0]);
    }

    #[test]
    fn split_by_feature_keeps_polygons_whole() {
        let s = stack();
        let mut feats = Vec::new();
        for i in 0..6 {
            let x = 100.0 + 0.03 * i as f64;
            feats.push(json!({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[x, -0.0], [x + 0.02, 0.0], [x + 0.02, -0.02], [x, -0.02], [x, 0.0]]]}, "properties": {"class": 1 + i % 2}}));
        }
        let refs: Vec<&Value> = feats.iter().collect();
        let got = collect(&s, &refs, "class", 1000, 1).unwrap();
        let (train, test, _) = split(&got, 0.7, true, 3);
        assert!(!test.is_empty() && !train.is_empty());
        for &t in &test {
            assert!(!train.iter().any(|&r| got.group[r] == got.group[t]));
        }
    }
}
