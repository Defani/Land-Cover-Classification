//! Composite Bands: several rasters, each of one or more bands and of any cell size, stacked on
//! one grid (as ArcGIS's Composite Bands does, but resampling bands of other cell sizes, such as
//! Sentinel-2's 10, 20 and 60 m bands, onto the grid chosen).
use crate::proj::Crs;
use crate::tif::{GeoTags, Grid, Image, Rect};

/// Which raster's grid the composite takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GridChoice {
    Finest,
    Coarsest,
    First,
}

impl GridChoice {
    pub fn parse(s: Option<&str>) -> GridChoice {
        match s.unwrap_or("").to_ascii_lowercase() {
            s if s.starts_with("coarse") => GridChoice::Coarsest,
            s if s.starts_with("first") => GridChoice::First,
            _ => GridChoice::Finest,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Resampling {
    Nearest,
    Bilinear,
}

impl Resampling {
    pub fn parse(s: Option<&str>) -> Resampling {
        if s.unwrap_or("").to_ascii_lowercase().starts_with("bilinear") { Resampling::Bilinear } else { Resampling::Nearest }
    }
}

pub struct Stack {
    pub grid: Grid,
    pub geo: GeoTags,
    pub crs: Crs,
    pub bands: Vec<Vec<f32>>,
    pub names: Vec<String>,
    pub sample: crate::tif::Sample,
    pub notes: Vec<String>,
}

impl Stack {
    /// The band values of cell `i`, or `None` when one of them is no-data.
    #[inline]
    pub fn pixel(&self, i: usize, out: &mut [f32]) -> bool {
        for (b, band) in self.bands.iter().enumerate() {
            let v = band[i];
            if v.is_nan() {
                return false;
            }
            out[b] = v;
        }
        // Sentinel-2 and Landsat fill outside the scene with zeros in every band.
        !out.iter().all(|v| *v == 0.0)
    }
}

/// The name of a band: the file's name without its extension, and its band number when the
/// file has several.
fn band_name(file: &str, b: usize, n: usize) -> String {
    let stem = file.rsplit(['/', '\\']).next().unwrap_or(file);
    let stem = stem.rsplit_once('.').map_or(stem, |(s, _)| s);
    if n == 1 { stem.to_string() } else { format!("{stem}_b{}", b + 1) }
}

/// Which of `grids` a composite takes, by `choice`.
pub fn pick(grids: &[&Grid], choice: GridChoice) -> usize {
    match choice {
        GridChoice::First => 0,
        GridChoice::Finest => (0..grids.len()).min_by(|&a, &b| grids[a].dx.total_cmp(&grids[b].dx)).unwrap_or(0),
        GridChoice::Coarsest => (0..grids.len()).max_by(|&a, &b| grids[a].dx.total_cmp(&grids[b].dx)).unwrap_or(0),
    }
}

/// The cells of `src` under `rects` of `dst` (a cell's margin around them, for resampling).
pub fn map_rects(rects: &[Rect], dst: &Grid, src: &Grid) -> Vec<Rect> {
    if dst.same(src) {
        return rects.to_vec();
    }
    rects
        .iter()
        .filter_map(|&(c0, r0, c1, r1)| {
            let (x0, y0) = (dst.x0 + c0 as f64 * dst.dx, dst.y0 - r0 as f64 * dst.dy);
            let (x1, y1) = (dst.x0 + c1 as f64 * dst.dx, dst.y0 - r1 as f64 * dst.dy);
            let (a, b) = src.to_pixel(x0, y0);
            let (c, d) = src.to_pixel(x1, y1);
            let clamp = |v: f64, n: usize| v.clamp(0.0, n as f64) as usize;
            let r =
                (clamp(a.floor() - 1.0, src.w), clamp(b.floor() - 1.0, src.h), clamp(c.ceil() + 1.0, src.w), clamp(d.ceil() + 1.0, src.h));
            (r.0 < r.2 && r.1 < r.3).then_some(r)
        })
        .collect()
}

/// Stacks `images` on one grid. All must be in the same coordinate system. With `regions`, only
/// the cells in them are resampled (the rest stay no-data).
pub fn composite(images: Vec<Image>, choice: GridChoice, resampling: Resampling) -> Result<Stack, String> {
    composite_in(images, choice, resampling, None)
}

pub fn composite_in(images: Vec<Image>, choice: GridChoice, resampling: Resampling, regions: Option<&[Rect]>) -> Result<Stack, String> {
    let first = images.first().ok_or("give at least one raster")?;
    for im in &images[1..] {
        if im.crs != first.crs {
            return Err(format!(
                "{} is in {} but {} is in {}: give rasters in one coordinate system (reproject with Analyze › Project Raster)",
                im.name,
                im.crs.describe(),
                first.name,
                first.crs.describe()
            ));
        }
    }
    let pick = pick(&images.iter().map(|i| &i.grid).collect::<Vec<_>>(), choice);
    let grid = images[pick].grid.clone();
    let ref_name = images[pick].name.clone();
    let geo = images[pick].geo.clone();
    let crs = images[pick].crs.clone();
    let integer = images.iter().all(|i| i.integer && i.bits <= 16);
    let signed = images.iter().any(|i| i.signed);
    let sample = match (integer, signed) {
        (true, false) => crate::tif::Sample::U16,
        (true, true) => crate::tif::Sample::I16,
        _ => crate::tif::Sample::F32,
    };
    let mut notes = Vec::new();
    let mut resampled = 0;
    let mut outside = 0;
    let mut bands = Vec::new();
    let mut names = Vec::new();
    for im in images {
        let n = im.bands.len();
        let same = im.grid.same(&grid);
        if !same {
            resampled += n;
            let overlap = overlaps(&im.grid, &grid);
            if !overlap {
                outside += 1;
            }
        }
        for (b, band) in im.bands.into_iter().enumerate() {
            names.push(band_name(&im.name, b, n));
            bands.push(if same { band } else { resample(&band, &im.grid, &grid, resampling, regions) });
        }
    }
    if outside > 0 {
        return Err(format!("{outside} raster(s) don't overlap the grid of {ref_name}: give rasters of the same place"));
    }
    if resampled > 0 {
        let how = if resampling == Resampling::Bilinear { "bilinear" } else { "nearest-neighbour" };
        notes.push(format!(
            "{resampled} band(s) resampled ({how}) onto the grid of {} ({} × {} cells of {})",
            ref_name,
            grid.w,
            grid.h,
            cell(&grid, &crs)
        ));
    }
    Ok(Stack { grid, geo, crs, bands, names, sample, notes })
}

/// A cell size for messages.
pub fn cell(grid: &Grid, crs: &Crs) -> String {
    if crs.metric() { format!("{} m", trim(grid.dx)) } else { format!("{}°", trim(grid.dx)) }
}

fn trim(v: f64) -> String {
    let s = format!("{v:.8}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn overlaps(a: &Grid, b: &Grid) -> bool {
    let (ax1, ay1) = (a.x0 + a.w as f64 * a.dx, a.y0 - a.h as f64 * a.dy);
    let (bx1, by1) = (b.x0 + b.w as f64 * b.dx, b.y0 - b.h as f64 * b.dy);
    a.x0 < bx1 && b.x0 < ax1 && ay1 < b.y0 && by1 < a.y0
}

/// `band` on `src` resampled onto `dst` (the same coordinate system), in `regions` of `dst` or
/// all of it. The source column and row of each output column and row are worked out once, so
/// the inner loop only indexes.
fn resample(band: &[f32], src: &Grid, dst: &Grid, how: Resampling, regions: Option<&[Rect]>) -> Vec<f32> {
    let mut out = vec![f32::NAN; dst.cells()];
    let ox = (dst.x0 - src.x0) / src.dx;
    let oy = (src.y0 - dst.y0) / src.dy;
    let fx: Vec<f64> = (0..dst.w).map(|c| ox + (c as f64 + 0.5) * dst.dx / src.dx).collect();
    let fy: Vec<f64> = (0..dst.h).map(|r| oy + (r as f64 + 0.5) * dst.dy / src.dy).collect();
    let near = |f: f64, n: usize| -> Option<usize> { if f >= 0.0 && (f as usize) < n { Some(f as usize) } else { None } };
    let cols: Vec<Option<usize>> = fx.iter().map(|&f| near(f, src.w)).collect();
    let whole = [(0, 0, dst.w, dst.h)];
    let regions = regions.unwrap_or(&whole);
    let lin = |f: f64, n: usize| -> (usize, usize, f32) {
        let x = (f - 0.5).clamp(0.0, (n - 1) as f64);
        let a = x.floor() as usize;
        (a, (a + 1).min(n - 1), (x - a as f64) as f32)
    };
    let cw: Vec<(usize, usize, f32)> = if how == Resampling::Bilinear { fx.iter().map(|&f| lin(f, src.w)).collect() } else { Vec::new() };
    for &(c0, r0, c1, r1) in regions {
        let (c1, r1) = (c1.min(dst.w), r1.min(dst.h));
        if c0 >= c1 {
            continue;
        }
        for r in r0..r1 {
            let Some(sr) = near(fy[r], src.h) else { continue };
            let srow = &band[sr * src.w..(sr + 1) * src.w];
            let orow = &mut out[r * dst.w + c0..r * dst.w + c1];
            match how {
                Resampling::Nearest => {
                    for (o, c) in orow.iter_mut().zip(&cols[c0..c1]) {
                        if let Some(c) = *c {
                            *o = srow[c];
                        }
                    }
                }
                Resampling::Bilinear => {
                    let (ra, rb, ty) = lin(fy[r], src.h);
                    let (a, b) = (&band[ra * src.w..(ra + 1) * src.w], &band[rb * src.w..(rb + 1) * src.w]);
                    for (i, o) in orow.iter_mut().enumerate() {
                        let c = c0 + i;
                        let Some(nc) = cols[c] else { continue };
                        let (x0, x1, tx) = cw[c];
                        let v = (a[x0] * (1.0 - tx) + a[x1] * tx) * (1.0 - ty) + (b[x0] * (1.0 - tx) + b[x1] * tx) * ty;
                        // Next to no-data: the nearest cell instead.
                        *o = if v.is_nan() { srow[nc] } else { v };
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(name: &str, w: usize, dx: f64, f: impl Fn(usize, usize) -> f32) -> Image {
        let grid = Grid { w, h: w, x0: 0.0, y0: 100.0, dx, dy: dx };
        let band = (0..w * w).map(|i| f(i % w, i / w)).collect();
        Image {
            name: name.into(),
            grid,
            bands: vec![band],
            geo: GeoTags::default(),
            crs: Crs::Utm { zone: 50, south: true },
            integer: true,
            bits: 16,
            signed: false,
            class_names: vec![],
        }
    }

    #[test]
    fn stacks_bands_of_two_cell_sizes_on_the_finest() {
        let fine = image("S2A_B4.tif", 10, 10.0, |c, _| c as f32);
        let coarse = image("S2A_B11.tif", 5, 20.0, |c, r| (c * 10 + r) as f32);
        let s = composite(vec![coarse, fine], GridChoice::Finest, Resampling::Nearest).unwrap();
        assert_eq!(s.grid.w, 10);
        assert_eq!(s.names, vec!["S2A_B11", "S2A_B4"]);
        // Fine cell (3, 2) lies in coarse cell (1, 1).
        assert_eq!(s.bands[0][2 * 10 + 3], 11.0);
        assert_eq!(s.bands[1][2 * 10 + 3], 3.0);
        assert_eq!(s.notes.len(), 1);
    }

    #[test]
    fn refuses_mixed_coordinate_systems() {
        let a = image("a.tif", 4, 10.0, |_, _| 1.0);
        let mut b = image("b.tif", 4, 10.0, |_, _| 1.0);
        b.crs = Crs::Geographic;
        assert!(composite(vec![a, b], GridChoice::First, Resampling::Nearest).is_err());
    }
}
