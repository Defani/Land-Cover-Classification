//! Reading single- and multi-band GeoTIFFs, and writing the classified map and the composite.
use crate::proj::Crs;

/// A north-up grid: column `c`, row `r` has its top-left corner at `(x0 + c·dx, y0 − r·dy)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub x0: f64,
    pub y0: f64,
    pub dx: f64,
    pub dy: f64,
}

impl Grid {
    /// The point `(x, y)` in pixel units (column, row) from the grid's top-left corner.
    pub fn to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.x0) / self.dx, (self.y0 - y) / self.dy)
    }

    /// The centre of the cell at column `c`, row `r`.
    pub fn center(&self, c: usize, r: usize) -> (f64, f64) {
        (self.x0 + (c as f64 + 0.5) * self.dx, self.y0 - (r as f64 + 0.5) * self.dy)
    }

    pub fn cells(&self) -> usize {
        self.w * self.h
    }

    /// Whether two grids are the same cells (within a thousandth of a cell).
    pub fn same(&self, o: &Grid) -> bool {
        let tol = 1e-3 * self.dx.abs().min(self.dy.abs());
        self.w == o.w
            && self.h == o.h
            && (self.x0 - o.x0).abs() < tol
            && (self.y0 - o.y0).abs() < tol
            && (self.dx - o.dx).abs() < tol * 1e-3 + 1e-12
            && (self.dy - o.dy).abs() < tol * 1e-3 + 1e-12
    }
}

/// The GeoTIFF keys of a file, copied unchanged into what is written from it.
#[derive(Clone, Debug, Default)]
pub struct GeoTags {
    pub keys: Vec<u16>,
    pub doubles: Vec<f64>,
    pub ascii: String,
}

/// One input raster: its bands as `f32` with no-data as NaN.
pub struct Image {
    pub name: String,
    pub grid: Grid,
    pub bands: Vec<Vec<f32>>,
    pub geo: GeoTags,
    pub crs: Crs,
    /// Whether its samples are whole numbers (so a composite can stay 16-bit).
    pub integer: bool,
    /// The largest sample type it uses, in bits (for the composite's type).
    pub bits: u16,
    pub signed: bool,
    /// Class names by value, from GDAL's metadata (`CLASS_<value>` items).
    pub class_names: Vec<(u32, String)>,
}

/// A TIFF's bytes: as they are, or still in base64 (decoded a range at a time, so a run that
/// needs a few tiles of a large file decodes only those).
#[derive(Clone, Copy)]
pub enum Src<'a> {
    Raw(&'a [u8]),
    B64(&'a [u8]),
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_table() -> [u8; 256] {
    let mut t = [255u8; 256];
    for (i, c) in B64.iter().enumerate() {
        t[*c as usize] = i as u8;
    }
    t
}

impl<'a> Src<'a> {
    pub fn len(&self) -> usize {
        match self {
            Src::Raw(b) => b.len(),
            Src::B64(s) => {
                let pad = s.iter().rev().take(2).take_while(|&&c| c == b'=').count();
                (s.len() / 4 * 3).saturating_sub(pad)
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `len` bytes from `off`.
    pub fn get(&self, off: usize, len: usize) -> Option<std::borrow::Cow<'a, [u8]>> {
        if off.checked_add(len)? > self.len() {
            return None;
        }
        match *self {
            Src::Raw(b) => Some(std::borrow::Cow::Borrowed(&b[off..off + len])),
            Src::B64(s) => {
                let t = b64_table();
                let (q0, q1) = (off / 3, (off + len).div_ceil(3));
                let mut out = vec![0u8; (q1 - q0) * 3];
                let mut bad = 0u8;
                for (q, o) in s[q0 * 4..(q1 * 4).min(s.len())].as_chunks::<4>().0.iter().zip(out.chunks_exact_mut(3)) {
                    // Padding decodes as zeros, which the length leaves out.
                    let v = |c: u8| if c == b'=' { 0 } else { t[c as usize] };
                    let (a, b, c, d) = (v(q[0]), v(q[1]), v(q[2]), v(q[3]));
                    bad |= a | b | c | d;
                    let n = (u32::from(a) << 18) | (u32::from(b) << 12) | (u32::from(c) << 6) | u32::from(d);
                    o[0] = (n >> 16) as u8;
                    o[1] = (n >> 8) as u8;
                    o[2] = n as u8;
                }
                if bad & 0xC0 != 0 {
                    return None;
                }
                let s0 = off - q0 * 3;
                out.truncate(s0 + len);
                out.drain(..s0);
                Some(std::borrow::Cow::Owned(out))
            }
        }
    }
}

/// Reads a GeoTIFF's first image (classic or BigTIFF, either byte order; uncompressed, LZW,
/// Deflate or PackBits, with or without a predictor; strips or tiles; bands interleaved or
/// apart; 8 to 64-bit integers or floats). Overviews are left alone.
///
/// Written for this plugin rather than taken from a crate: RustGIS runs plugins in an
/// interpreter that counts every instruction, and this decoder reads only the tiles a run needs.
pub fn read(name: &str, bytes: &[u8]) -> Result<Image, String> {
    let o = open(name, Src::Raw(bytes))?;
    let bands = o.read_region(&[(0, 0, o.grid.w, o.grid.h)])?;
    Ok(o.into_image(bands))
}

struct Tiff<'a> {
    src: Src<'a>,
    le: bool,
    big: bool,
    /// Tag → (type, count, offset of the value or of the inline value).
    tags: Vec<(u16, u16, u64, usize)>,
}

impl Tiff<'_> {
    fn uint(&self, o: usize, n: usize) -> Option<u64> {
        let s = self.src.get(o, n)?;
        let mut v = 0u64;
        if self.le {
            for &x in s.iter().rev() {
                v = v << 8 | u64::from(x);
            }
        } else {
            for &x in s.iter() {
                v = v << 8 | u64::from(x);
            }
        }
        Some(v)
    }

    fn tag(&self, id: u16) -> Option<&(u16, u16, u64, usize)> {
        self.tags.iter().find(|t| t.0 == id)
    }

    /// A tag's values as numbers.
    fn nums(&self, id: u16) -> Option<Vec<f64>> {
        let &(_, typ, count, at) = self.tag(id)?;
        let size = match typ {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 | 16 | 17 => 8,
            _ => return None,
        };
        let count = usize::try_from(count).ok()?.min(self.src.len() / size + 1);
        let all = self.src.get(at, count * size)?;
        let rd = |o: usize, n: usize| -> u64 {
            let s = &all[o..o + n];
            if self.le {
                s.iter().rev().fold(0u64, |a, &x| a << 8 | u64::from(x))
            } else {
                s.iter().fold(0u64, |a, &x| a << 8 | u64::from(x))
            }
        };
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let o = i * size;
            let u = rd(o, size);
            out.push(match typ {
                6 => u as u8 as i8 as f64,
                8 => u as u16 as i16 as f64,
                9 => u as u32 as i32 as f64,
                17 => u as i64 as f64,
                11 => f32::from_bits(u as u32) as f64,
                12 => f64::from_bits(u),
                5 | 10 => {
                    let (n, d) = (rd(o, 4), rd(o + 4, 4));
                    if typ == 10 { (n as u32 as i32) as f64 / (d as u32 as i32).max(1) as f64 } else { n as f64 / d.max(1) as f64 }
                }
                _ => u as f64,
            });
        }
        Some(out)
    }

    fn num(&self, id: u16) -> Option<f64> {
        let &(_, _, count, _) = self.tag(id)?;
        if count == 0 {
            return None;
        }
        // Only the first value.
        let mut t = self.tags.clone();
        if let Some(e) = t.iter_mut().find(|e| e.0 == id) {
            e.2 = 1;
        }
        Tiff { src: self.src, le: self.le, big: self.big, tags: t }.nums(id)?.first().copied()
    }

    fn ascii(&self, id: u16) -> Option<String> {
        let &(_, typ, count, at) = self.tag(id)?;
        if typ != 2 {
            return None;
        }
        let s = self.src.get(at, usize::try_from(count).ok()?)?;
        Some(String::from_utf8_lossy(&s).trim_end_matches('\0').to_string())
    }
}

fn read_tiff(src: Src<'_>) -> Result<Tiff<'_>, String> {
    let le = match src.get(0, 2).as_deref() {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return Err("isn't a TIFF".into()),
    };
    let mut t = Tiff { src, le, big: false, tags: Vec::new() };
    let magic = t.uint(2, 2).ok_or("isn't a TIFF")?;
    let (ifd, count_size, entry, inline) = match magic {
        42 => (t.uint(4, 4).ok_or("truncated")? as usize, 2, 12, 4),
        43 => {
            t.big = true;
            (t.uint(8, 8).ok_or("truncated")? as usize, 8, 20, 8)
        }
        _ => return Err("isn't a TIFF".into()),
    };
    let n = t.uint(ifd, count_size).ok_or("truncated directory")? as usize;
    let n = n.min(4096);
    let base = ifd + count_size;
    let dir = src.get(base, n * entry).ok_or("truncated directory")?;
    let rd = |o: usize, k: usize| -> u64 {
        let s = &dir[o..o + k];
        if le { s.iter().rev().fold(0u64, |a, &x| a << 8 | u64::from(x)) } else { s.iter().fold(0u64, |a, &x| a << 8 | u64::from(x)) }
    };
    for e in 0..n {
        let o = e * entry;
        let id = rd(o, 2) as u16;
        let typ = rd(o + 2, 2) as u16;
        let (count, vo) = if t.big { (rd(o + 4, 8), o + 12) } else { (rd(o + 4, 4), o + 8) };
        let size: u64 = match typ {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            _ => 8,
        };
        let at = if count.saturating_mul(size) <= inline as u64 { base + vo } else { rd(vo, inline) as usize };
        t.tags.push((id, typ, count, at));
    }
    Ok(t)
}

/// A GeoTIFF opened: its georeferencing and the layout of its image data, read from the
/// header only. [`Opened::read_region`] decodes the pixels.
pub struct Opened<'a> {
    pub name: String,
    pub grid: Grid,
    pub geo: GeoTags,
    pub crs: Crs,
    pub bands: usize,
    pub integer: bool,
    pub bits: u16,
    pub signed: bool,
    pub class_names: Vec<(u32, String)>,
    t: Tiff<'a>,
    nb: usize,
    fmt: u16,
    planar: u16,
    compression: u16,
    predictor: u16,
    tiled: bool,
    cw: usize,
    ch: usize,
    offsets: Vec<f64>,
    counts: Vec<f64>,
    nodata: Option<f64>,
}

pub fn open<'a>(name: &str, src: Src<'a>) -> Result<Opened<'a>, String> {
    let t = read_tiff(src).map_err(|e| format!("{name}: {e}"))?;
    let w = t.num(256).ok_or_else(|| format!("{name}: has no width"))? as usize;
    let h = t.num(257).ok_or_else(|| format!("{name}: has no height"))? as usize;
    let spp = t.num(277).unwrap_or(1.0).max(1.0) as usize;
    let bits = t.num(258).unwrap_or(1.0) as u16;
    let fmt = t.num(339).unwrap_or(1.0) as u16;
    let planar = t.num(284).unwrap_or(1.0) as u16;
    let compression = t.num(259).unwrap_or(1.0) as u16;
    let predictor = t.num(317).unwrap_or(1.0) as u16;
    if w == 0 || h == 0 || w.saturating_mul(h) > 400_000_000 {
        return Err(format!("{name}: its size ({w} × {h}) can't be read"));
    }
    if !matches!(bits, 8 | 16 | 32 | 64) {
        return Err(format!("{name}: {bits}-bit samples aren't supported"));
    }
    if spp > 64 {
        return Err(format!("{name}: {spp} bands per pixel aren't supported"));
    }
    match compression {
        1 | 5 | 8 | 32946 | 32773 => {}
        7 => return Err(format!("{name}: JPEG-compressed rasters aren't supported; save it with LZW or Deflate")),
        c => return Err(format!("{name}: compression {c} isn't supported; save it with LZW or Deflate")),
    }
    let nodata = t.ascii(42113).and_then(|s| s.trim().parse::<f64>().ok());
    let keys: Vec<u16> = t.nums(34735).unwrap_or_default().into_iter().map(|v| v as u16).collect();
    let geo = GeoTags { keys, doubles: t.nums(34736).unwrap_or_default(), ascii: t.ascii(34737).unwrap_or_default() };
    let class_names = t.ascii(42112).map(|x| class_items(&x)).unwrap_or_default();
    let mut grid = match t.nums(34264) {
        Some(m) if m.len() >= 8 => {
            if m[1].abs() > 1e-12 || m[4].abs() > 1e-12 {
                return Err(format!("{name}: the raster is rotated; warp it to a north-up grid first (Analyze › Project Raster)"));
            }
            Grid { w, h, x0: m[3], y0: m[7], dx: m[0], dy: -m[5] }
        }
        _ => {
            let scale = t.nums(33550).ok_or_else(|| format!("{name}: has no georeferencing (it isn't a GeoTIFF)"))?;
            let tie = t.nums(33922).ok_or_else(|| format!("{name}: has no tie point (it isn't a GeoTIFF)"))?;
            if scale.len() < 2 || tie.len() < 6 {
                return Err(format!("{name}: its georeferencing is incomplete"));
            }
            Grid { w, h, x0: tie[3] - tie[0] * scale[0], y0: tie[4] + tie[1] * scale[1], dx: scale[0], dy: scale[1] }
        }
    };
    if !(grid.dx > 0.0 && grid.dy > 0.0) {
        return Err(format!("{name}: the raster isn't north-up (its cell size is {} × {})", grid.dx, grid.dy));
    }
    // GTRasterTypeGeoKey 2 (PixelIsPoint): the tie point is a cell's centre, as GDAL reads it.
    if crate::proj::geokey(&geo.keys, 1025) == Some(2) {
        grid.x0 -= grid.dx / 2.0;
        grid.y0 += grid.dy / 2.0;
    }
    let crs = crate::proj::Crs::from_geokeys(&geo).map_err(|e| format!("{name}: {e}"))?;
    // Chunks: tiles, or strips (a strip is a tile as wide as the image).
    let tiled = t.tag(322).is_some();
    let (cw, ch) = if tiled {
        (t.num(322).unwrap_or(0.0) as usize, t.num(323).unwrap_or(0.0) as usize)
    } else {
        (w, (t.num(278).unwrap_or(h as f64) as usize).clamp(1, h))
    };
    if cw == 0 || ch == 0 {
        return Err(format!("{name}: its tiles have no size"));
    }
    let offsets = t.nums(if tiled { 324 } else { 273 }).ok_or_else(|| format!("{name}: has no image data"))?;
    let counts = t.nums(if tiled { 325 } else { 279 }).ok_or_else(|| format!("{name}: has no image data"))?;
    Ok(Opened {
        name: name.to_string(),
        grid,
        geo,
        crs,
        bands: spp,
        integer: fmt != 3,
        bits,
        signed: fmt == 2,
        class_names,
        t,
        nb: (bits / 8) as usize,
        fmt,
        planar,
        compression,
        predictor,
        tiled,
        cw,
        ch,
        offsets,
        counts,
        nodata,
    })
}

/// A rectangle of cells: columns `c0..c1`, rows `r0..r1`.
pub type Rect = (usize, usize, usize, usize);

impl Opened<'_> {
    fn chunk_rect(&self, ci: usize) -> (usize, usize, Rect) {
        let (w, h) = (self.grid.w, self.grid.h);
        let across = w.div_ceil(self.cw);
        let per_band = across * h.div_ceil(self.ch);
        let band0 = if self.planar == 2 { ci / per_band } else { 0 };
        let tile = ci % per_band;
        let (tx, ty) = (tile % across, tile / across);
        let (c0, r0) = (tx * self.cw, ty * self.ch);
        (band0, tile, (c0, r0, (c0 + self.cw).min(w), (r0 + self.ch).min(h)))
    }

    /// About how many fuel units decoding the chunks under `rects` costs (LZW is the dear one).
    pub fn cost(&self, rects: &[Rect]) -> f64 {
        let per_byte = match self.compression {
            5 => 260.0,
            8 | 32946 => 90.0,
            _ => 30.0,
        };
        (0..self.offsets.len())
            .filter(|&ci| {
                let (_, _, r) = self.chunk_rect(ci);
                rects.iter().any(|q| overlap(&r, q))
            })
            .map(|ci| {
                let (_, _, r) = self.chunk_rect(ci);
                self.counts[ci] * per_byte + ((r.2 - r.0) * (r.3 - r.1) * self.bands) as f64 * 30.0
            })
            .sum()
    }

    /// The bands, decoded where a chunk meets one of `rects` (NaN elsewhere and at no-data).
    pub fn read_region(&self, rects: &[Rect]) -> Result<Vec<Vec<f32>>, String> {
        let name = &self.name;
        let (w, h) = (self.grid.w, self.grid.h);
        let spp = self.bands;
        let nb = self.nb;
        let chunk_spp = if self.planar == 2 { 1 } else { spp };
        let mut bands: Vec<Vec<f32>> = (0..spp).map(|_| vec![f32::NAN; w * h]).collect();
        let mut raw: Vec<u8> = Vec::new();
        let row_bytes = self.cw * chunk_spp * nb;
        let mut missing = 0;
        let nd = self.nodata.map(|v| v as f32);
        for ci in 0..self.offsets.len().min(self.counts.len()) {
            let (band0, _, r) = self.chunk_rect(ci);
            if band0 >= spp || !rects.iter().any(|q| overlap(&r, q)) {
                continue;
            }
            let rows = r.3 - r.1;
            let want = row_bytes * if self.tiled { self.ch } else { rows };
            let Some(src) = self.t.src.get(self.offsets[ci] as usize, self.counts[ci] as usize) else {
                missing += 1;
                continue;
            };
            raw.clear();
            match self.compression {
                5 => lzw(&src, want, &mut raw),
                8 | 32946 => {
                    raw = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&src, want)
                        .or_else(|_| miniz_oxide::inflate::decompress_to_vec_with_limit(&src, want))
                        .map_err(|e| format!("{name}: a Deflate chunk is damaged ({e:?})"))?;
                }
                32773 => packbits(&src, want, &mut raw),
                _ => raw.extend_from_slice(&src),
            }
            raw.resize(want, 0);
            if self.predictor == 2 {
                undo_horizontal(&mut raw, row_bytes, chunk_spp, nb, self.t.le);
            } else if self.predictor == 3 {
                undo_float(&mut raw, self.cw, chunk_spp, nb, self.t.le);
            }
            let cols = r.2 - r.0;
            for rr in 0..rows {
                let y = r.1 + rr;
                let line = &raw[rr * row_bytes..(rr + 1) * row_bytes];
                for b in 0..chunk_spp {
                    let dst = &mut bands[band0 + b][y * w + r.0..y * w + r.0 + cols];
                    decode_samples(line, b, chunk_spp, nb, self.fmt, self.t.le, dst, self.predictor == 3);
                    if let Some(nd) = nd {
                        for v in dst.iter_mut() {
                            if *v == nd {
                                *v = f32::NAN;
                            }
                        }
                    }
                }
            }
        }
        if missing > 0 {
            return Err(format!("{name}: {missing} chunk(s) of image data lie past the end of the file (it is truncated)"));
        }
        Ok(bands)
    }

    pub fn into_image(self, bands: Vec<Vec<f32>>) -> Image {
        Image {
            name: self.name,
            grid: self.grid,
            bands,
            geo: self.geo,
            crs: self.crs,
            integer: self.integer,
            bits: self.bits,
            signed: self.signed,
            class_names: self.class_names,
        }
    }

    /// An image of the bands decoded under `rects`.
    pub fn image(&self, rects: &[Rect]) -> Result<Image, String> {
        let bands = self.read_region(rects)?;
        Ok(Image {
            name: self.name.clone(),
            grid: self.grid.clone(),
            bands,
            geo: self.geo.clone(),
            crs: self.crs.clone(),
            integer: self.integer,
            bits: self.bits,
            signed: self.signed,
            class_names: self.class_names.clone(),
        })
    }
}

pub fn overlap(a: &Rect, b: &Rect) -> bool {
    a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3
}

/// One band's samples of a line into `dst` (as many as it holds).
#[allow(clippy::too_many_arguments)]
fn decode_samples(line: &[u8], b: usize, spp: usize, nb: usize, fmt: u16, le: bool, dst: &mut [f32], shuffled_float: bool) {
    let stride = spp * nb;
    let mut o = b * nb;
    // A floating-point predictor leaves the bytes big-endian, whatever the file's order.
    let le = if shuffled_float { false } else { le };
    // The common types get a loop of their own (one branch per line, not per sample).
    macro_rules! fill {
        ($f:expr) => {{
            for d in dst.iter_mut() {
                *d = $f(&line[o..o + nb]);
                o += stride;
            }
            return;
        }};
    }
    match (nb, fmt, le) {
        (1, 1, _) => fill!(|s: &[u8]| s[0] as f32),
        (2, 1, true) => fill!(|s: &[u8]| u16::from_le_bytes([s[0], s[1]]) as f32),
        (2, 2, true) => fill!(|s: &[u8]| i16::from_le_bytes([s[0], s[1]]) as f32),
        (4, 3, true) => fill!(|s: &[u8]| f32::from_le_bytes([s[0], s[1], s[2], s[3]])),
        _ => {}
    }
    for d in dst.iter_mut() {
        let s = &line[o..o + nb];
        *d = match (nb, fmt, le) {
            (1, 2, _) => s[0] as i8 as f32,
            (1, _, _) => s[0] as f32,
            (2, 2, true) => i16::from_le_bytes([s[0], s[1]]) as f32,
            (2, 2, false) => i16::from_be_bytes([s[0], s[1]]) as f32,
            (2, _, true) => u16::from_le_bytes([s[0], s[1]]) as f32,
            (2, _, false) => u16::from_be_bytes([s[0], s[1]]) as f32,
            (4, 3, true) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
            (4, 3, false) => f32::from_be_bytes([s[0], s[1], s[2], s[3]]),
            (4, 2, true) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32,
            (4, 2, false) => i32::from_be_bytes([s[0], s[1], s[2], s[3]]) as f32,
            (4, _, true) => u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32,
            (4, _, false) => u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as f32,
            (8, 3, true) => f64::from_le_bytes(s.try_into().unwrap_or([0; 8])) as f32,
            (8, 3, false) => f64::from_be_bytes(s.try_into().unwrap_or([0; 8])) as f32,
            (8, 2, true) => i64::from_le_bytes(s.try_into().unwrap_or([0; 8])) as f32,
            (8, 2, false) => i64::from_be_bytes(s.try_into().unwrap_or([0; 8])) as f32,
            (8, _, true) => u64::from_le_bytes(s.try_into().unwrap_or([0; 8])) as f32,
            (_, _, _) => u64::from_be_bytes(s.try_into().unwrap_or([0; 8])) as f32,
        };
        o += stride;
    }
}

/// TIFF's LZW (MSB-first codes, 9 to 12 bits, early change), into `out` resized to `want`.
/// Strings are written backwards straight into place, and codes are read from a 64-bit buffer
/// filled several bytes at a time: every instruction counts in RustGIS's metered interpreter.
fn lzw(src: &[u8], want: usize, out: &mut Vec<u8>) {
    const CLEAR: usize = 256;
    const EOI: usize = 257;
    out.clear();
    out.resize(want, 0);
    let o = &mut out[..];
    let mut prefix = [0u16; 4096];
    let mut suffix = [0u8; 4096];
    let mut first = [0u8; 4096];
    let mut len = [0u16; 4096];
    for i in 0..256 {
        suffix[i] = i as u8;
        first[i] = i as u8;
        len[i] = 1;
    }
    let (mut bits, mut have, mut pos) = (0u64, 0u32, 0usize);
    let mut width = 9u32;
    let mut next = 258usize;
    let mut old = usize::MAX;
    let mut w = 0usize;
    while w < want {
        if have < width {
            while have <= 56 && pos < src.len() {
                bits |= u64::from(src[pos]) << (56 - have);
                have += 8;
                pos += 1;
            }
            if have < width {
                break;
            }
        }
        let code = (bits >> (64 - width)) as usize & 4095;
        bits <<= width;
        have -= width;
        if code == EOI {
            break;
        }
        if code == CLEAR {
            width = 9;
            next = 258;
            old = usize::MAX;
            continue;
        }
        if old == usize::MAX {
            if code > 255 {
                break;
            }
            o[w] = code as u8;
            w += 1;
            old = code;
            continue;
        }
        let known = code < next;
        if !known && code != next {
            break;
        }
        let emit = if known { code } else { old };
        let l = len[emit] as usize;
        let end = w + l;
        if end > want {
            break;
        }
        let mut c = emit;
        let mut i = end;
        while i > w {
            i -= 1;
            o[i] = suffix[c];
            c = prefix[c] as usize & 4095;
        }
        w = end;
        let f = if known { first[code] } else { first[old] };
        if !known {
            if w >= want {
                break;
            }
            o[w] = f;
            w += 1;
        }
        if next < 4096 {
            prefix[next] = old as u16;
            suffix[next] = f;
            first[next] = first[old];
            len[next] = len[old] + 1;
            next += 1;
        }
        width = if next >= 2047 {
            12
        } else if next >= 1023 {
            11
        } else if next >= 511 {
            10
        } else {
            9
        };
        old = code;
    }
}

fn packbits(src: &[u8], want: usize, out: &mut Vec<u8>) {
    let mut i = 0;
    while i < src.len() && out.len() < want {
        let n = src[i] as i8;
        i += 1;
        if n >= 0 {
            let k = n as usize + 1;
            out.extend_from_slice(&src[i..(i + k).min(src.len())]);
            i += k;
        } else if n != -128 {
            let k = (1 - n as isize) as usize;
            if let Some(&v) = src.get(i) {
                out.extend(std::iter::repeat_n(v, k));
            }
            i += 1;
        }
    }
}

/// Undoes horizontal differencing (predictor 2), sample by sample along each row.
fn undo_horizontal(raw: &mut [u8], row_bytes: usize, spp: usize, nb: usize, le: bool) {
    for row in raw.chunks_mut(row_bytes) {
        match nb {
            1 => {
                for i in spp..row.len() {
                    row[i] = row[i].wrapping_add(row[i - spp]);
                }
            }
            2 => {
                let get = |r: &[u8], i: usize| {
                    if le { u16::from_le_bytes([r[2 * i], r[2 * i + 1]]) } else { u16::from_be_bytes([r[2 * i], r[2 * i + 1]]) }
                };
                for i in spp..row.len() / 2 {
                    let v = get(row, i).wrapping_add(get(row, i - spp));
                    let b = if le { v.to_le_bytes() } else { v.to_be_bytes() };
                    row[2 * i] = b[0];
                    row[2 * i + 1] = b[1];
                }
            }
            4 => {
                let get = |r: &[u8], i: usize| {
                    let s = [r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]];
                    if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) }
                };
                for i in spp..row.len() / 4 {
                    let v = get(row, i).wrapping_add(get(row, i - spp));
                    let b = if le { v.to_le_bytes() } else { v.to_be_bytes() };
                    row[4 * i..4 * i + 4].copy_from_slice(&b);
                }
            }
            _ => {}
        }
    }
}

/// Undoes the floating-point predictor (3): byte differencing along the row, then the bytes of
/// each sample gathered from their planes (most significant first).
fn undo_float(raw: &mut [u8], width: usize, spp: usize, nb: usize, _le: bool) {
    let row_bytes = width * spp * nb;
    let mut tmp = vec![0u8; row_bytes];
    for row in raw.chunks_mut(row_bytes) {
        for i in spp..row.len() {
            row[i] = row[i].wrapping_add(row[i - spp]);
        }
        let count = width * spp;
        for s in 0..count {
            for byte in 0..nb {
                tmp[s * nb + byte] = row[byte * count + s];
            }
        }
        row.copy_from_slice(&tmp[..row.len()]);
    }
}

/// The `<Item name="CLASS_<value>">name</Item>` items of GDAL's metadata.
fn class_items(xml: &str) -> Vec<(u32, String)> {
    let mut out = Vec::new();
    for part in xml.split("<Item name=\"CLASS_").skip(1) {
        let Some((v, rest)) = part.split_once("\">") else { continue };
        let Some((name, _)) = rest.split_once("</Item>") else { continue };
        if let Ok(v) = v.parse::<u32>() {
            let name = name.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&amp;", "&");
            out.push((v, name));
        }
    }
    out
}

// ---- Writing -------------------------------------------------------------------------------

enum V {
    Short(Vec<u16>),
    Long(Vec<u32>),
    Double(Vec<f64>),
    Ascii(String),
}

impl V {
    fn kind(&self) -> (u16, usize, usize) {
        // (TIFF type, count, bytes)
        match self {
            V::Short(v) => (3, v.len(), v.len() * 2),
            V::Long(v) => (4, v.len(), v.len() * 4),
            V::Double(v) => (12, v.len(), v.len() * 8),
            V::Ascii(s) => (2, s.len() + 1, s.len() + 1),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            V::Short(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            V::Long(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            V::Double(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            V::Ascii(s) => {
                let mut b = s.as_bytes().to_vec();
                b.push(0);
                b
            }
        }
    }
}

/// Escapes text for GDAL's XML metadata.
fn xml(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// A classic little-endian TIFF from compressed strips and tags (the strip tags are added).
fn assemble(strips: Vec<Vec<u8>>, mut tags: Vec<(u16, V)>, rows_per_strip: u32) -> Result<Vec<u8>, String> {
    let data_len: usize = strips.iter().map(Vec::len).sum();
    if data_len > 3_900_000_000 {
        return Err("the output is too large for a TIFF".into());
    }
    let mut out = Vec::with_capacity(data_len + 4096);
    out.extend_from_slice(b"II*\0\0\0\0\0");
    let mut offsets = Vec::with_capacity(strips.len());
    let mut counts = Vec::with_capacity(strips.len());
    // Each strip is freed once copied: the module has 512 MB.
    for s in strips {
        offsets.push(out.len() as u32);
        counts.push(s.len() as u32);
        out.extend_from_slice(&s);
        drop(s);
        if out.len() % 2 == 1 {
            out.push(0);
        }
    }
    tags.push((273, V::Long(offsets)));
    tags.push((278, V::Long(vec![rows_per_strip])));
    tags.push((279, V::Long(counts)));
    tags.sort_by_key(|t| t.0);
    // Values longer than 4 bytes go before the directory.
    let mut placed: Vec<(u16, u16, u32, Vec<u8>, bool)> = Vec::new();
    for (tag, v) in &tags {
        let (typ, count, len) = v.kind();
        let b = v.bytes();
        if len > 4 {
            let off = out.len() as u32;
            out.extend_from_slice(&b);
            if out.len() % 2 == 1 {
                out.push(0);
            }
            placed.push((*tag, typ, count as u32, off.to_le_bytes().to_vec(), true));
        } else {
            let mut inline = b;
            inline.resize(4, 0);
            placed.push((*tag, typ, count as u32, inline, false));
        }
    }
    let ifd = out.len() as u32;
    out[4..8].copy_from_slice(&ifd.to_le_bytes());
    out.extend_from_slice(&(placed.len() as u16).to_le_bytes());
    for (tag, typ, count, value, _) in &placed {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&typ.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(value);
    }
    out.extend_from_slice(&0u32.to_le_bytes());
    Ok(out)
}

fn geo_tags(grid: &Grid, geo: &GeoTags, tags: &mut Vec<(u16, V)>) {
    tags.push((33550, V::Double(vec![grid.dx, grid.dy, 0.0])));
    tags.push((33922, V::Double(vec![0.0, 0.0, 0.0, grid.x0, grid.y0, 0.0])));
    // The keys are copied with the raster type set to PixelIsArea (the tie point is a corner).
    let mut keys = geo.keys.clone();
    let mut i = 4;
    while i + 3 < keys.len() {
        if keys[i] == 1025 && keys[i + 1] == 0 {
            keys[i + 3] = 1;
        }
        i += 4;
    }
    if !keys.is_empty() {
        tags.push((34735, V::Short(keys)));
    }
    if !geo.doubles.is_empty() {
        tags.push((34736, V::Double(geo.doubles.clone())));
    }
    if !geo.ascii.is_empty() {
        tags.push((34737, V::Ascii(geo.ascii.trim_end_matches('\0').to_string())));
    }
}

const LEVEL: u8 = 4;

/// A one-band byte GeoTIFF whose values are classes: a palette image (its colours from
/// `colors`, by value) with the class names in GDAL's metadata, 0 as no-data.
pub fn write_classes(grid: &Grid, geo: &GeoTags, data: &[u8], colors: &[(u8, [u8; 3])], names: &[(u8, String)]) -> Result<Vec<u8>, String> {
    let rows = 64usize;
    let mut strips = Vec::new();
    for chunk in data.chunks(grid.w * rows) {
        strips.push(miniz_oxide::deflate::compress_to_vec_zlib(chunk, LEVEL));
    }
    let mut map = vec![0u16; 768];
    for (v, c) in colors {
        let i = *v as usize;
        map[i] = u16::from(c[0]) * 257;
        map[256 + i] = u16::from(c[1]) * 257;
        map[512 + i] = u16::from(c[2]) * 257;
    }
    let items: String = names.iter().map(|(v, n)| format!("<Item name=\"CLASS_{v}\">{}</Item>", xml(n))).collect();
    let mut tags = vec![
        (256, V::Long(vec![grid.w as u32])),
        (257, V::Long(vec![grid.h as u32])),
        (258, V::Short(vec![8])),
        (259, V::Short(vec![8])),
        (262, V::Short(vec![3])),
        (277, V::Short(vec![1])),
        (284, V::Short(vec![1])),
        (320, V::Short(map)),
        (339, V::Short(vec![1])),
        (42112, V::Ascii(format!("<GDALMetadata>{items}</GDALMetadata>"))),
        (42113, V::Ascii("0".into())),
    ];
    geo_tags(grid, geo, &mut tags);
    assemble(strips, tags, rows as u32)
}

/// The sample type a composite is written in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sample {
    U16,
    I16,
    F32,
}

/// A multi-band GeoTIFF, its bands stored apart (planar), with their names in GDAL's
/// metadata. NaN becomes `nodata`. Uncompressed: Deflate costs about 300 instructions a byte in
/// RustGIS's metered interpreter, more than a tool may spend on a scene.
pub fn write_bands(
    grid: &Grid,
    geo: &GeoTags,
    bands: Vec<Vec<f32>>,
    names: &[String],
    sample: Sample,
    nodata: f64,
) -> Result<Vec<u8>, String> {
    let rows = 32usize;
    let size = if sample == Sample::F32 { 4 } else { 2 };
    if bands.len() * grid.cells() * size > 64_000_000 {
        return Err(format!(
            "the composite would be {} MB, more than a plugin may hand RustGIS (64 MB): stack fewer bands, take the coarsest grid, or clip the rasters first (Analyze › Process › Clip)",
            bands.len() * grid.cells() * size / 1_000_000
        ));
    }
    let n = bands.len();
    let mut strips = Vec::new();
    for band in bands {
        for chunk in band.chunks(grid.w * rows) {
            let mut raw = vec![0u8; chunk.len() * size];
            match sample {
                Sample::U16 => {
                    for (o, v) in raw.as_chunks_mut::<2>().0.iter_mut().zip(chunk) {
                        o.copy_from_slice(&(if v.is_nan() { nodata as u16 } else { v.round().clamp(0.0, 65535.0) as u16 }).to_le_bytes());
                    }
                }
                Sample::I16 => {
                    for (o, v) in raw.as_chunks_mut::<2>().0.iter_mut().zip(chunk) {
                        o.copy_from_slice(
                            &(if v.is_nan() { nodata as i16 } else { v.round().clamp(-32768.0, 32767.0) as i16 }).to_le_bytes(),
                        );
                    }
                }
                Sample::F32 => {
                    for (o, v) in raw.as_chunks_mut::<4>().0.iter_mut().zip(chunk) {
                        o.copy_from_slice(&(if v.is_nan() { nodata as f32 } else { *v }).to_le_bytes());
                    }
                }
            }
            strips.push(raw);
        }
        // `band` is freed here, before the next is converted.
    }
    let (bits, fmt) = match sample {
        Sample::U16 => (16, 1),
        Sample::I16 => (16, 2),
        Sample::F32 => (32, 3),
    };
    let items: String = names
        .iter()
        .enumerate()
        .map(|(i, s)| format!("<Item name=\"DESCRIPTION\" sample=\"{i}\" role=\"description\">{}</Item>", xml(s)))
        .collect();
    let mut tags = vec![
        (256, V::Long(vec![grid.w as u32])),
        (257, V::Long(vec![grid.h as u32])),
        (258, V::Short(vec![bits; n])),
        (259, V::Short(vec![1])),
        (262, V::Short(vec![1])),
        (277, V::Short(vec![n as u16])),
        (284, V::Short(vec![2])),
        (339, V::Short(vec![fmt; n])),
        (42112, V::Ascii(format!("<GDALMetadata>{items}</GDALMetadata>"))),
        (42113, V::Ascii(format!("{nodata}"))),
    ];
    if n > 1 {
        tags.push((338, V::Short(vec![0; n - 1])));
    }
    geo_tags(grid, geo, &mut tags);
    assemble(strips, tags, rows as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_classes_read_back() {
        let grid = Grid { w: 70, h: 130, x0: 500000.0, y0: 9000000.0, dx: 10.0, dy: 10.0 };
        let geo = GeoTags { keys: vec![1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32750], doubles: vec![], ascii: String::new() };
        let data: Vec<u8> = (0..grid.cells()).map(|i| (i % 4) as u8).collect();
        let tif = write_classes(&grid, &geo, &data, &[(1, [0, 0, 255])], &[(1, "Water".into())]).unwrap();
        let img = read("c.tif", &tif).unwrap();
        assert!(img.grid.same(&grid));
        assert_eq!(img.bands.len(), 1);
        assert!(img.bands[0][0].is_nan()); // 0 is no-data
        assert_eq!(img.bands[0][5], 1.0);
        assert_eq!(img.crs, Crs::Utm { zone: 50, south: true });
    }

    #[test]
    fn written_bands_read_back() {
        let grid = Grid { w: 33, h: 41, x0: 106.0, y0: -6.0, dx: 0.001, dy: 0.001 };
        let geo = GeoTags { keys: vec![1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326], doubles: vec![], ascii: String::new() };
        let b1: Vec<f32> = (0..grid.cells()).map(|i| i as f32).collect();
        let mut b2: Vec<f32> = (0..grid.cells()).map(|i| (i * 2) as f32).collect();
        b2[3] = f32::NAN;
        let tif = write_bands(&grid, &geo, vec![b1, b2], &["B4".into(), "B8".into()], Sample::U16, 0.0).unwrap();
        let img = read("s.tif", &tif).unwrap();
        assert_eq!(img.bands.len(), 2);
        assert_eq!(img.bands[0][7], 7.0);
        assert_eq!(img.bands[1][7], 14.0);
        assert!(img.bands[1][3].is_nan());
        assert_eq!(img.crs, Crs::Geographic);
    }
}

#[cfg(test)]
mod real_files {
    /// The user's Sentinel-2 bands (LZW, tiled BigTIFF), when present: each band's sum and two
    /// of its samples as GDAL reads them (rasterio).
    #[test]
    fn sentinel2_bands_match_gdal() {
        let dir = "C:/Users/ASUS/Downloads/drive-download-20261105T034948Z-1-001/";
        for (band, sum, at, last) in
            [("B2", 1_755_261_973f64, 432f32, 553f32), ("B11", 1_093_267_809.0, 1360.0, 158.0), ("B1", 48_541_932.0, 547.0, 609.0)]
        {
            let Ok(bytes) = std::fs::read(format!("{dir}S2A_{band}.tif")) else { return };
            let img = super::read(band, &bytes).unwrap();
            let b = &img.bands[0];
            assert_eq!(b.iter().map(|v| *v as f64).sum::<f64>(), sum, "{band}");
            assert_eq!(b[100 * img.grid.w + 200], at, "{band}");
            assert_eq!(b[img.grid.cells() - 1], last, "{band}");
        }
    }
}

#[cfg(test)]
mod bench {
    #[test]
    #[ignore]
    fn time_b2() {
        let bytes = std::fs::read("C:/Users/ASUS/Downloads/drive-download-20261105T034948Z-1-001/S2A_B2.tif").unwrap();
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let _ = super::read("B2", &bytes).unwrap();
        }
        println!("B2 read: {:.1} ms", t.elapsed().as_secs_f64() * 100.0);
    }
}
