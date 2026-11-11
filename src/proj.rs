//! The coordinate systems a raster may be in, read from its GeoTIFF keys, and the forward
//! projection of longitude/latitude (WGS 84) into them. Training samples come from RustGIS in
//! degrees; the raster may be geographic, Web Mercator or Transverse Mercator (UTM, the
//! Indonesian TM-3 zones, or one its keys define).
use crate::tif::GeoTags;

#[derive(Clone, Debug, PartialEq)]
pub enum Crs {
    Geographic,
    WebMercator,
    Utm { zone: u8, south: bool },
    Tm { lon0: f64, lat0: f64, k0: f64, fe: f64, fn_: f64 },
}

/// A GeoTIFF key held in the directory itself (location 0).
pub fn geokey(keys: &[u16], id: u16) -> Option<u16> {
    let mut i = 4;
    while i + 3 < keys.len() {
        if keys[i] == id && keys[i + 1] == 0 {
            return Some(keys[i + 3]);
        }
        i += 4;
    }
    None
}

/// A GeoTIFF key held in the double parameters (location 34736).
fn geokey_double(geo: &GeoTags, id: u16) -> Option<f64> {
    let k = &geo.keys;
    let mut i = 4;
    while i + 3 < k.len() {
        if k[i] == id && k[i + 1] == 34736 {
            return geo.doubles.get(k[i + 3] as usize).copied();
        }
        i += 4;
    }
    None
}

fn from_epsg(code: u16) -> Option<Crs> {
    let c = code as u32;
    let utm = |zone: u32, south: bool| Some(Crs::Utm { zone: zone as u8, south });
    match c {
        32601..=32660 => utm(c - 32600, false),
        32701..=32760 => utm(c - 32700, true),
        // NAD83 / UTM 1N–23N, ETRS89 / UTM 28N–38N: GRS80, within millimetres of WGS 84's.
        26901..=26923 => utm(c - 26900, false),
        25828..=25838 => utm(c - 25800, false),
        // DGN95 / UTM 46N–52N and 47S–54S.
        23866..=23872 => utm(c - 23820, false),
        23877..=23884 => utm(c - 23830, true),
        // DGN95 / Indonesia TM-3 zones 46.2 to 54.1.
        23830..=23845 => Some(Crs::Tm { lon0: 94.5 + 3.0 * (c - 23830) as f64, lat0: 0.0, k0: 0.9999, fe: 200_000.0, fn_: 1_500_000.0 }),
        3857 | 3785 | 900913 | 102100 | 102113 => Some(Crs::WebMercator),
        _ => None,
    }
}

impl Crs {
    pub fn from_geokeys(geo: &GeoTags) -> Result<Crs, String> {
        let keys = &geo.keys;
        if keys.is_empty() {
            return Err("it has no coordinate system (no GeoTIFF keys)".into());
        }
        match geokey(keys, 1024) {
            Some(2) => return Ok(Crs::Geographic),
            Some(3) => return Err("it is in geocentric coordinates, which aren't supported".into()),
            _ => {}
        }
        let pcs = geokey(keys, 3072);
        if let Some(c) = pcs.and_then(from_epsg) {
            return Ok(c);
        }
        // A projection given by its EPSG code (UTM north 16001–16060, south 16101–16160).
        if let Some(p) = geokey(keys, 3074) {
            match p {
                16001..=16060 => return Ok(Crs::Utm { zone: (p - 16000) as u8, south: false }),
                16101..=16160 => return Ok(Crs::Utm { zone: (p - 16100) as u8, south: true }),
                _ => {}
            }
        }
        // A user-defined Transverse Mercator, by its parameters.
        if geokey(keys, 3075) == Some(1) {
            let lon0 = geokey_double(geo, 3080).or_else(|| geokey_double(geo, 3088));
            if let Some(lon0) = lon0 {
                return Ok(Crs::Tm {
                    lon0,
                    lat0: geokey_double(geo, 3081).or_else(|| geokey_double(geo, 3089)).unwrap_or(0.0),
                    k0: geokey_double(geo, 3092).unwrap_or(1.0),
                    fe: geokey_double(geo, 3082).or_else(|| geokey_double(geo, 3086)).unwrap_or(0.0),
                    fn_: geokey_double(geo, 3083).or_else(|| geokey_double(geo, 3087)).unwrap_or(0.0),
                });
            }
        }
        if geokey(keys, 1024) == Some(1) || pcs.is_some() {
            let code = pcs.map(|c| format!("EPSG:{c}")).unwrap_or_else(|| "a user-defined projection".into());
            return Err(format!(
                "it is in {code}, which this plugin can't place samples in. Use rasters in WGS 84 (EPSG:4326), Web Mercator or UTM, or reproject them first (Analyze › Project Raster)"
            ));
        }
        // No model type: GDAL treats a geographic code as geographic.
        if geokey(keys, 2048).is_some() {
            return Ok(Crs::Geographic);
        }
        Err("its coordinate system can't be read".into())
    }

    /// A short name for messages.
    pub fn describe(&self) -> String {
        match self {
            Crs::Geographic => "geographic (degrees)".into(),
            Crs::WebMercator => "Web Mercator".into(),
            Crs::Utm { zone, south } => format!("UTM zone {zone}{}", if *south { "S" } else { "N" }),
            Crs::Tm { lon0, .. } => format!("Transverse Mercator (central meridian {lon0}°)"),
        }
    }

    /// Whether its units are metres.
    pub fn metric(&self) -> bool {
        !matches!(self, Crs::Geographic)
    }

    /// Longitude/latitude in degrees to this system's coordinates.
    pub fn forward(&self, lon: f64, lat: f64) -> (f64, f64) {
        match *self {
            Crs::Geographic => (lon, lat),
            Crs::WebMercator => {
                const R: f64 = 6_378_137.0;
                let lat = lat.clamp(-85.06, 85.06);
                (R * lon.to_radians(), R * (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0).tan().ln())
            }
            Crs::Utm { zone, south } => {
                let lon0 = -183.0 + 6.0 * zone as f64;
                tm(lon, lat, lon0, 0.0, 0.9996, 500_000.0, if south { 10_000_000.0 } else { 0.0 })
            }
            Crs::Tm { lon0, lat0, k0, fe, fn_ } => tm(lon, lat, lon0, lat0, k0, fe, fn_),
        }
    }
}

/// Transverse Mercator on WGS 84 by Krüger's series (Karney 2011, eqs. 7–10, to n³: well under a
/// millimetre within a zone).
fn tm(lon: f64, lat: f64, lon0: f64, lat0: f64, k0: f64, fe: f64, fn_: f64) -> (f64, f64) {
    const A: f64 = 6_378_137.0;
    const F: f64 = 1.0 / 298.257_223_563;
    let n = F / (2.0 - F);
    let e = (F * (2.0 - F)).sqrt();
    let big_a = A / (1.0 + n) * (1.0 + n * n / 4.0 + n.powi(4) / 64.0);
    let alpha =
        [n / 2.0 - 2.0 * n * n / 3.0 + 5.0 * n.powi(3) / 16.0, 13.0 * n * n / 48.0 - 3.0 * n.powi(3) / 5.0, 61.0 * n.powi(3) / 240.0];
    let xi_eta = |lon: f64, lat: f64| {
        let phi = lat.to_radians();
        let l = (lon - lon0).to_radians();
        let s = phi.sin();
        let t = (s.atanh() - e * (e * s).atanh()).sinh();
        let xi1 = t.atan2(l.cos());
        let eta1 = (l.sin() / (1.0 + t * t).sqrt()).atanh();
        let mut xi = xi1;
        let mut eta = eta1;
        for (j, a) in alpha.iter().enumerate() {
            let k = 2.0 * (j + 1) as f64;
            xi += a * (k * xi1).sin() * (k * eta1).cosh();
            eta += a * (k * xi1).cos() * (k * eta1).sinh();
        }
        (xi, eta)
    };
    let (xi, eta) = xi_eta(lon, lat);
    let xi0 = if lat0 == 0.0 { 0.0 } else { xi_eta(lon0, lat0).0 };
    (fe + k0 * big_a * eta, fn_ + k0 * big_a * (xi - xi0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f64, f64), b: (f64, f64)) {
        assert!((a.0 - b.0).abs() < 0.01 && (a.1 - b.1).abs() < 0.01, "{a:?} != {b:?}");
    }

    #[test]
    fn matches_proj() {
        // Reference values from PROJ (pyproj).
        close(from_epsg(32652).unwrap().forward(127.9, 0.5), (377594.6009, 55275.2918));
        close(from_epsg(32748).unwrap().forward(107.6, -6.9), (787329.7978, 9236522.1495));
        close(from_epsg(23834).unwrap().forward(107.6, -6.9), (321565.5835, 736936.6581));
        close(from_epsg(3857).unwrap().forward(107.6, -6.9), (11977977.2094, -769967.8620));
        close(from_epsg(32633).unwrap().forward(15.0, 60.0), (500000.0, 6651411.1904));
        close(from_epsg(32610).unwrap().forward(-123.5, 45.2), (460729.6926, 5005289.8417));
    }

    #[test]
    fn reads_keys() {
        let g = |keys: Vec<u16>| GeoTags { keys, doubles: vec![], ascii: String::new() };
        assert_eq!(Crs::from_geokeys(&g(vec![1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326])).unwrap(), Crs::Geographic);
        assert_eq!(Crs::from_geokeys(&g(vec![1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32750])).unwrap(), Crs::Utm { zone: 50, south: true });
        assert!(Crs::from_geokeys(&g(vec![1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 2154])).unwrap_err().contains("EPSG:2154"));
    }
}
