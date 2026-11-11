//! Classify: training samples → model → accuracy → classified map.
//!
//! A RustGIS tool run may spend 30 billion units of fuel (about as many WebAssembly
//! instructions). Reading a scene and running a forest over every cell can take more, so the
//! work is split into passes: the first reads only the tiles under the samples, trains the
//! model, assesses it and classifies the rows it has budget for; when rows remain it leaves a
//! *job* layer (the model, the rasters, the rows done) that `add-results` hands back to this
//! tool, pass after pass, until the map is whole.
use crate::metrics::Confusion;
use crate::sample::{self, ClassInfo, hex, parse_color};
use crate::stack::{self, GridChoice, Resampling, Stack};
use crate::tif::{self, Grid, Opened, Rect, Src};
use crate::{ml, num, pct, round, table, text};
use rustgis_plugin::{Request, ToolOutput, Value, geojson, json};

/// What one pass plans to spend, of a tool's 30 billion.
const PASS_BUDGET: f64 = 25e9;
/// The most passes a classification may take.
const MAX_PASSES: usize = 40;
/// The most samples one pair of classes trains an SVM on (a random choice beyond it).
const SVM_PAIR: usize = 1000;
/// The field a job layer is known by.
pub const JOB: &str = "lcc_job";

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Algorithm {
    RandomForest,
    Svm,
    Cart,
    Knn,
    Bayes,
    MinDist,
    Mlc,
}

impl Algorithm {
    pub fn parse(s: &str) -> Algorithm {
        let s = s.to_ascii_lowercase();
        if s.contains("support") || s.starts_with("svm") {
            Algorithm::Svm
        } else if s.contains("cart") || s.contains("decision") {
            Algorithm::Cart
        } else if s.contains("neighbo") || s.contains("knn") {
            Algorithm::Knn
        } else if s.contains("bayes") {
            Algorithm::Bayes
        } else if s.contains("minimum") {
            Algorithm::MinDist
        } else if s.contains("likelihood") {
            Algorithm::Mlc
        } else {
            Algorithm::RandomForest
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Algorithm::RandomForest => "Random Forest",
            Algorithm::Svm => "Support Vector Machine",
            Algorithm::Cart => "CART",
            Algorithm::Knn => "k-Nearest Neighbors",
            Algorithm::Bayes => "Naive Bayes",
            Algorithm::MinDist => "Minimum Distance",
            Algorithm::Mlc => "Maximum Likelihood",
        }
    }
}

/// The classifier's settings, as the job keeps them.
#[derive(Clone, Debug)]
struct Settings {
    algorithm: Algorithm,
    trees: usize,
    mtry: usize,
    min_leaf: usize,
    max_depth: usize,
    linear: bool,
    cost: f64,
    gamma: f64,
    k: usize,
    seed: u64,
}

impl Settings {
    fn from_request(req: &Request) -> Settings {
        Settings {
            algorithm: Algorithm::parse(req.str("algorithm").unwrap_or("Random Forest")),
            trees: num(req, "trees", 100.0).clamp(1.0, 1000.0) as usize,
            mtry: num(req, "variablesPerSplit", 0.0).max(0.0) as usize,
            min_leaf: num(req, "minLeaf", 1.0).max(1.0) as usize,
            max_depth: num(req, "maxDepth", 0.0).max(0.0) as usize,
            linear: req.str("svmKernel").unwrap_or("").to_ascii_lowercase().starts_with("lin"),
            cost: num(req, "svmCost", 10.0).max(1e-6),
            gamma: num(req, "svmGamma", 0.0).max(0.0),
            k: num(req, "k", 5.0).max(1.0) as usize,
            seed: num(req, "seed", 42.0).max(0.0) as u64,
        }
    }

    fn to_json(&self) -> Value {
        json!({"algorithm": self.algorithm.name(), "trees": self.trees, "mtry": self.mtry, "minLeaf": self.min_leaf, "maxDepth": self.max_depth,
               "linear": self.linear, "cost": self.cost, "gamma": self.gamma, "k": self.k, "seed": self.seed})
    }

    fn from_json(v: &Value) -> Settings {
        let n = |k: &str, d: f64| v[k].as_f64().unwrap_or(d);
        Settings {
            algorithm: Algorithm::parse(v["algorithm"].as_str().unwrap_or("")),
            trees: n("trees", 100.0) as usize,
            mtry: n("mtry", 0.0) as usize,
            min_leaf: n("minLeaf", 1.0) as usize,
            max_depth: n("maxDepth", 0.0) as usize,
            linear: v["linear"].as_bool().unwrap_or(false),
            cost: n("cost", 10.0),
            gamma: n("gamma", 0.0),
            k: n("k", 5.0) as usize,
            seed: n("seed", 42.0) as u64,
        }
    }

    fn mtry_for(&self, d: usize) -> usize {
        if self.algorithm != Algorithm::RandomForest {
            d
        } else if self.mtry == 0 {
            ((d as f64).sqrt().floor() as usize).max(1)
        } else {
            self.mtry.min(d)
        }
    }

    fn describe(&self, d: usize, model: &ml::Model) -> String {
        let depth = if self.max_depth == 0 { "unlimited".to_string() } else { self.max_depth.to_string() };
        match model {
            ml::Model::Forest(f) if self.algorithm == Algorithm::RandomForest => format!(
                "{} trees, {} variables per split, min leaf {}, max depth {depth}, mean depth {:.1}, seed {}",
                f.trees(),
                self.mtry_for(d),
                self.min_leaf,
                f.depth,
                self.seed
            ),
            ml::Model::Forest(f) => format!("{} nodes, min leaf {}, max depth {depth}", f.nodes(), self.min_leaf),
            ml::Model::Svm(m) => format!(
                "{} kernel, C {}, gamma {}, {} support vectors",
                if self.linear { "linear" } else { "RBF" },
                self.cost,
                if self.gamma > 0.0 { self.gamma.to_string() } else { format!("1/{d} (auto)") },
                m.support_vectors()
            ),
            ml::Model::Knn(_) => format!("k = {} (bands standardised)", self.k),
            _ => String::new(),
        }
    }
}

fn train(data: &ml::Data, s: &Settings) -> ml::Model {
    let d = data.d;
    match s.algorithm {
        Algorithm::RandomForest | Algorithm::Cart => {
            let rf = s.algorithm == Algorithm::RandomForest;
            ml::Model::Forest(ml::forest(
                data,
                &ml::ForestParams {
                    trees: if rf { s.trees } else { 1 },
                    tree: ml::TreeParams { mtry: s.mtry_for(d), min_leaf: s.min_leaf, max_depth: s.max_depth },
                    bootstrap: rf,
                    seed: s.seed,
                },
            ))
        }
        Algorithm::Svm => {
            let gamma = if s.gamma > 0.0 { s.gamma } else { 1.0 / d as f64 } as f32;
            let kernel = if s.linear { ml::Kernel::Linear } else { ml::Kernel::Rbf(gamma) };
            ml::Model::Svm(ml::svm(data, &ml::SvmParams { kernel, c: s.cost as f32, max_pair: SVM_PAIR, seed: s.seed }))
        }
        Algorithm::Knn => ml::Model::Knn(ml::knn(data, s.k, s.seed)),
        Algorithm::Bayes => ml::Model::Bayes(ml::bayes(data)),
        Algorithm::MinDist => ml::Model::MinDist(ml::min_dist(data)),
        Algorithm::Mlc => ml::Model::Mlc(ml::mlc(data)),
    }
}

/// About what training costs, in fuel (measured on Sentinel-2 samples).
fn train_cost(s: &Settings, n: usize, d: usize, k: usize) -> f64 {
    let n = n as f64;
    match s.algorithm {
        Algorithm::RandomForest => s.trees as f64 * n * 8000.0 * (s.mtry_for(d) as f64 / 3.0).max(0.5),
        Algorithm::Cart => n * 8000.0 * d as f64 / 3.0,
        Algorithm::Svm => {
            let m = (2.0 * n / k as f64).min(SVM_PAIR as f64);
            (k * (k - 1) / 2) as f64 * m * m * (d as f64 * 30.0 + 400.0)
        }
        _ => n * d as f64 * 200.0,
    }
}

/// A cell's area in hectares (for degrees, at latitude `lat`).
fn cell_area(grid: &Grid, metric: bool, lat: f64) -> f64 {
    if metric {
        grid.dx * grid.dy / 10_000.0
    } else {
        let m = 111_320.0;
        grid.dx * m * lat.to_radians().cos() * grid.dy * m / 10_000.0
    }
}

/// The rasters a run reads: their names and base64 contents (decoded only where needed).
struct Files {
    list: Vec<(String, String)>,
}

impl Files {
    fn from_request(req: &mut Request) -> Result<Files, String> {
        let mut list = Vec::new();
        for i in 1..=12 {
            let p = format!("raster{i}");
            let Some(mut f) = req.files.remove(&p) else { continue };
            let name = f.get("name").and_then(Value::as_str).unwrap_or("raster.tif").to_string();
            let Some(Value::String(data)) = f.get_mut("data").map(Value::take) else {
                return Err(format!("{name}: the file has no contents"));
            };
            list.push((name, data));
        }
        if list.is_empty() {
            return Err("give at least one raster (a GeoTIFF of one or more bands)".into());
        }
        Ok(Files { list })
    }

    fn open(&self) -> Result<Vec<Opened<'_>>, String> {
        let opened: Vec<Opened> = self.list.iter().map(|(n, d)| tif::open(n, Src::B64(d.as_bytes()))).collect::<Result<_, _>>()?;
        if let Some(o) = opened.iter().skip(1).find(|o| o.crs != opened[0].crs) {
            return Err(format!(
                "{} is in {} but {} is in {}: give rasters in one coordinate system (reproject with Analyze › Project Raster)",
                o.name,
                o.crs.describe(),
                opened[0].name,
                opened[0].crs.describe()
            ));
        }
        Ok(opened)
    }

    fn bytes(&self) -> usize {
        self.list.iter().map(|f| f.1.len()).sum()
    }
}

/// The reference grid of `opened` and its index.
fn reference(opened: &[Opened], choice: GridChoice) -> (usize, Grid) {
    let i = stack::pick(&opened.iter().map(|o| &o.grid).collect::<Vec<_>>(), choice);
    (i, opened[i].grid.clone())
}

/// The bands stacked on the reference grid, read and resampled only in `regions`.
fn stack_in(opened: &[Opened], choice: GridChoice, grid: &Grid, regions: &[Rect]) -> Result<Stack, String> {
    let images: Vec<tif::Image> = opened.iter().map(|o| o.image(&stack::map_rects(regions, grid, &o.grid))).collect::<Result<_, _>>()?;
    stack::composite_in(images, choice, Resampling::Nearest, Some(regions))
}

/// About what reading `regions` of the reference grid costs.
fn read_cost(opened: &[Opened], grid: &Grid, regions: &[Rect]) -> f64 {
    opened.iter().map(|o| o.cost(&stack::map_rects(regions, grid, &o.grid))).sum::<f64>()
        + regions.iter().map(|r| ((r.2 - r.0) * (r.3 - r.1)) as f64).sum::<f64>() * 15.0 * opened.len() as f64
}

/// What predicting one cell costs (calibrated on the harness: a forest's own estimate is low).
fn cell_cost(model: &ml::Model, d: usize, k: usize) -> f64 {
    (model.cost(d, k) + 60.0 * d as f64 + 80.0) * 1.15
}

/// How many rows from `r0` fit in `budget`.
fn rows_for(opened: &[Opened], grid: &Grid, per_cell: f64, r0: usize, budget: f64) -> usize {
    let mut r1 = r0;
    // Grow by blocks of rows until the budget is spent.
    let step = 32.max(grid.h / 200);
    while r1 < grid.h {
        let next = (r1 + step).min(grid.h);
        let rect = [(0, r0, grid.w, next)];
        let cost = read_cost(opened, grid, &rect) + (grid.w * (next - r0)) as f64 * per_cell;
        if cost > budget {
            break;
        }
        r1 = next;
    }
    r1
}

/// Classifies rows `r0..r1`, adding to the class counts and areas.
#[allow(clippy::too_many_arguments)]
fn predict_rows(
    opened: &[Opened],
    choice: GridChoice,
    grid: &Grid,
    model: &ml::Model,
    values: &[u8],
    r0: usize,
    r1: usize,
    count: &mut [u64],
    area: &mut [f64],
) -> Result<Vec<u8>, String> {
    let st = stack_in(opened, choice, grid, &[(0, r0, grid.w, r1)])?;
    let d = st.bands.len();
    let k = values.len();
    let mut scratch = ml::Scratch::new(d, k);
    let mut px = vec![0f32; d];
    let mut out = vec![0u8; grid.w * (r1 - r0)];
    let metric = st.crs.metric();
    for r in r0..r1 {
        let a = cell_area(grid, metric, if metric { 0.0 } else { grid.center(0, r).1 });
        for c in 0..grid.w {
            let i = r * grid.w + c;
            if st.pixel(i, &mut px) {
                let p = model.predict(&px, &mut scratch) as usize;
                out[(r - r0) * grid.w + c] = values[p];
                count[p] += 1;
                area[p] += a;
            }
        }
    }
    Ok(out)
}

fn subset(s: &sample::Samples, idx: &[usize]) -> (Vec<f32>, Vec<u16>) {
    let mut x = Vec::with_capacity(idx.len() * s.d);
    let mut y = Vec::with_capacity(idx.len());
    for &i in idx {
        x.extend_from_slice(&s.x[i * s.d..(i + 1) * s.d]);
        y.push(s.y[i]);
    }
    (x, y)
}

/// The training features of `trainingPoints` and `trainingPolygons` (a tool's layer parameters).
fn training_features(req: &Request) -> Result<Vec<Value>, String> {
    let mut features = Vec::new();
    let mut named = false;
    for p in ["trainingPoints", "trainingPolygons", "training"] {
        if text(req, p).is_some() {
            named = true;
            features.extend(req.features(p)?.iter().cloned());
        }
    }
    if !named {
        return Err(format!(
            "give the training samples: a point layer, a polygon layer or both (the panel draws them into '{}' and '{}')",
            crate::POINTS,
            crate::POLYGONS
        ));
    }
    if features.is_empty() {
        return Err("the training layers have no features".into());
    }
    Ok(features)
}

fn label_field(c: &ClassInfo) -> String {
    format!("{} {}", c.value, c.label)
}

pub(crate) fn confusion_table(cm: &Confusion, classes: &[ClassInfo]) -> Value {
    let k = cm.k;
    let mut rows = Vec::new();
    let head = "Reference \\ Classified";
    for a in 0..k {
        let mut r: Vec<(String, Value)> = vec![(head.into(), json!(label_field(&classes[a])))];
        for p in 0..k {
            r.push((label_field(&classes[p]), json!(cm.m[a * k + p] as f64)));
        }
        r.push(("Total".into(), json!(cm.row(a) as f64)));
        r.push(("Producer's accuracy (%)".into(), round(cm.producers(a) * 100.0, 2)));
        rows.push(r);
    }
    let mut total: Vec<(String, Value)> = vec![(head.into(), json!("Total"))];
    for p in 0..k {
        total.push((label_field(&classes[p]), json!(cm.col(p) as f64)));
    }
    total.push(("Total".into(), json!(cm.total() as f64)));
    total.push(("Producer's accuracy (%)".into(), Value::Null));
    rows.push(total);
    let mut users: Vec<(String, Value)> = vec![(head.into(), json!("User's accuracy (%)"))];
    for p in 0..k {
        users.push((label_field(&classes[p]), round(cm.users(p) * 100.0, 2)));
    }
    users.push(("Total".into(), Value::Null));
    users.push(("Producer's accuracy (%)".into(), round(cm.overall() * 100.0, 2)));
    rows.push(users);
    table(rows)
}

fn class_table(cm: &Confusion, s: &sample::Samples, train: &[usize], test: &[usize]) -> Value {
    let rows = s
        .classes
        .iter()
        .enumerate()
        .map(|(c, info)| {
            vec![
                ("Value".into(), json!(info.value)),
                ("Class".into(), json!(info.label)),
                ("Colour".into(), json!(hex(info.color))),
                ("Samples".into(), json!(s.features[c])),
                ("Training pixels".into(), json!(train.iter().filter(|&&i| s.y[i] as usize == c).count())),
                ("Test pixels".into(), json!(test.iter().filter(|&&i| s.y[i] as usize == c).count())),
                ("Producer's accuracy (%)".into(), round(cm.producers(c) * 100.0, 2)),
                ("User's accuracy (%)".into(), round(cm.users(c) * 100.0, 2)),
                ("F1 score".into(), round(cm.f1(c), 4)),
                ("Omission error (%)".into(), round((1.0 - cm.producers(c)) * 100.0, 2)),
                ("Commission error (%)".into(), round((1.0 - cm.users(c)) * 100.0, 2)),
            ]
        })
        .collect();
    table(rows)
}

fn area_table(classes: &[ClassInfo], count: &[u64], area: &[f64]) -> Value {
    let total: f64 = area.iter().sum();
    let rows = classes
        .iter()
        .enumerate()
        .map(|(c, info)| {
            vec![
                ("Value".into(), json!(info.value)),
                ("Class".into(), json!(info.label)),
                ("Cells".into(), json!(count[c] as f64)),
                ("Area (ha)".into(), round(area[c], 2)),
                ("Area (km²)".into(), round(area[c] / 100.0, 4)),
                ("Area (%)".into(), round(if total > 0.0 { area[c] / total * 100.0 } else { f64::NAN }, 2)),
            ]
        })
        .collect();
    table(rows)
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16s(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The point a result layer sits at: somewhere inside the reference grid, in degrees.
fn anchor(grid: &Grid, crs: &crate::proj::Crs) -> [f64; 2] {
    let st = Stack {
        grid: grid.clone(),
        geo: Default::default(),
        crs: crs.clone(),
        bands: vec![],
        names: vec![],
        sample: tif::Sample::U16,
        notes: vec![],
    };
    crate::centre_degrees(&st).unwrap_or([0.0, 0.0])
}

// ---- The first pass --------------------------------------------------------------------------

pub fn classify_tool(req: &mut Request) -> Result<Value, String> {
    if text(req, "job").is_some() {
        return continue_job(req);
    }
    let files = Files::from_request(req)?;
    let opened = files.open()?;
    let choice = GridChoice::parse(req.str("grid"));
    let (_, grid) = reference(&opened, choice);
    let crs = opened[0].crs.clone();
    let features = training_features(req)?;
    let refs: Vec<&Value> = features.iter().collect();
    let class_field = text(req, "classField").unwrap_or("class").to_string();
    let s = Settings::from_request(req);
    let max_pixels = num(req, "maxPixels", 500.0).max(1.0) as usize;

    // Only the tiles under the samples are read for training.
    let regions = sample::regions(&refs, &grid, &crs);
    if regions.is_empty() {
        return Err("none of the training samples lies on the rasters: draw them over the image".into());
    }
    let st = stack_in(&opened, choice, &grid, &regions)?;
    let samples = sample::collect(&st, &refs, &class_field, max_pixels, s.seed)?;
    let band_names = st.names.clone();
    let stack_notes = st.notes.clone();
    drop(st);
    drop(features);

    let ratio = num(req, "splitRatio", 0.7);
    let ratio = if ratio > 1.0 { ratio / 100.0 } else { ratio }.clamp(0.05, 1.0);
    let by_feature = req.str("splitBy").unwrap_or("").to_ascii_lowercase().starts_with("sample");
    let (train_idx, test_idx, split_notes) = sample::split(&samples, ratio, by_feature, s.seed);
    let (d, k) = (samples.d, samples.classes.len());
    let (tx, ty) = subset(&samples, &train_idx);
    for c in 0..k {
        if !ty.iter().any(|&v| v as usize == c) {
            return Err(format!("{} has no training pixels: draw more samples of it", samples.classes[c].label));
        }
    }
    let data = ml::Data { x: &tx, y: &ty, d, k };
    let model = train(&data, &s);

    let mut scratch = ml::Scratch::new(d, k);
    let mut test_cm = Confusion::new(k);
    for &i in &test_idx {
        test_cm.add(samples.y[i] as usize, model.predict(&samples.x[i * d..(i + 1) * d], &mut scratch) as usize);
    }
    let mut train_cm = Confusion::new(k);
    for i in 0..ty.len() {
        train_cm.add(ty[i] as usize, model.predict(&tx[i * d..(i + 1) * d], &mut scratch) as usize);
    }

    // The passes the map takes.
    let per_cell = cell_cost(&model, d, k);
    let whole = [(0, 0, grid.w, grid.h)];
    let total = read_cost(&opened, &grid, &whole) + grid.cells() as f64 * per_cell;
    let spent = 1.5e9 + files.bytes() as f64 * 12.0 + read_cost(&opened, &grid, &regions) + train_cost(&s, ty.len(), d, k);
    let later = PASS_BUDGET
        - 1.5e9
        - files.bytes() as f64 * 14.0
        - if matches!(model, ml::Model::Forest(_) | ml::Model::Svm(_)) { 0.0 } else { train_cost(&s, ty.len(), d, k) };
    if later < PASS_BUDGET * 0.3 {
        return Err(
            "the rasters are too large for a plugin to read in one go: clip them to the area you need (Analyze › Process › Clip)".into()
        );
    }
    let passes = 1 + ((total - (PASS_BUDGET - spent).max(0.0)).max(0.0) / later).ceil() as usize;
    if passes > MAX_PASSES {
        let advice = match s.algorithm {
            Algorithm::RandomForest => format!("fewer trees (about {})", ((s.trees * MAX_PASSES) as f64 / passes as f64).floor().max(1.0)),
            Algorithm::Svm => "a linear kernel, fewer samples (Max pixels per sample) or a larger C".into(),
            Algorithm::Knn => "fewer samples (Max pixels per sample)".into(),
            _ => "fewer bands".into(),
        };
        return Err(format!(
            "classifying {} × {} cells with this {} would take about {passes} runs of the tool, more than {MAX_PASSES}. Try {advice}, the coarsest grid, or clip the rasters to the area you need (Analyze › Process › Clip).",
            grid.w,
            grid.h,
            s.algorithm.name()
        ));
    }
    let r1 = rows_for(&opened, &grid, per_cell, 0, PASS_BUDGET - spent);
    let mut count = vec![0u64; k];
    let mut area = vec![0f64; k];
    let values: Vec<u8> = samples.classes.iter().map(|c| c.value).collect();
    let strip = if r1 > 0 { predict_rows(&opened, choice, &grid, &model, &values, 0, r1, &mut count, &mut area)? } else { Vec::new() };

    // The accuracy report.
    let name = text(req, "name").map(str::to_string).unwrap_or_else(|| format!("Land Cover ({})", s.algorithm.name()));
    let has_test = !test_idx.is_empty();
    let mut out = ToolOutput::new();
    for n in stack_notes.iter().chain(&samples.notes).chain(&split_notes).chain(&model.notes()) {
        out = out.message(n.clone());
    }
    let mut summary: Vec<(String, String)> = vec![
        ("Algorithm".into(), s.algorithm.name().into()),
        ("Bands".into(), band_names.join(", ")),
        ("Grid".into(), format!("{} × {} cells of {} ({})", grid.w, grid.h, stack::cell(&grid, &crs), crs.describe())),
        ("Classes".into(), k.to_string()),
        ("Training pixels".into(), train_idx.len().to_string()),
        ("Test pixels".into(), test_idx.len().to_string()),
        (
            "Split".into(),
            format!(
                "{:.0}% training / {:.0}% test, by {}",
                ratio * 100.0,
                (1.0 - ratio) * 100.0,
                if by_feature { "sample" } else { "pixel" }
            ),
        ),
    ];
    let desc = s.describe(d, &model);
    if !desc.is_empty() {
        summary.push(("Parameters".into(), desc));
    }
    if has_test {
        summary.push(("Overall accuracy (test)".into(), format!("{}%", pct(test_cm.overall()))));
        summary.push(("Kappa (test)".into(), format!("{:.4}", test_cm.kappa())));
    }
    summary.push(("Overall accuracy (training)".into(), format!("{}%", pct(train_cm.overall()))));
    summary.push(("Kappa (training)".into(), format!("{:.4}", train_cm.kappa())));
    if let ml::Model::Forest(f) = &model
        && let Some(oob) = f.oob
    {
        summary.push(("Out-of-bag accuracy".into(), format!("{}%", pct(oob))));
    }
    out = out.table(
        &format!("{name} – Accuracy"),
        table(summary.iter().map(|(k, v)| vec![("Metric".into(), json!(k)), ("Value".into(), json!(v))]).collect()),
    );
    let cm = if has_test { &test_cm } else { &train_cm };
    out = out.table(
        &format!("{name} – Confusion Matrix ({})", if has_test { "test" } else { "training" }),
        confusion_table(cm, &samples.classes),
    );
    out = out.table(&format!("{name} – Class Accuracy"), class_table(cm, &samples, &train_idx, &test_idx));
    if let ml::Model::Forest(f) = &model
        && s.algorithm == Algorithm::RandomForest
    {
        let mut imp: Vec<(usize, f64)> = f.importance.iter().copied().enumerate().collect();
        imp.sort_by(|a, b| b.1.total_cmp(&a.1));
        out = out.table(
            &format!("{name} – Variable Importance"),
            table(imp.iter().map(|(b, v)| vec![("Band".into(), json!(band_names[*b])), ("Importance (%)".into(), round(*v, 2))]).collect()),
        );
    }
    let mut report = format!("{name}\n{} · {} training / {} test pixels", s.algorithm.name(), train_idx.len(), test_idx.len());
    if has_test {
        report += &format!("\nOverall accuracy {}% · Kappa {:.3} (test)", pct(test_cm.overall()), test_cm.kappa());
    }
    report += &format!("\nOverall accuracy {}% · Kappa {:.3} (training)", pct(train_cm.overall()), train_cm.kappa());
    if let ml::Model::Forest(f) = &model
        && let Some(oob) = f.oob
    {
        report += &format!("\nOut-of-bag accuracy {}%", pct(oob));
    }
    for (c, info) in samples.classes.iter().enumerate() {
        report += &format!("\n{}: PA {}% · UA {}%", info.label, pct(cm.producers(c)), pct(cm.users(c)));
    }
    out = out.message(report.replace('\n', " | "));

    let job = JobState {
        name,
        report,
        settings: s.clone(),
        choice,
        classes: samples.classes.clone(),
        names: band_names,
        rows: r1,
        count,
        area,
        d,
    };
    let at = anchor(&grid, &crs);
    if r1 >= grid.h {
        return finish(out, &opened, &grid, &job, &strip, at);
    }
    let model_bytes = match &model {
        ml::Model::Forest(f) => Some([b"F".as_slice(), &f.to_bytes()].concat()),
        ml::Model::Svm(m) => Some([b"S".as_slice(), &m.to_bytes()].concat()),
        _ => None,
    };
    let train_bytes = (f32s(&tx), u16s(&ty));
    out = out.message(format!(
        "Classified {:.0}% of the map. Run Land Cover › Add Results to Map to finish it (about {} more run(s) of this tool, each up to a minute).",
        r1 as f64 / grid.h as f64 * 100.0,
        passes.saturating_sub(1).max(1)
    ));
    Ok(out.layer(&job.layer_name(grid.h), job.layer(at, &files, model_bytes.as_deref(), &train_bytes, &strip)).into())
}

// ---- Jobs ------------------------------------------------------------------------------------

struct JobState {
    name: String,
    report: String,
    settings: Settings,
    choice: GridChoice,
    classes: Vec<ClassInfo>,
    names: Vec<String>,
    rows: usize,
    count: Vec<u64>,
    area: Vec<f64>,
    d: usize,
}

impl JobState {
    /// The job layer's name, which says how far it is (a command reads it without the layer's
    /// contents, which are too large for a command's budget).
    fn layer_name(&self, h: usize) -> String {
        format!("{} (classifying, {}% done)", self.name, self.rows * 100 / h.max(1))
    }

    fn to_json(&self) -> Value {
        json!({
            "v": 1, "name": self.name, "report": self.report, "settings": self.settings.to_json(),
            "grid": match self.choice { GridChoice::Finest => "Finest raster", GridChoice::Coarsest => "Coarsest raster", GridChoice::First => "First raster" },
            "classes": self.classes.iter().map(|c| json!({"value": c.value, "label": c.label, "color": hex(c.color)})).collect::<Vec<_>>(),
            "names": self.names, "rows": self.rows, "count": self.count, "area": self.area, "d": self.d,
        })
    }

    fn from_json(v: &Value) -> Result<JobState, String> {
        if v["v"].as_u64() != Some(1) {
            return Err("this classification job was made by another version of the plugin: run Classify again".into());
        }
        let classes: Vec<ClassInfo> = v["classes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| {
                Some(ClassInfo {
                    value: c["value"].as_u64()? as u8,
                    label: c["label"].as_str()?.to_string(),
                    color: c["color"].as_str().and_then(parse_color).unwrap_or([128, 128, 128]),
                })
            })
            .collect();
        let k = classes.len();
        let count: Vec<u64> = v["count"].as_array().into_iter().flatten().filter_map(Value::as_u64).collect();
        let area: Vec<f64> = v["area"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
        if k < 2 || count.len() != k || area.len() != k {
            return Err("the classification job is damaged: run Classify again".into());
        }
        Ok(JobState {
            name: v["name"].as_str().unwrap_or("Land Cover").to_string(),
            report: v["report"].as_str().unwrap_or("").to_string(),
            settings: Settings::from_json(&v["settings"]),
            choice: GridChoice::parse(v["grid"].as_str()),
            classes,
            names: v["names"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect(),
            rows: v["rows"].as_u64().unwrap_or(0) as usize,
            count,
            area,
            d: v["d"].as_u64().unwrap_or(0) as usize,
        })
    }

    /// The job as a one-feature layer: its state, the rasters, the model and the rows done.
    fn layer(&self, at: [f64; 2], files: &Files, model: Option<&[u8]>, train: &(Vec<u8>, Vec<u8>), strip: &[u8]) -> Value {
        let mut props = rustgis_plugin::Map::new();
        props.insert(JOB.into(), json!(self.to_json().to_string()));
        props.insert("lcc_progress".into(), json!(format!("{} rows of the map classified", self.rows)));
        if let Some(m) = model {
            props.insert("lcc_model".into(), json!(crate::base64(m)));
        }
        props.insert("lcc_train_x".into(), json!(crate::base64(&train.0)));
        props.insert("lcc_train_y".into(), json!(crate::base64(&train.1)));
        props.insert("lcc_strip".into(), json!(crate::base64(strip)));
        for (i, (n, d)) in files.list.iter().enumerate() {
            props.insert(format!("lcc_name{}", i + 1), json!(n));
            props.insert(format!("lcc_file{}", i + 1), json!(crate::splice(d.clone())));
        }
        geojson::collection(vec![geojson::feature(geojson::point(at[0], at[1]), Value::Object(props))])
    }
}

fn b64_prop(p: &Value, key: &str) -> Result<Vec<u8>, String> {
    let s = p[key].as_str().ok_or_else(|| format!("the classification job has no {key}: run Classify again"))?;
    crate::decode_base64(s.as_bytes().to_vec()).ok_or_else(|| format!("the classification job's {key} is damaged"))
}

/// A later pass: the next rows of the map, and the map itself once they are all done.
fn continue_job(req: &mut Request) -> Result<Value, String> {
    let feature = req.features("job")?.first().cloned().ok_or("the classification job layer is empty")?;
    req.layers.clear();
    let p = &feature["properties"];
    let mut job = JobState::from_json(
        &serde_json::from_str(p[JOB].as_str().ok_or("this layer isn't a classification job")?).map_err(|e| e.to_string())?,
    )?;
    let mut list = Vec::new();
    for i in 1..=12 {
        if let (Some(n), Some(d)) = (p[format!("lcc_name{i}")].as_str(), p[format!("lcc_file{i}")].as_str()) {
            list.push((n.to_string(), d.to_string()));
        }
    }
    let files = Files { list };
    let opened = files.open()?;
    let (_, grid) = reference(&opened, job.choice);
    let crs = opened[0].crs.clone();
    let k = job.classes.len();
    let d = job.d;
    let damaged = "the classification job's model is damaged: run Classify again";
    let model = match p.get("lcc_model").and_then(Value::as_str) {
        Some(_) => {
            let b = b64_prop(p, "lcc_model")?;
            match b.split_first() {
                Some((b'F', rest)) => ml::Model::Forest(ml::Forest::from_bytes(rest, d).ok_or(damaged)?),
                Some((b'S', rest)) => ml::Model::Svm(ml::Svm::from_bytes(rest).filter(|m| m.dims() == d).ok_or(damaged)?),
                _ => return Err(damaged.into()),
            }
        }
        None => {
            let xb = b64_prop(p, "lcc_train_x")?;
            let yb = b64_prop(p, "lcc_train_y")?;
            let x: Vec<f32> = xb.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let y: Vec<u16> = yb.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            if x.len() != y.len() * d || y.iter().any(|&c| c as usize >= k) {
                return Err("the classification job's samples are damaged: run Classify again".into());
            }
            train(&ml::Data { x: &x, y: &y, d, k }, &job.settings)
        }
    };
    let mut strip = b64_prop(p, "lcc_strip")?;
    let model_bytes = p.get("lcc_model").and_then(Value::as_str).map(|s| s.to_string());
    let train_x = p["lcc_train_x"].as_str().unwrap_or_default().to_string();
    let train_y = p["lcc_train_y"].as_str().unwrap_or_default().to_string();
    drop(feature);
    if strip.len() != job.rows * grid.w || job.rows >= grid.h {
        return Err("the classification job is damaged: run Classify again".into());
    }
    let per_cell = cell_cost(&model, d, k);
    let overhead = 1.5e9
        + files.bytes() as f64 * 14.0
        + if matches!(model, ml::Model::Forest(_) | ml::Model::Svm(_)) {
            0.0
        } else {
            train_cost(&job.settings, train_y.len() * 3 / 4 / 2, d, k)
        };
    let r0 = job.rows;
    let r1 = rows_for(&opened, &grid, per_cell, r0, PASS_BUDGET - overhead).max((r0 + 1).min(grid.h));
    let values: Vec<u8> = job.classes.iter().map(|c| c.value).collect();
    let more = predict_rows(&opened, job.choice, &grid, &model, &values, r0, r1, &mut job.count, &mut job.area)?;
    strip.extend_from_slice(&more);
    job.rows = r1;
    let at = anchor(&grid, &crs);
    let out = ToolOutput::new();
    if r1 >= grid.h {
        return finish(out, &opened, &grid, &job, &strip, at);
    }
    let out = out.message(format!("Classified {:.0}% of the map", r1 as f64 / grid.h as f64 * 100.0));
    // The model and samples go on as they came.
    let mut layer = job.layer(at, &files, None, &(Vec::new(), Vec::new()), &strip);
    let props = &mut layer["features"][0]["properties"];
    if let Some(m) = model_bytes {
        props["lcc_model"] = json!(m);
    }
    props["lcc_train_x"] = json!(train_x);
    props["lcc_train_y"] = json!(train_y);
    Ok(out.layer(&job.layer_name(grid.h), layer).into())
}

/// The finished map: its GeoTIFF handed over for `add-results`, and the class areas.
fn finish(mut out: ToolOutput, opened: &[Opened], grid: &Grid, job: &JobState, strip: &[u8], at: [f64; 2]) -> Result<Value, String> {
    let colors: Vec<(u8, [u8; 3])> = job.classes.iter().map(|c| (c.value, c.color)).collect();
    let names: Vec<(u8, String)> = job.classes.iter().map(|c| (c.value, c.label.clone())).collect();
    let geo = &opened[stack::pick(&opened.iter().map(|o| &o.grid).collect::<Vec<_>>(), job.choice)].geo;
    let tif = tif::write_classes(grid, geo, strip, &colors, &names)?;
    out = out.table(&format!("{} – Class Areas", job.name), area_table(&job.classes, &job.count, &job.area));
    let total: f64 = job.area.iter().sum();
    let mut report = job.report.clone();
    for (c, info) in job.classes.iter().enumerate() {
        report +=
            &format!("\n{} area: {:.1} ha ({:.1}%)", info.label, job.area[c], if total > 0.0 { job.area[c] / total * 100.0 } else { 0.0 });
    }
    out = out.message("Run Land Cover › Add Results to Map to put the classified map on the map.");
    let file = format!("{}.tif", crate::file_stem(&job.name));
    Ok(out.layer(&format!("{} (raster data)", job.name), crate::payload(at, &job.name, &file, "classified", &tif, &report)).into())
}
