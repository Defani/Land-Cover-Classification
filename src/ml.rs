//! The classifiers, written for this plugin (no external machine-learning crate, so they run in
//! RustGIS's WebAssembly sandbox): Random Forest and CART (Breiman 2001; Breiman et al. 1984),
//! SVM (C-SVC, one-against-one, trained by SMO with second-order working-set selection, Fan,
//! Chen & Lin 2005), k-nearest neighbours (on a k-d tree), Gaussian naive Bayes, minimum
//! distance and maximum likelihood.

/// Training data: `n` rows of `d` features, with class indexes `0..k`.
pub struct Data<'a> {
    pub x: &'a [f32],
    pub y: &'a [u16],
    pub d: usize,
    pub k: usize,
}

impl Data<'_> {
    pub fn n(&self) -> usize {
        self.y.len()
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.x[i * self.d..(i + 1) * self.d]
    }
}

/// SplitMix64: small, seeded, and the same on every platform.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A whole number in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * n as f64) as usize % n.max(1)
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

// ---- Decision trees and forests ---------------------------------------------------------------

#[derive(Clone, Copy)]
struct Node {
    thr: f32,
    /// The feature split on, or `LEAF`.
    f: u16,
    /// A leaf's class.
    class: u16,
    /// The left child; the right one follows it.
    left: u32,
}

const LEAF: u16 = u16::MAX;

pub struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    /// The tree's class for `x`, which must hold every feature the tree splits on
    /// ([`Forest::predict`] checks it).
    #[inline]
    fn predict(&self, x: &[f32]) -> u16 {
        let nodes = &self.nodes[..];
        let mut i = 0usize;
        loop {
            // SAFETY: a tree's nodes are checked as they are grown (`grow`) or read
            // (`Forest::from_bytes`): a split's children are nodes of the tree and its feature is
            // below the forest's `d`, which `x` holds. This loop is most of a classification's
            // cost in RustGIS's metered interpreter, so it does without bounds checks.
            let n = unsafe { nodes.get_unchecked(i) };
            if n.f == LEAF {
                return n.class;
            }
            let v = unsafe { *x.get_unchecked(n.f as usize) };
            i = n.left as usize + usize::from(v > n.thr);
        }
    }

    /// The mean depth of its leaves (weighted equally), for cost estimates.
    fn mean_depth(&self) -> f64 {
        let mut stack = vec![(0usize, 0usize)];
        let (mut sum, mut leaves) = (0usize, 0usize);
        while let Some((i, d)) = stack.pop() {
            let n = &self.nodes[i];
            if n.f == LEAF {
                sum += d;
                leaves += 1;
            } else {
                stack.push((n.left as usize, d + 1));
                stack.push((n.left as usize + 1, d + 1));
            }
        }
        sum as f64 / leaves.max(1) as f64
    }
}

/// The features binned for growing trees: each sample's bin (at most 256 per feature) and the
/// threshold between bins `b` and `b + 1` (a value `<=` it is in bin `b` or lower).
struct Bins {
    bin: Vec<u8>,
    thr: Vec<Vec<f32>>,
}

fn bins(data: &Data) -> Bins {
    let (n, d) = (data.n(), data.d);
    let mut bin = vec![0u8; n * d];
    let mut thr = Vec::with_capacity(d);
    let mut vals: Vec<f32> = Vec::with_capacity(n);
    for f in 0..d {
        vals.clear();
        vals.extend((0..n).map(|i| data.x[i * d + f]));
        vals.sort_by(f32::total_cmp);
        vals.dedup();
        let t: Vec<f32> = if vals.len() <= 256 {
            vals.windows(2).map(|w| w[0] + (w[1] - w[0]) / 2.0).collect()
        } else {
            // 256 bins of about equal counts of distinct values.
            let mut t: Vec<f32> = (1..256)
                .map(|q| {
                    let i = q * vals.len() / 256;
                    vals[i - 1] + (vals[i] - vals[i - 1]) / 2.0
                })
                .collect();
            t.dedup();
            t
        };
        for i in 0..n {
            let v = data.x[i * d + f];
            bin[i * d + f] = t.partition_point(|&c| c < v) as u8;
        }
        thr.push(t);
    }
    Bins { bin, thr }
}

pub struct TreeParams {
    pub mtry: usize,
    pub min_leaf: usize,
    pub max_depth: usize,
}

/// Grows a tree on the samples `idx` (repeats allowed), adding each split's Gini decrease to
/// `importance`.
fn grow(data: &Data, bins: &Bins, idx: &mut [u32], p: &TreeParams, rng: &mut Rng, importance: &mut [f64]) -> Tree {
    let (d, k) = (data.d, data.k);
    let mut nodes = vec![Node { thr: 0.0, f: LEAF, class: 0, left: 0 }];
    let mut stack = vec![(0usize, 0usize, idx.len(), 0usize)];
    let mut counts = vec![0u32; k];
    let mut hist = vec![0u32; 256 * k];
    let mut feats: Vec<usize> = (0..d).collect();
    let mut left = vec![0u32; k];
    let mut bin_n = vec![0u32; 256];
    let mut touched: Vec<u8> = Vec::with_capacity(256);
    let max_depth = if p.max_depth == 0 { usize::MAX } else { p.max_depth };
    let min_leaf = p.min_leaf.max(1);
    while let Some((node, lo, hi, depth)) = stack.pop() {
        let part = &mut idx[lo..hi];
        counts.iter_mut().for_each(|c| *c = 0);
        for &i in part.iter() {
            counts[data.y[i as usize] as usize] += 1;
        }
        let n = part.len();
        let majority = (0..k).max_by_key(|&c| (counts[c], std::cmp::Reverse(c))).unwrap_or(0) as u16;
        nodes[node].class = majority;
        let pure = counts.iter().filter(|&&c| c > 0).count() <= 1;
        if pure || n < 2 * min_leaf || depth >= max_depth {
            continue;
        }
        let parent: f64 = counts.iter().map(|&c| (c as f64) * (c as f64)).sum::<f64>() / n as f64;
        let mut best = (parent + 1e-9, usize::MAX, 0usize);
        // mtry features drawn without replacement.
        let m = p.mtry.clamp(1, d);
        for j in 0..m {
            let r = j + rng.below(d - j);
            feats.swap(j, r);
        }
        for &f in &feats[..m] {
            let nb = bins.thr[f].len() + 1;
            if nb < 2 {
                continue;
            }
            // The class counts of the bins this node's samples fall in; only those bins are
            // scanned (a small node touches few of the 256).
            touched.clear();
            for &i in part.iter() {
                let i = i as usize;
                let b = bins.bin[i * d + f] as usize;
                if bin_n[b] == 0 {
                    touched.push(b as u8);
                }
                bin_n[b] += 1;
                hist[b * k + data.y[i] as usize] += 1;
            }
            if touched.len() > 48 {
                touched.clear();
                touched.extend((0..nb).filter(|&b| bin_n[b] > 0).map(|b| b as u8));
            } else {
                touched.sort_unstable();
            }
            left.iter_mut().for_each(|c| *c = 0);
            let mut nl = 0usize;
            for (ti, &b) in touched.iter().enumerate() {
                let b = b as usize;
                let row = &hist[b * k..(b + 1) * k];
                for c in 0..k {
                    left[c] += row[c];
                }
                nl += bin_n[b] as usize;
                // The last bin leaves nothing on the right.
                if ti + 1 == touched.len() {
                    break;
                }
                let nr = n - nl;
                if nl < min_leaf {
                    continue;
                }
                if nr < min_leaf {
                    break;
                }
                let (mut sl, mut sr) = (0.0f64, 0.0f64);
                for c in 0..k {
                    let l = left[c] as f64;
                    let r = (counts[c] - left[c]) as f64;
                    sl += l * l;
                    sr += r * r;
                }
                let score = sl / nl as f64 + sr / nr as f64;
                if score > best.0 {
                    best = (score, f, b);
                }
            }
            for &b in touched.iter() {
                let b = b as usize;
                bin_n[b] = 0;
                hist[b * k..(b + 1) * k].iter_mut().for_each(|h| *h = 0);
            }
        }
        if best.1 == usize::MAX {
            continue;
        }
        let (score, f, b) = best;
        importance[f] += score - parent;
        // Partition: bins <= b to the left.
        let mut i = 0;
        let mut j = part.len();
        while i < j {
            if bins.bin[part[i] as usize * d + f] as usize <= b {
                i += 1;
            } else {
                j -= 1;
                part.swap(i, j);
            }
        }
        let l = nodes.len();
        nodes[node].f = f as u16;
        nodes[node].thr = bins.thr[f][b];
        nodes[node].left = l as u32;
        nodes.push(Node { thr: 0.0, f: LEAF, class: majority, left: 0 });
        nodes.push(Node { thr: 0.0, f: LEAF, class: majority, left: 0 });
        stack.push((l, lo, lo + i, depth + 1));
        stack.push((l + 1, lo + i, hi, depth + 1));
    }
    Tree { nodes }
}

pub struct Forest {
    trees: Vec<Tree>,
    k: usize,
    /// The features a sample has.
    d: usize,
    /// Mean decrease in Gini impurity per feature, in percent of the total.
    pub importance: Vec<f64>,
    /// Out-of-bag accuracy (a forest of bootstrap samples only).
    pub oob: Option<f64>,
    pub depth: f64,
}

pub struct ForestParams {
    pub trees: usize,
    pub tree: TreeParams,
    /// Bootstrap samples (a random forest), or every sample once (a single CART tree).
    pub bootstrap: bool,
    pub seed: u64,
}

pub fn forest(data: &Data, p: &ForestParams) -> Forest {
    let n = data.n();
    let bins = bins(data);
    let mut rng = Rng::new(p.seed);
    let mut importance = vec![0.0; data.d];
    let mut trees = Vec::with_capacity(p.trees);
    let mut oob_votes = if p.bootstrap { vec![0u16; n * data.k] } else { Vec::new() };
    let mut inbag = vec![false; n];
    let mut idx: Vec<u32> = Vec::with_capacity(n);
    for _ in 0..p.trees.max(1) {
        idx.clear();
        if p.bootstrap {
            inbag.iter_mut().for_each(|b| *b = false);
            for _ in 0..n {
                let i = rng.below(n);
                inbag[i] = true;
                idx.push(i as u32);
            }
        } else {
            idx.extend(0..n as u32);
        }
        let t = grow(data, &bins, &mut idx, &p.tree, &mut rng, &mut importance);
        if p.bootstrap {
            for i in (0..n).filter(|&i| !inbag[i]) {
                let c = t.predict(data.row(i)) as usize;
                oob_votes[i * data.k + c] = oob_votes[i * data.k + c].saturating_add(1);
            }
        }
        trees.push(t);
    }
    let total: f64 = importance.iter().sum();
    if total > 0.0 {
        importance.iter_mut().for_each(|v| *v = *v / total * 100.0);
    }
    let oob = p.bootstrap.then(|| {
        let (mut right, mut voted) = (0usize, 0usize);
        for i in 0..n {
            let v = &oob_votes[i * data.k..(i + 1) * data.k];
            if v.iter().any(|&c| c > 0) {
                voted += 1;
                let c = (0..data.k).max_by_key(|&c| (v[c], std::cmp::Reverse(c))).unwrap_or(0);
                right += usize::from(c == data.y[i] as usize);
            }
        }
        right as f64 / voted.max(1) as f64
    });
    let depth = trees.iter().map(Tree::mean_depth).sum::<f64>() / trees.len().max(1) as f64;
    Forest { trees, k: data.k, d: data.d, importance, oob, depth }
}

impl Forest {
    /// The trees as bytes (for a classification that runs in several passes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.k as u32).to_le_bytes());
        out.extend_from_slice(&(self.trees.len() as u32).to_le_bytes());
        for t in &self.trees {
            out.extend_from_slice(&(t.nodes.len() as u32).to_le_bytes());
            for n in &t.nodes {
                out.extend_from_slice(&n.thr.to_le_bytes());
                out.extend_from_slice(&n.f.to_le_bytes());
                out.extend_from_slice(&n.class.to_le_bytes());
                out.extend_from_slice(&n.left.to_le_bytes());
            }
        }
        out
    }

    /// Trees from [`Forest::to_bytes`], checked so a damaged one can't send a lookup astray.
    pub fn from_bytes(b: &[u8], d: usize) -> Option<Forest> {
        let u32at = |o: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?)) };
        let k = u32at(0)? as usize;
        let n = u32at(4)? as usize;
        let mut o = 8;
        let mut trees = Vec::with_capacity(n.min(10_000));
        for _ in 0..n {
            let m = u32at(o)? as usize;
            o += 4;
            let mut nodes = Vec::with_capacity(m.min(b.len() / 12));
            for _ in 0..m {
                let s = b.get(o..o + 12)?;
                let node = Node {
                    thr: f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                    f: u16::from_le_bytes([s[4], s[5]]),
                    class: u16::from_le_bytes([s[6], s[7]]),
                    left: u32::from_le_bytes([s[8], s[9], s[10], s[11]]),
                };
                let ok = if node.f == LEAF { (node.class as usize) < k } else { (node.f as usize) < d && (node.left as usize) + 1 < m };
                if !ok {
                    return None;
                }
                nodes.push(node);
                o += 12;
            }
            trees.push(Tree { nodes });
        }
        let depth = trees.iter().map(Tree::mean_depth).sum::<f64>() / trees.len().max(1) as f64;
        Some(Forest { trees, k, d, importance: Vec::new(), oob: None, depth })
    }

    fn predict(&self, x: &[f32], votes: &mut [u16]) -> u16 {
        assert!(x.len() >= self.d, "a sample has fewer features than the forest");
        if self.trees.len() == 1 {
            return self.trees[0].predict(x);
        }
        votes.iter_mut().for_each(|v| *v = 0);
        for t in &self.trees {
            votes[t.predict(x) as usize] += 1;
        }
        (0..self.k).max_by_key(|&c| (votes[c], std::cmp::Reverse(c))).unwrap_or(0) as u16
    }

    pub fn trees(&self) -> usize {
        self.trees.len()
    }

    pub fn nodes(&self) -> usize {
        self.trees.iter().map(|t| t.nodes.len()).sum()
    }
}

// ---- Feature scaling ------------------------------------------------------------------------

/// Scales each feature to zero mean and unit variance (by the training samples).
pub struct Scaler {
    mean: Vec<f32>,
    inv: Vec<f32>,
}

impl Scaler {
    pub fn fit(data: &Data) -> Scaler {
        let (n, d) = (data.n().max(1) as f64, data.d);
        let mut mean = vec![0f64; d];
        let mut sq = vec![0f64; d];
        for i in 0..data.n() {
            for (f, v) in data.row(i).iter().enumerate() {
                mean[f] += *v as f64;
                sq[f] += (*v as f64) * (*v as f64);
            }
        }
        let mean: Vec<f64> = mean.iter().map(|m| m / n).collect();
        let inv = (0..d)
            .map(|f| {
                let var = (sq[f] / n - mean[f] * mean[f]).max(0.0);
                if var > 1e-20 { (1.0 / var.sqrt()) as f32 } else { 1.0 }
            })
            .collect();
        Scaler { mean: mean.iter().map(|m| *m as f32).collect(), inv }
    }

    #[inline]
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        for f in 0..x.len() {
            out[f] = (x[f] - self.mean[f]) * self.inv[f];
        }
    }

    pub fn all(&self, data: &Data) -> Vec<f32> {
        let mut out = vec![0f32; data.x.len()];
        for i in 0..data.n() {
            self.apply(data.row(i), &mut out[i * data.d..(i + 1) * data.d]);
        }
        out
    }
}

// ---- Support vector machine -----------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kernel {
    Rbf(f32),
    Linear,
}

/// e^x for x ≤ 0, to about 1 part in 10⁶: x = n·ln 2 + r with |r| ≤ ln 2 / 2, e^r by its
/// Taylor series to r⁶, 2ⁿ written into the exponent. A third of `f32::exp`'s instructions,
/// which counts in RustGIS's metered interpreter: an RBF machine calls it once per support
/// vector per pixel.
#[inline]
pub fn exp_neg(x: f32) -> f32 {
    if x < -87.0 {
        return 0.0;
    }
    let n = (x * std::f32::consts::LOG2_E).round();
    // ln 2 split in two (Cody & Waite), so r keeps its precision when n is large.
    let r = (x - n * 0.693_145_75) - n * 1.428_606_8e-6;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[inline]
fn kernel(k: Kernel, a: &[f32], b: &[f32]) -> f32 {
    match k {
        Kernel::Rbf(g) => {
            let mut s = 0f32;
            for (x, y) in a.iter().zip(b) {
                let t = x - y;
                s += t * t;
            }
            exp_neg(-g * s)
        }
        Kernel::Linear => a.iter().zip(b).map(|(x, y)| x * y).sum(),
    }
}

/// One binary machine: class `a` (+1) against class `b` (−1).
struct Pair {
    a: u16,
    b: u16,
    /// (support vector, coefficient αᵢyᵢ).
    sv: Vec<(u32, f32)>,
    rho: f32,
    /// A linear machine's weights.
    w: Vec<f32>,
}

pub struct Svm {
    kernel: Kernel,
    scaler: Scaler,
    /// The support vectors, scaled.
    sv: Vec<f32>,
    d: usize,
    pairs: Vec<Pair>,
    k: usize,
    pub notes: Vec<String>,
}

pub struct SvmParams {
    pub kernel: Kernel,
    pub c: f32,
    /// The most samples a pair of classes trains on (subsampled beyond it).
    pub max_pair: usize,
    pub seed: u64,
}

/// Solves the dual of a C-SVC with SMO (libsvm's algorithm, Fan et al. 2005): returns α and ρ.
fn smo(q: &[f32], y: &[f32], c: f32, max_iter: usize) -> (Vec<f32>, f32, bool) {
    let m = y.len();
    let qd: Vec<f32> = (0..m).map(|i| q[i * m + i]).collect();
    let mut a = vec![0f32; m];
    let mut g = vec![-1f32; m];
    let eps = 1e-3f32;
    let tau = 1e-12f32;
    let mut converged = false;
    for _ in 0..max_iter {
        // i: the most violating in I_up; j: second-order choice in I_low.
        let mut gmax = f32::NEG_INFINITY;
        let mut i = usize::MAX;
        for t in 0..m {
            let up = if y[t] > 0.0 { a[t] < c } else { a[t] > 0.0 };
            if up {
                let v = -y[t] * g[t];
                if v >= gmax {
                    gmax = v;
                    i = t;
                }
            }
        }
        if i == usize::MAX {
            converged = true;
            break;
        }
        let qi = &q[i * m..(i + 1) * m];
        let mut gmin = f32::INFINITY;
        let mut j = usize::MAX;
        let mut obj_min = f32::INFINITY;
        for t in 0..m {
            let low = if y[t] > 0.0 { a[t] > 0.0 } else { a[t] < c };
            if !low {
                continue;
            }
            let v = -y[t] * g[t];
            if v <= gmin {
                gmin = v;
            }
            let b = gmax - v;
            if b > 0.0 {
                // Q here is y_i y_j K, so K_it = y_i y_t Q_it.
                let mut aa = qd[i] + qd[t] - 2.0 * y[i] * y[t] * qi[t];
                if aa <= 0.0 {
                    aa = tau;
                }
                let o = -(b * b) / aa;
                if o <= obj_min {
                    obj_min = o;
                    j = t;
                }
            }
        }
        if gmax - gmin < eps || j == usize::MAX {
            converged = true;
            break;
        }
        let (ai, aj) = (a[i], a[j]);
        let qij = qi[j];
        if y[i] != y[j] {
            let mut quad = qd[i] + qd[j] + 2.0 * qij;
            if quad <= 0.0 {
                quad = tau;
            }
            let delta = (-g[i] - g[j]) / quad;
            let diff = a[i] - a[j];
            a[i] += delta;
            a[j] += delta;
            if diff > 0.0 {
                if a[j] < 0.0 {
                    a[j] = 0.0;
                    a[i] = diff;
                }
            } else if a[i] < 0.0 {
                a[i] = 0.0;
                a[j] = -diff;
            }
            if diff > 0.0 {
                if a[i] > c {
                    a[i] = c;
                    a[j] = c - diff;
                }
            } else if a[j] > c {
                a[j] = c;
                a[i] = c + diff;
            }
        } else {
            let mut quad = qd[i] + qd[j] - 2.0 * qij;
            if quad <= 0.0 {
                quad = tau;
            }
            let delta = (g[i] - g[j]) / quad;
            let sum = a[i] + a[j];
            a[i] -= delta;
            a[j] += delta;
            if sum > c {
                if a[i] > c {
                    a[i] = c;
                    a[j] = sum - c;
                }
            } else if a[j] < 0.0 {
                a[j] = 0.0;
                a[i] = sum;
            }
            if sum > c {
                if a[j] > c {
                    a[j] = c;
                    a[i] = sum - c;
                }
            } else if a[i] < 0.0 {
                a[i] = 0.0;
                a[j] = sum;
            }
        }
        let (dai, daj) = (a[i] - ai, a[j] - aj);
        let qj = &q[j * m..(j + 1) * m];
        for t in 0..m {
            g[t] += qi[t] * dai + qj[t] * daj;
        }
    }
    // ρ: the mean of y·G over free vectors, else the middle of the feasible range.
    let (mut ub, mut lb, mut sum, mut free) = (f32::INFINITY, f32::NEG_INFINITY, 0f32, 0usize);
    for t in 0..m {
        let yg = y[t] * g[t];
        if a[t] >= c {
            if y[t] < 0.0 { ub = ub.min(yg) } else { lb = lb.max(yg) }
        } else if a[t] <= 0.0 {
            if y[t] > 0.0 { ub = ub.min(yg) } else { lb = lb.max(yg) }
        } else {
            free += 1;
            sum += yg;
        }
    }
    let rho = if free > 0 { sum / free as f32 } else { (ub + lb) / 2.0 };
    (a, rho, converged)
}

pub fn svm(data: &Data, p: &SvmParams) -> Svm {
    let d = data.d;
    let scaler = Scaler::fit(data);
    let xs = scaler.all(data);
    let row = |i: usize| &xs[i * d..(i + 1) * d];
    let mut by_class: Vec<Vec<u32>> = vec![Vec::new(); data.k];
    for (i, &c) in data.y.iter().enumerate() {
        by_class[c as usize].push(i as u32);
    }
    let mut rng = Rng::new(p.seed);
    let mut pairs = Vec::new();
    let mut notes = Vec::new();
    let mut subsampled = 0;
    let mut unconverged = 0;
    let mut used: Vec<i64> = vec![-1; data.n()];
    let mut sv: Vec<f32> = Vec::new();
    for a in 0..data.k {
        for b in a + 1..data.k {
            if by_class[a].is_empty() || by_class[b].is_empty() {
                continue;
            }
            let mut ia = by_class[a].clone();
            let mut ib = by_class[b].clone();
            let half = p.max_pair / 2;
            if ia.len() + ib.len() > p.max_pair {
                subsampled += 1;
                let (na, nb) = if ia.len() < half {
                    (ia.len(), p.max_pair - ia.len())
                } else if ib.len() < half {
                    (p.max_pair - ib.len(), ib.len())
                } else {
                    (half, half)
                };
                rng.shuffle(&mut ia);
                rng.shuffle(&mut ib);
                ia.truncate(na);
                ib.truncate(nb);
            }
            let idx: Vec<u32> = ia.iter().chain(ib.iter()).copied().collect();
            let y: Vec<f32> = idx.iter().map(|&i| if data.y[i as usize] as usize == a { 1.0 } else { -1.0 }).collect();
            let m = idx.len();
            let mut q = vec![0f32; m * m];
            for s in 0..m {
                for t in s..m {
                    let v = y[s] * y[t] * kernel(p.kernel, row(idx[s] as usize), row(idx[t] as usize));
                    q[s * m + t] = v;
                    q[t * m + s] = v;
                }
            }
            let (alpha, rho, ok) = smo(&q, &y, p.c, (100 * m).clamp(10_000, 400_000));
            drop(q);
            if !ok {
                unconverged += 1;
            }
            let mut pair = Pair { a: a as u16, b: b as u16, sv: Vec::new(), rho, w: Vec::new() };
            if p.kernel == Kernel::Linear {
                pair.w = vec![0f32; d];
            }
            for (s, &al) in alpha.iter().enumerate() {
                if al <= 0.0 {
                    continue;
                }
                let gi = idx[s] as usize;
                let coef = al * y[s];
                if p.kernel == Kernel::Linear {
                    for (f, v) in row(gi).iter().enumerate() {
                        pair.w[f] += coef * v;
                    }
                } else {
                    if used[gi] < 0 {
                        used[gi] = (sv.len() / d) as i64;
                        sv.extend_from_slice(row(gi));
                    }
                    pair.sv.push((used[gi] as u32, coef));
                }
            }
            pairs.push(pair);
        }
    }
    if subsampled > 0 {
        notes.push(format!("SVM: {subsampled} pair(s) of classes trained on a random {} of their samples", p.max_pair));
    }
    if unconverged > 0 {
        notes.push(format!("SVM: {unconverged} pair(s) stopped at the iteration limit before converging (try a smaller C)"));
    }
    Svm { kernel: p.kernel, scaler, sv, d, pairs, k: data.k, notes }
}

struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, v: usize) {
        self.0.extend_from_slice(&(v as u32).to_le_bytes());
    }
    fn f32s(&mut self, v: &[f32]) {
        self.u32(v.len());
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
}

struct Reader<'a>(&'a [u8], usize);

impl Reader<'_> {
    fn u32(&mut self) -> Option<usize> {
        let v = u32::from_le_bytes(self.0.get(self.1..self.1 + 4)?.try_into().ok()?);
        self.1 += 4;
        Some(v as usize)
    }
    fn f32(&mut self) -> Option<f32> {
        let v = f32::from_le_bytes(self.0.get(self.1..self.1 + 4)?.try_into().ok()?);
        self.1 += 4;
        Some(v)
    }
    fn f32s(&mut self) -> Option<Vec<f32>> {
        let n = self.u32()?;
        if n > (self.0.len() - self.1) / 4 {
            return None;
        }
        (0..n).map(|_| self.f32()).collect()
    }
}

impl Svm {
    /// The machine as bytes (for a classification that runs in several passes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer(Vec::new());
        match self.kernel {
            Kernel::Linear => {
                w.u32(0);
                w.0.extend_from_slice(&0f32.to_le_bytes());
            }
            Kernel::Rbf(g) => {
                w.u32(1);
                w.0.extend_from_slice(&g.to_le_bytes());
            }
        }
        w.u32(self.d);
        w.u32(self.k);
        w.f32s(&self.scaler.mean);
        w.f32s(&self.scaler.inv);
        w.f32s(&self.sv);
        w.u32(self.pairs.len());
        for p in &self.pairs {
            w.u32(p.a as usize);
            w.u32(p.b as usize);
            w.0.extend_from_slice(&p.rho.to_le_bytes());
            w.f32s(&p.w);
            w.u32(p.sv.len());
            for (i, c) in &p.sv {
                w.u32(*i as usize);
                w.0.extend_from_slice(&c.to_le_bytes());
            }
        }
        w.0
    }

    /// A machine from [`Svm::to_bytes`], checked.
    pub fn from_bytes(b: &[u8]) -> Option<Svm> {
        let mut r = Reader(b, 0);
        let kind = r.u32()?;
        let g = r.f32()?;
        let kernel = if kind == 0 { Kernel::Linear } else { Kernel::Rbf(g) };
        let d = r.u32()?;
        let k = r.u32()?;
        let mean = r.f32s()?;
        let inv = r.f32s()?;
        let sv = r.f32s()?;
        if d == 0 || mean.len() != d || inv.len() != d || sv.len() % d != 0 {
            return None;
        }
        let nsv = sv.len() / d;
        let np = r.u32()?;
        let mut pairs = Vec::with_capacity(np.min(1 << 16));
        for _ in 0..np {
            let (a, bb) = (r.u32()?, r.u32()?);
            let rho = r.f32()?;
            let w = r.f32s()?;
            let n = r.u32()?;
            if a >= k || bb >= k || !(w.is_empty() || w.len() == d) || n > b.len() / 8 {
                return None;
            }
            let mut svs = Vec::with_capacity(n);
            for _ in 0..n {
                let i = r.u32()?;
                if i >= nsv {
                    return None;
                }
                svs.push((i as u32, r.f32()?));
            }
            pairs.push(Pair { a: a as u16, b: bb as u16, sv: svs, rho, w });
        }
        Some(Svm { kernel, scaler: Scaler { mean, inv }, sv, d, pairs, k, notes: Vec::new() })
    }

    /// The features a sample has.
    pub fn dims(&self) -> usize {
        self.d
    }

    pub fn support_vectors(&self) -> usize {
        if self.kernel == Kernel::Linear { self.pairs.len() } else { self.sv.len() / self.d.max(1) }
    }

    fn predict(&self, x: &[f32], s: &mut Scratch) -> u16 {
        self.scaler.apply(x, &mut s.z);
        let z = &s.z;
        if self.kernel != Kernel::Linear {
            let nsv = self.sv.len() / self.d;
            s.kv.resize(nsv, 0.0);
            for v in 0..nsv {
                s.kv[v] = kernel(self.kernel, z, &self.sv[v * self.d..(v + 1) * self.d]);
            }
        }
        s.votes.iter_mut().for_each(|v| *v = 0);
        for p in &self.pairs {
            let f = if self.kernel == Kernel::Linear {
                p.w.iter().zip(z.iter()).map(|(w, v)| w * v).sum::<f32>()
            } else {
                p.sv.iter().map(|&(i, c)| c * s.kv[i as usize]).sum::<f32>()
            } - p.rho;
            let c = if f > 0.0 { p.a } else { p.b };
            s.votes[c as usize] += 1;
        }
        (0..self.k).max_by_key(|&c| (s.votes[c], std::cmp::Reverse(c))).unwrap_or(0) as u16
    }
}

// ---- k-nearest neighbours -------------------------------------------------------------------

pub struct Knn {
    scaler: Scaler,
    /// Scaled samples in k-d tree order.
    x: Vec<f32>,
    y: Vec<u16>,
    d: usize,
    k: usize,
    classes: usize,
    /// Tree nodes: (lo, hi, split feature or LEAF, split value, left child, right child).
    nodes: Vec<(u32, u32, u16, f32, u32, u32)>,
}

/// The most samples k-NN searches (a random choice beyond it): each pixel's search grows with
/// them, and RustGIS meters every instruction.
pub const KNN_MAX: usize = 3000;

pub fn knn(data: &Data, k: usize, seed: u64) -> Knn {
    let d = data.d;
    let scaler = Scaler::fit(data);
    let xs = scaler.all(data);
    let mut order: Vec<u32> = (0..data.n() as u32).collect();
    if order.len() > KNN_MAX {
        Rng::new(seed ^ 0x4B4E).shuffle(&mut order);
        order.truncate(KNN_MAX);
    }
    let mut nodes = Vec::new();
    build_kd(&xs, d, &mut order, 0, &mut nodes);
    let mut x = Vec::with_capacity(xs.len());
    let mut y = Vec::with_capacity(order.len());
    for &i in &order {
        x.extend_from_slice(&xs[i as usize * d..(i as usize + 1) * d]);
        y.push(data.y[i as usize]);
    }
    Knn { scaler, k: k.clamp(1, y.len().max(1)), x, y, d, classes: data.k, nodes }
}

fn build_kd(xs: &[f32], d: usize, order: &mut [u32], offset: usize, nodes: &mut Vec<(u32, u32, u16, f32, u32, u32)>) -> u32 {
    let id = nodes.len() as u32;
    let (lo, hi) = (offset as u32, (offset + order.len()) as u32);
    if order.len() <= 8 {
        nodes.push((lo, hi, LEAF, 0.0, 0, 0));
        return id;
    }
    // The feature of widest spread, split at its median.
    let mut best = (0usize, -1f32);
    for f in 0..d {
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for &i in order.iter() {
            let v = xs[i as usize * d + f];
            mn = mn.min(v);
            mx = mx.max(v);
        }
        if mx - mn > best.1 {
            best = (f, mx - mn);
        }
    }
    let f = best.0;
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |&a, &b| xs[a as usize * d + f].total_cmp(&xs[b as usize * d + f]));
    let split = xs[order[mid] as usize * d + f];
    nodes.push((lo, hi, f as u16, split, 0, 0));
    let (l, r) = order.split_at_mut(mid);
    let left = build_kd(xs, d, l, offset, nodes);
    let right = build_kd(xs, d, r, offset + mid, nodes);
    nodes[id as usize].4 = left;
    nodes[id as usize].5 = right;
    id
}

impl Knn {
    fn search(&self, node: u32, z: &[f32], best: &mut Vec<(f32, u16)>) {
        let (lo, hi, f, split, l, r) = self.nodes[node as usize];
        if f == LEAF {
            for i in lo as usize..hi as usize {
                let p = &self.x[i * self.d..(i + 1) * self.d];
                let mut dist = 0f32;
                for j in 0..self.d {
                    let t = p[j] - z[j];
                    dist += t * t;
                }
                if best.len() < self.k || dist < best[best.len() - 1].0 {
                    let at = best.partition_point(|b| b.0 <= dist);
                    best.insert(at, (dist, self.y[i]));
                    if best.len() > self.k {
                        best.pop();
                    }
                }
            }
            return;
        }
        let diff = z[f as usize] - split;
        let (near, far) = if diff < 0.0 { (l, r) } else { (r, l) };
        self.search(near, z, best);
        if best.len() < self.k || diff * diff < best[best.len() - 1].0 {
            self.search(far, z, best);
        }
    }

    fn predict(&self, x: &[f32], s: &mut Scratch) -> u16 {
        self.scaler.apply(x, &mut s.z);
        s.best.clear();
        let z = std::mem::take(&mut s.z);
        let mut best = std::mem::take(&mut s.best);
        self.search(0, &z, &mut best);
        s.votes.iter_mut().for_each(|v| *v = 0);
        for &(_, c) in &best {
            s.votes[c as usize] += 1;
        }
        let top = s.votes.iter().copied().max().unwrap_or(0);
        // A tie goes to the tied class with the nearest sample.
        let c = best.iter().find(|(_, c)| s.votes[*c as usize] == top).map_or(0, |b| b.1);
        s.z = z;
        s.best = best;
        let _ = self.classes;
        c
    }
}

// ---- Gaussian classifiers -------------------------------------------------------------------

pub struct Bayes {
    mean: Vec<f64>,
    inv_var: Vec<f64>,
    bias: Vec<f64>,
    d: usize,
    k: usize,
}

fn class_means(data: &Data) -> (Vec<f64>, Vec<usize>) {
    let (d, k) = (data.d, data.k);
    let mut mean = vec![0f64; k * d];
    let mut n = vec![0usize; k];
    for i in 0..data.n() {
        let c = data.y[i] as usize;
        n[c] += 1;
        for (f, v) in data.row(i).iter().enumerate() {
            mean[c * d + f] += *v as f64;
        }
    }
    for c in 0..k {
        for f in 0..d {
            mean[c * d + f] /= n[c].max(1) as f64;
        }
    }
    (mean, n)
}

pub fn bayes(data: &Data) -> Bayes {
    let (d, k) = (data.d, data.k);
    let (mean, n) = class_means(data);
    let mut var = vec![0f64; k * d];
    for i in 0..data.n() {
        let c = data.y[i] as usize;
        for (f, v) in data.row(i).iter().enumerate() {
            let t = *v as f64 - mean[c * d + f];
            var[c * d + f] += t * t;
        }
    }
    let total = data.n().max(1) as f64;
    let floor = {
        let mx = (0..k * d).map(|j| var[j] / n[j / d].max(1) as f64).fold(0.0f64, f64::max);
        (mx * 1e-9).max(1e-12)
    };
    let mut inv_var = vec![0f64; k * d];
    let mut bias = vec![0f64; k];
    for c in 0..k {
        bias[c] = ((n[c].max(1)) as f64 / total).ln();
        for f in 0..d {
            let v = (var[c * d + f] / n[c].max(1) as f64).max(floor);
            inv_var[c * d + f] = 1.0 / v;
            bias[c] -= 0.5 * v.ln();
        }
    }
    Bayes { mean, inv_var, bias, d, k }
}

impl Bayes {
    fn predict(&self, x: &[f32]) -> u16 {
        let mut best = (f64::NEG_INFINITY, 0u16);
        for c in 0..self.k {
            let mut s = self.bias[c];
            for f in 0..self.d {
                let t = x[f] as f64 - self.mean[c * self.d + f];
                s -= 0.5 * t * t * self.inv_var[c * self.d + f];
            }
            if s > best.0 {
                best = (s, c as u16);
            }
        }
        best.1
    }
}

pub struct MinDist {
    mean: Vec<f64>,
    d: usize,
    k: usize,
}

pub fn min_dist(data: &Data) -> MinDist {
    MinDist { mean: class_means(data).0, d: data.d, k: data.k }
}

impl MinDist {
    fn predict(&self, x: &[f32]) -> u16 {
        let mut best = (f64::INFINITY, 0u16);
        for c in 0..self.k {
            let mut s = 0.0;
            for f in 0..self.d {
                let t = x[f] as f64 - self.mean[c * self.d + f];
                s += t * t;
            }
            if s < best.0 {
                best = (s, c as u16);
            }
        }
        best.1
    }
}

/// Maximum likelihood with equal priors: each class a multivariate normal distribution.
pub struct Mlc {
    mean: Vec<f64>,
    /// The inverse of each class's covariance's Cholesky factor (lower triangular), row-major.
    linv: Vec<f64>,
    logdet: Vec<f64>,
    d: usize,
    k: usize,
    pub notes: Vec<String>,
}

pub fn mlc(data: &Data) -> Mlc {
    let (d, k) = (data.d, data.k);
    let (mean, n) = class_means(data);
    let mut cov = vec![0f64; k * d * d];
    for i in 0..data.n() {
        let c = data.y[i] as usize;
        let x = data.row(i);
        for a in 0..d {
            let ta = x[a] as f64 - mean[c * d + a];
            for b in 0..=a {
                cov[c * d * d + a * d + b] += ta * (x[b] as f64 - mean[c * d + b]);
            }
        }
    }
    let mut linv = vec![0f64; k * d * d];
    let mut logdet = vec![0f64; k];
    let mut ridged = 0;
    for c in 0..k {
        let m = &mut cov[c * d * d..(c + 1) * d * d];
        let denom = (n[c].max(2) - 1) as f64;
        for a in 0..d {
            for b in 0..=a {
                m[a * d + b] /= denom;
                m[b * d + a] = m[a * d + b];
            }
        }
        let trace: f64 = (0..d).map(|a| m[a * d + a]).sum::<f64>() / d as f64;
        let mut ridge = 0.0;
        if n[c] <= d {
            ridge = trace.max(1e-12) * 1e-3;
            ridged += 1;
        }
        let l = loop {
            let mut t = m.to_vec();
            for a in 0..d {
                t[a * d + a] += ridge;
            }
            if let Some(l) = cholesky(&t, d) {
                break l;
            }
            ridge = if ridge == 0.0 { trace.max(1e-12) * 1e-6 } else { ridge * 10.0 };
            if ridge > trace.max(1e-12) * 10.0 {
                ridged += 1;
                break cholesky(&identity(d, trace.max(1e-12)), d).unwrap_or_else(|| identity(d, 1.0));
            }
        };
        logdet[c] = 2.0 * (0..d).map(|a| l[a * d + a].ln()).sum::<f64>();
        linv[c * d * d..(c + 1) * d * d].copy_from_slice(&lower_inverse(&l, d));
    }
    let mut notes = Vec::new();
    if ridged > 0 {
        notes.push(format!("Maximum likelihood: {ridged} class(es) had too few samples for a full covariance (fewer than bands + 1, or collinear bands): it was regularised"));
    }
    Mlc { mean, linv, logdet, d, k, notes }
}

fn identity(d: usize, v: f64) -> Vec<f64> {
    let mut m = vec![0.0; d * d];
    for a in 0..d {
        m[a * d + a] = v;
    }
    m
}

fn cholesky(m: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = m[i * d + j];
            for p in 0..j {
                s -= l[i * d + p] * l[j * d + p];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i * d + i] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Some(l)
}

fn lower_inverse(l: &[f64], d: usize) -> Vec<f64> {
    let mut inv = vec![0f64; d * d];
    for i in 0..d {
        inv[i * d + i] = 1.0 / l[i * d + i];
        for j in 0..i {
            let mut s = 0.0;
            for p in j..i {
                s -= l[i * d + p] * inv[p * d + j];
            }
            inv[i * d + j] = s / l[i * d + i];
        }
    }
    inv
}

impl Mlc {
    fn predict(&self, x: &[f32], s: &mut Scratch) -> u16 {
        let d = self.d;
        let mut best = (f64::INFINITY, 0u16);
        s.diff.resize(d, 0.0);
        for c in 0..self.k {
            for f in 0..d {
                s.diff[f] = x[f] as f64 - self.mean[c * d + f];
            }
            // Mahalanobis distance: |L⁻¹ (x − μ)|².
            let li = &self.linv[c * d * d..(c + 1) * d * d];
            let mut m = 0.0;
            for a in 0..d {
                let mut t = 0.0;
                for b in 0..=a {
                    t += li[a * d + b] * s.diff[b];
                }
                m += t * t;
            }
            let score = m + self.logdet[c];
            if score < best.0 {
                best = (score, c as u16);
            }
        }
        best.1
    }
}

// ---- One interface --------------------------------------------------------------------------

pub enum Model {
    Forest(Forest),
    Svm(Svm),
    Knn(Knn),
    Bayes(Bayes),
    MinDist(MinDist),
    Mlc(Mlc),
}

/// Buffers a prediction reuses.
pub struct Scratch {
    votes: Vec<u16>,
    z: Vec<f32>,
    kv: Vec<f32>,
    best: Vec<(f32, u16)>,
    diff: Vec<f64>,
}

impl Scratch {
    pub fn new(d: usize, k: usize) -> Scratch {
        Scratch { votes: vec![0; k], z: vec![0.0; d], kv: Vec::new(), best: Vec::new(), diff: vec![0.0; d] }
    }
}

impl Model {
    #[inline]
    pub fn predict(&self, x: &[f32], s: &mut Scratch) -> u16 {
        match self {
            Model::Forest(f) => f.predict(x, &mut s.votes),
            Model::Svm(m) => m.predict(x, s),
            Model::Knn(m) => m.predict(x, s),
            Model::Bayes(m) => m.predict(x),
            Model::MinDist(m) => m.predict(x),
            Model::Mlc(m) => m.predict(x, s),
        }
    }

    /// About how many WebAssembly instructions one pixel's prediction takes (for the budget).
    pub fn cost(&self, d: usize, k: usize) -> f64 {
        let (d, k) = (d as f64, k as f64);
        match self {
            Model::Forest(f) => f.trees.len() as f64 * (f.depth * 30.0 + 25.0) + 20.0 * k,
            Model::Svm(m) => {
                let nsv = m.support_vectors() as f64;
                if m.kernel == Kernel::Linear {
                    m.pairs.len() as f64 * (d * 25.0 + 40.0)
                } else {
                    nsv * (d * 30.0 + 120.0) + m.pairs.iter().map(|p| p.sv.len() as f64 * 25.0).sum::<f64>()
                }
            }
            Model::Knn(m) => 170.0 * d * (m.y.len() as f64).sqrt() * (m.k as f64).powf(0.3),
            Model::Bayes(_) | Model::MinDist(_) => k * d * 40.0,
            Model::Mlc(_) => k * d * d * 60.0,
        }
    }

    pub fn notes(&self) -> Vec<String> {
        match self {
            Model::Svm(m) => m.notes.clone(),
            Model::Mlc(m) => m.notes.clone(),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three Gaussian blobs in 3-D, overlapping a little.
    fn blobs(n: usize, seed: u64) -> (Vec<f32>, Vec<u16>) {
        let mut r = Rng::new(seed);
        let mut g = || {
            // Box-Muller
            let u1 = (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-12;
            let u2 = (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        };
        let centres = [[0.0, 0.0, 0.0], [3.0, 3.0, 0.0], [0.0, 3.0, 3.0]];
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let c = i % 3;
            for f in 0..3 {
                x.push(centres[c][f] + g());
            }
            y.push(c as u16);
        }
        (x, y)
    }

    fn accuracy(m: &Model, x: &[f32], y: &[u16]) -> f64 {
        let mut s = Scratch::new(3, 3);
        let right = (0..y.len()).filter(|&i| m.predict(&x[i * 3..i * 3 + 3], &mut s) == y[i]).count();
        right as f64 / y.len() as f64
    }

    #[test]
    fn every_classifier_learns_blobs() {
        let (x, y) = blobs(600, 1);
        let (tx, ty) = blobs(600, 2);
        let data = Data { x: &x, y: &y, d: 3, k: 3 };
        let models = vec![
            (
                "rf",
                Model::Forest(forest(
                    &data,
                    &ForestParams { trees: 50, tree: TreeParams { mtry: 1, min_leaf: 1, max_depth: 0 }, bootstrap: true, seed: 7 },
                )),
            ),
            (
                "cart",
                Model::Forest(forest(
                    &data,
                    &ForestParams { trees: 1, tree: TreeParams { mtry: 3, min_leaf: 5, max_depth: 0 }, bootstrap: false, seed: 7 },
                )),
            ),
            ("svm", Model::Svm(svm(&data, &SvmParams { kernel: Kernel::Rbf(1.0 / 3.0), c: 10.0, max_pair: 2000, seed: 7 }))),
            (
                "svm-bytes",
                Model::Svm(
                    Svm::from_bytes(
                        &svm(&data, &SvmParams { kernel: Kernel::Rbf(1.0 / 3.0), c: 10.0, max_pair: 2000, seed: 7 }).to_bytes(),
                    )
                    .unwrap(),
                ),
            ),
            ("svm-linear", Model::Svm(svm(&data, &SvmParams { kernel: Kernel::Linear, c: 1.0, max_pair: 2000, seed: 7 }))),
            ("knn", Model::Knn(knn(&data, 5, 1))),
            ("bayes", Model::Bayes(bayes(&data))),
            ("mindist", Model::MinDist(min_dist(&data))),
            ("mlc", Model::Mlc(mlc(&data))),
        ];
        for (name, m) in &models {
            let a = accuracy(m, &tx, &ty);
            assert!(a > 0.85, "{name}: {a}");
        }
        if let Model::Forest(f) = &models[0].1 {
            assert!(f.oob.unwrap() > 0.85);
            assert!((f.importance.iter().sum::<f64>() - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn fast_exp_is_close() {
        for i in 0..2000 {
            let x = -(i as f32) * 0.02;
            let (a, b) = (exp_neg(x), x.exp());
            assert!((a - b).abs() <= 1e-6 * b, "{x}: {a} {b}");
        }
    }

    #[test]
    fn forests_survive_bytes() {
        let (x, y) = blobs(300, 4);
        let data = Data { x: &x, y: &y, d: 3, k: 3 };
        let f =
            forest(&data, &ForestParams { trees: 7, tree: TreeParams { mtry: 2, min_leaf: 1, max_depth: 0 }, bootstrap: true, seed: 1 });
        let g = Forest::from_bytes(&f.to_bytes(), 3).unwrap();
        let mut v = vec![0u16; 3];
        for i in 0..300 {
            assert_eq!(f.predict(&x[i * 3..i * 3 + 3], &mut v), g.predict(&x[i * 3..i * 3 + 3], &mut v));
        }
        assert!(Forest::from_bytes(&f.to_bytes()[..50], 3).is_none());
    }

    #[test]
    fn knn_matches_brute_force() {
        let (x, y) = blobs(300, 3);
        let data = Data { x: &x, y: &y, d: 3, k: 3 };
        let m = knn(&data, 1, 1);
        // 1-NN on the training set gives back each sample's own class.
        let mut s = Scratch::new(3, 3);
        let right = (0..y.len()).filter(|&i| m.predict(&x[i * 3..i * 3 + 3], &mut s) == y[i]).count();
        assert_eq!(right, y.len());
    }
}
