//! Accuracy assessment: the confusion matrix (rows the reference class, columns the class
//! predicted), overall accuracy, Cohen's kappa, producer's and user's accuracy and F1.

pub struct Confusion {
    pub k: usize,
    /// `m[actual * k + predicted]`.
    pub m: Vec<u64>,
}

impl Confusion {
    pub fn new(k: usize) -> Confusion {
        Confusion { k, m: vec![0; k * k] }
    }

    pub fn add(&mut self, actual: usize, predicted: usize) {
        self.m[actual * self.k + predicted] += 1;
    }

    pub fn total(&self) -> u64 {
        self.m.iter().sum()
    }

    pub fn row(&self, c: usize) -> u64 {
        self.m[c * self.k..(c + 1) * self.k].iter().sum()
    }

    pub fn col(&self, c: usize) -> u64 {
        (0..self.k).map(|r| self.m[r * self.k + c]).sum()
    }

    pub fn diag(&self, c: usize) -> u64 {
        self.m[c * self.k + c]
    }

    pub fn overall(&self) -> f64 {
        let n = self.total();
        if n == 0 { f64::NAN } else { (0..self.k).map(|c| self.diag(c)).sum::<u64>() as f64 / n as f64 }
    }

    /// Cohen's kappa: (pₒ − pₑ) / (1 − pₑ).
    pub fn kappa(&self) -> f64 {
        let n = self.total() as f64;
        if n == 0.0 {
            return f64::NAN;
        }
        let po = self.overall();
        let pe: f64 = (0..self.k).map(|c| self.row(c) as f64 * self.col(c) as f64).sum::<f64>() / (n * n);
        if (1.0 - pe).abs() < 1e-12 { 1.0 } else { (po - pe) / (1.0 - pe) }
    }

    /// Producer's accuracy (recall): of the reference pixels of `c`, how many were classified `c`.
    pub fn producers(&self, c: usize) -> f64 {
        let r = self.row(c);
        if r == 0 { f64::NAN } else { self.diag(c) as f64 / r as f64 }
    }

    /// User's accuracy (precision): of the pixels classified `c`, how many are `c`.
    pub fn users(&self, c: usize) -> f64 {
        let k = self.col(c);
        if k == 0 { f64::NAN } else { self.diag(c) as f64 / k as f64 }
    }

    pub fn f1(&self, c: usize) -> f64 {
        let (p, u) = (self.producers(c), self.users(c));
        if p.is_nan() || u.is_nan() || p + u == 0.0 { f64::NAN } else { 2.0 * p * u / (p + u) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn congalton_example() {
        // Congalton & Green's worked example (Assessing the Accuracy of Remotely Sensed Data,
        // table 5.1): overall 321/434 = 0.74, kappa 0.65.
        let rows = [[65, 4, 22, 24], [6, 81, 5, 8], [0, 11, 85, 19], [4, 7, 3, 90]];
        let mut c = Confusion::new(4);
        // The table's rows are the classified map, its columns the reference.
        for (p, row) in rows.iter().enumerate() {
            for (a, &n) in row.iter().enumerate() {
                for _ in 0..n {
                    c.add(a, p);
                }
            }
        }
        assert!((c.overall() - 321.0 / 434.0).abs() < 1e-9, "{}", c.overall());
        assert!((c.kappa() - 0.65).abs() < 0.005, "{}", c.kappa());
        assert!((c.producers(0) - 65.0 / 75.0).abs() < 1e-9);
        assert!((c.users(0) - 65.0 / 115.0).abs() < 1e-9);
    }
}
