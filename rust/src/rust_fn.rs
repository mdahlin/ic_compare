extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::prelude::*;

use ndarray_linalg::cholesky::*;
use ndarray_linalg::solve::Inverse;

use ndarray_utils::rank::*;

use statrs::distribution::{ContinuousCDF, Normal};

use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn ic_reorder(x: Array2<f64>, ctar: Array2<f64>) -> Array2<f64> {

    let n_sim = x.shape()[0];
    let n_col = x.shape()[1];

    let std_norm = Normal::new(0.0, 1.0).unwrap();
    let b: Array1<f64> = Array::range(1., n_sim as f64 + 1., 1.) / (n_sim as f64 + 1.);
    let c = b.map(|x: &f64| std_norm.inverse_cdf(*x));
    let c_std = &c.std(1.0);
    let z = c / *c_std;
    
    let mut m = Array::from_elem((n_sim, n_col), -999.);
    let mut rng = thread_rng();
    let mut permutation: Vec<usize> = (0..n_sim).collect();
    for i in 0..n_col {
        permutation.shuffle(&mut rng);
        m.column_mut(i).assign(&z.select(Axis(0), &permutation));
    }

    let e = m.t().dot(&m)/(n_sim as f64);
    let f = e.cholesky(UPLO::Upper).unwrap();
    let inv_f = f.inv().unwrap();

    let c = ctar.cholesky(UPLO::Upper).unwrap();
    let tt = m.dot(&inv_f).dot(&c);
    let r = tt.rank_axis(Axis(1), RankMethod::Minimum) - 1;

    //TODO: more efficient
    let mut y = Array::from_elem((n_sim, n_col), -999.);
    for i in 0..n_sim {
        for j in 0..n_col {
            y[[i, j]] = x[[r[[i, j]], j]]
        }
    }
    y
}
