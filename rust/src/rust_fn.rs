extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::prelude::*;

use ndarray_linalg::cholesky::*;
use ndarray_linalg::solve::Inverse;

use ndarray_utils::rank::*;

use ndarray_stats::CorrelationExt;

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

fn sum_cerr(c1: &Array2<f64>, c2: &Array2<f64>) -> f64 {
    // assumes c1 and c2 are correlation matrices of same dimension
    let mut res: f64 = 0.;
    let nrow = &c1.nrows();
    let ncol = &c1.ncols();
    for i in 0..*nrow {
        for j in 0..*ncol {
            if j > i {
                res += (c1[[i, j]] - c2[[i, j]]).abs();
            }
        }
    }
    res
}

pub fn  ic_ipc_reorder(x: Array2<f64>, ctar: Array2<f64>) -> Array2<f64> {
    // start off with IC method
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

    let htar = ctar.cholesky(UPLO::Upper).unwrap();
    let tt = m.dot(&inv_f).dot(&htar);
    let r = tt.rank_axis(Axis(1), RankMethod::Minimum) - 1;

    //TODO: more efficient
    let mut y = Array::from_elem((n_sim, n_col), -999.);
    for i in 0..n_sim {
        for j in 0..n_col {
            y[[i, j]] = x[[r[[i, j]], j]]
        }
    }

    // set up for the ILS portion
    let mut e0 = 999 as f64;
    let mut e1 = 99 as f64;

    let cic = y.t().pearson_correlation().unwrap();
    let hic = cic.cholesky(UPLO::Upper).unwrap();

    let mut hper = htar.clone();

    // inital perturb
    for i in 0..n_col {
        for j in 0..n_col {
            if i != j {
                hper[[i, j]] = hper[[i, j]] + hper[[i, j]] - hic[[i, j]]
            }
        }
    }

    while e1 < e0 {

        let tt = m.dot(&inv_f).dot(&hper);
        let r = tt.rank_axis(Axis(1), RankMethod::Minimum) - 1;
        
        //TODO: more efficient
        for i in 0..n_sim {
            for j in 0..n_col {
                y[[i, j]] = x[[r[[i, j]], j]]
            }
        }

        let cic = y.t().pearson_correlation().unwrap();
        e0 = e1;
        e1 = sum_cerr(&cic, &ctar);

        let hic = cic.cholesky(UPLO::Upper).unwrap();

        // perturb
        for i in 0..n_col {
            for j in 0..n_col {
                if i != j {
                    hper[[i, j]] = hper[[i, j]] + htar[[i, j]] - hic[[i, j]]
                }
            }
        }

    }
    y
}
