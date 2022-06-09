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
use rand::prelude::IteratorRandom;

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


fn max_axis_error(c1: &Array2<f64>, c2: &Array2<f64>, axis: Axis) -> usize {
    let diff = c1 - c2;
    let abs_axis_sum = diff.fold_axis(axis, 0_f64, |r, x| r + x.abs());

    let mut idx = 0;
    let mut max = 0_f64;
    for (i, e) in abs_axis_sum.iter().enumerate() {
        if e > &max {
            max = *e;
            idx = i;
        }

    }
    idx
}

pub fn ils_reorder(x: Array2<f64>, ctar: Array2<f64>, niter: usize) -> Array2<f64> {
    //x here is usually output of an IC func
    let mut xi = x.clone(); // probably not needed

    let mut c0 = xi.t().pearson_correlation().unwrap();
    let mut e0 = sum_cerr(&c0, &ctar);

    let mut rng = thread_rng();
    let sims_pop: Vec<usize> = (0..x.nrows()).collect();

    for _ in 0..niter {
        let mut xcand = xi.clone();
        let col_idx = max_axis_error(&ctar, &c0, Axis(0));

        let rnd_2ndx = sims_pop.iter().choose_multiple(&mut rng, 2);
        xcand.column_mut(col_idx).swap(*rnd_2ndx[0], *rnd_2ndx[1]);

        let c1 = xcand.t().pearson_correlation().unwrap();
        let e1 = sum_cerr(&c1, &ctar);

        if e1 < e0 {
            xi = xcand;
            e0 = e1;
            c0 = c1;
        }
    }
    xi

}
