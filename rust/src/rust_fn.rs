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

/// Iman Conover Reordering
///
/// Applies transformation proposed by Iman and Conover
/// (1982) to reorder simulations so column-wise correlation
/// matches the provided target correlation
///
/// * `x` - n by y array of simulations to be reordered
/// * `ctar` - y by y target correlation matrix
///
/// Note: transformation will not result in simulations perfectly
///       matching the target correlation
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

    // TODO: more efficient (and separate function)
    let mut y = Array::from_elem((n_sim, n_col), -999.);
    for i in 0..n_sim {
        for j in 0..n_col {
            y[[i, j]] = x[[r[[i, j]], j]]
        }
    }
    y
}

/// Sum of Correlation Error
///
/// Calculates the total absolute difference between the
/// upper triangular correlation factors
///
/// * `c1` - a correlation matrix
/// * `c2` - a correlation matrix
///
/// Notes: assumes `c1` and `c2` symmetric and of the same dimension.
///        Could probably do something with Zip instead
fn sum_cerr(c1: &Array2<f64>, c2: &Array2<f64>) -> f64 {
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

/// Iterative Perturbed Cholesky Iman Conover Reordering
///
/// Expands on Iman Conover Reordering with additional perturbation
/// to get the resulting simulations closer to the target correlation
/// than only the Iman Conover methodology alone
///
/// * `x` - n by y array of simulations to be reordered
/// * `ctar` - y by y target correlation matrix
///
/// Notes: does not just call `ic_reorder` because other calculated
///        pieces are needed on top of the output of the funtion
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

    // set up for the IPC portion
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

/// Max Axis Error
///
/// Calculates which index (for a specified axis) has th
/// largest absolute element difference
///
/// * `c1` - a correlation matrix
/// * `c2` - a correlation matrix
/// * `axis` - Axis(0) for columns; Axis(1) for rows
///
/// Notes: assumes `c1` and `c2` are of the same dimension.
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

/// Iterative Local Search Reordering
///
/// ILS process to reduce error between target correlation and
/// output simlulations column-wise correlation. Faster/more
/// efficient to call `ic_reorder` or `ic_ipc_reorder` first
/// and use that was the `x` input
///
/// * `x` - n by y array of simulations to be reordered
/// * `ctar` - y by y target correlation matrix
/// * `niter` - number of iterations to attempt improvments
///
/// Notes: `x` is generally the result of an another reordering function.
///        Could change `niter` to some sort of error target.
///        Might want to check if reassigning versus cloning would change
///          runtime
pub fn ils_reorder(x: Array2<f64>, ctar: Array2<f64>, niter: usize) -> Array2<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_cerr() {
        let c1: Array2<f64> = array![
            [ 1.00, 0.50, 0.25, 0.05],
            [ 0.50, 1.00, 0.00, 0.30],
            [ 0.25, 0.00, 1.00, 0.00],
            [ 0.05, 0.30, 0.00, 1.00]
        ];

        let mut c2 = c1.clone();

        for i in 0..c1.nrows() {
            for j in 0..c1.ncols() {
                if i != j {
                    c2[[i, j]] -= 1.
                }
            }
        }
        let result = sum_cerr(&c1, &c2);


        assert_eq!(result, 6f64);
    }

    #[test]
    fn test_ils_reorder_improves_corr() {
        use ndarray::stack;
        let x: Array2<f64> = stack![
            Axis(1),
            Array::range(0., 1000., 1.),
            Array::range(0., 1000., 1.),
            Array::range(0., 1000., 1.),
            Array::range(0., 1000., 1.)
        ];

        let c1: Array2<f64> = array![
            [ 1.00, 0.50, 0.25, 0.05],
            [ 0.50, 1.00, 0.00, 0.30],
            [ 0.25, 0.00, 1.00, 0.00],
            [ 0.05, 0.30, 0.00, 1.00]
        ];

        let initial_corr = x.t().pearson_correlation().unwrap();
        let initial_err = sum_cerr(&c1, &initial_corr);

        let post_ils = ils_reorder(x, c1, 500); // technically could fail
        let post_ils_corr = post_ils.t().pearson_correlation().unwrap();
        let post_ils_err = sum_cerr(&initial_corr, &post_ils_corr);

        assert!(post_ils_err < initial_err);

    }
}
