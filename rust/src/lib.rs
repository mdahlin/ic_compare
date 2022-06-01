mod rust_fn; 

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};


#[pymodule]
fn rs_agg_methods(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn ic_reorder<'py>(py: Python<'py>,
                       x_py: PyReadonlyArray2<f64>,
                       ctar_py: PyReadonlyArray2<f64>
    ) -> &'py PyArray2<f64> {
        let x = x_py.as_array().to_owned();
        let ctar = ctar_py.as_array().to_owned();
        let res = rust_fn::ic_reorder(x, ctar);
        res.into_pyarray(py)
    }
    Ok(())
}

