# Condition number estimation in Rust

This crate implements the matrix 1-norm estimator by [Higham and Tisseur]
(Algorithm 2.4 on page 7 (1190)) in the linked PDF.

It uses the excellent [`rust-ndarray`] crate for matrix storage.

[Higham and Tisseur]: http://eprints.ma.man.ac.uk/321/1/covered/MIMS_ep2006_145.pdf
[`rust-ndarray`]: https://github.com/rust-ndarray/ndarray

## Todo

Right now, this crate only returns the 1-norm, and further allocates anew on every call
to `normest1`. It does not currently yield the vectors necessary to calculate the actual
condition number. This isn't hard to do, but needs to be implemented... Outstanding are:

+ [ ] Return vectors required for calculating the 1-norm;
+ [x] Create a struct holding the necessary temporaries to repeatedly call `normest1` without extra allocation.
+ [ ] Implement extra tests to mimic the numerical experiments in [Higham and Tisseur].
+ [ ] Make some nice docs.

## Example usage

The example below generates a random matrix `a` and estimates its 1-norm. On average, this gives
pretty good results. Of course, there are some matrices where this algorithm severely underestimates
the actual 1-norm. See [Higham and Tisseur] for more.

**Important:** You need to explicitly link to a BLAS + LAPACK provider such as `openblas_src`.
See the explanations given at the [`blas-lapack-rs` organization].

[`blas-lapack-rs` organization]: https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki

```rust
extern crate openblas_src; // Need to declare `openblas_src` (or some other BLAS provider) explicitly to link to a BLAS library.

use ndarray::{
    prelude::*,
    Zip,
};
use ndarray_rand::RandomExt;
use rand::{
    SeedableRng,
};
use rand::distributions::StandardNormal;
use rand_xoshiro::Xoshiro256Plus;

fn main() {
    let t = 2;
    let n = 100;
    let itmax = 5;

    let mut rng = Xoshiro256Plus::seed_from_u64(1234);
    let distribution = StandardNormal;

    let mut a = Array::random_using((n, n), distribution, &mut rng);
    a.mapv_inplace(|x| 1.0/x);

    let estimated = condest::normest1(&a, t, itmax);
    let expected = {
        let (layout, a_slice) = if let Some(a_slice) = a.as_slice() {
            (cblas::Layout::RowMajor, a_slice)
        } else if let Some(a_slice) = a.as_slice_memory_order() {
            (cblas::Layout::ColumnMajor, a_slice)
        } else {
            panic!("Matrix not contiguous!")
        };

        unsafe {
            lapacke::dlange(
            layout,
            b'1',
            n as i32,
            n as i32,
            a_slice,
            n as i32,
        )}
    };

    assert_eq!(estimated, expected);
}
```
