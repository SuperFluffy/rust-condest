# Condition number and 1-norm estimation in Rust

This crate implements the matrix 1-norm estimator by [Higham and Tisseur],
Algorithm 2.4 on page 7 (1190) in the linked document. It allows for 1-norm estimation
of a single matrices, `A`, matrix powers, `A^m`, and matrix products, `A1 * A2 * ... * An`,
which can be cheaper than explicitly calculating them.

It uses the excellent [`rust-ndarray`] crate for matrix storage.

[Higham and Tisseur]: http://eprints.ma.man.ac.uk/321/1/covered/MIMS_ep2006_145.pdf
[`rust-ndarray`]: https://github.com/rust-ndarray/ndarray

## Example usage

The example below generates a random matrix `a` and estimates its 1-norm. On
average, this gives pretty good results. Of course, there are some matrices
where this algorithm severely underestimates the actual 1-norm. See [Higham and
Tisseur] for more.

`condest::normest1` creates a `Normest1` struct, uses it to estimate the
1-norm, and throws it away. If you want to repeatedly estimate 1-norms of
matrices of the same dimensions, initialize `Normest1` and call `normest1`,
`normest1_pow` or `normest1_prod` on it.


**Important:** You need to explicitly link to a BLAS + LAPACK provider such as
`openblas_src`. See the explanations given at the [`blas-lapack-rs` organization].

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

## Todo

Right now, only 1-norm estimates are exposed. The vectors needed to estimate
the condition number are implemented, but are not yet accessible through an API.
Outstanding points are:

+ [ ] Return vectors required for calculating the 1-norm;
+ [x] Create a struct holding the necessary temporaries to repeatedly call `normest1` without extra allocation.
+ [ ] Implement extra tests to mimic the numerical experiments in [Higham and Tisseur].
+ [ ] Make some nice docs.

