//! This crate implements the matrix 1-norm estimator by [Higham and Tisseur].
//!
//! [Higham and Tisseur]: http://eprints.ma.man.ac.uk/321/1/covered/MIMS_ep2006_145.pdf

use ndarray::{
    prelude::*,
    ArrayBase,
    Data,
    DataMut,
    Dimension,
    Ix1,
    Ix2,
    s,
};
use ndarray_rand::RandomExt;
use ordered_float::NotNan;
use rand::{
    distributions::{
        Distribution,
        Uniform,
    },
    Rng,
    SeedableRng,
};
use rand_xorshift::XorShiftRng;
use std::collections::BTreeSet;
use std::cmp;
use std::slice;

/// Estimates the 1-norm of matrix `a`.
///
/// The parameter `t` is the number of vectors that have to fulfill some bound. See [Higham,
/// Tisseur] for more information. `itmax` is the maximum number of sweeps permitted.
///
/// [Higham, Tisseur]: http://eprints.ma.man.ac.uk/321/1/covered/MIMS_ep2006_145.pdf
pub fn normest1(a_matrix: Array2<f64>, t: usize, itmax: usize) -> f64 {
    assert!(itmax > 1);
    let (n_rows, n_cols) = a_matrix.dim();
    assert!(t < n_cols);

    let mut thread_rng = rand::thread_rng();
    // TODO: Exchange for xoshiro, once it's merged into rust-rand
    let mut rng = XorShiftRng::from_rng(&mut thread_rng).expect("Rng initialization failed.");
    let distribution = Uniform::new_inclusive(-1., 1.);

    let mut sign_matrix = unsafe { Array2::<f64>::uninitialized((n_rows, t)) };
    let mut sign_matrix_old = unsafe { Array2::<f64>::uninitialized((n_rows, t)) };
    // X
    // “We now explain our choice of starting matrix. We take the first column of X to be the
    // vector of 1s, which is the starting vector used in Algorithm 2.1. This has the advantage
    // that for a matrix with nonnegative elements the algorithm converges with an exact estimate
    // on the second iteration, and such matrices arise in applications, for example as a
    // stochastic matrix or as the inverse of an M -matrix.”
    //
    // “The remaining columns are chosen as rand {− 1 , 1 } , with a check for and correction of
    // parallel columns, exactly as for S in the body of the algorithm. We choose random vectors
    // because it is difficult to argue for any particular fixed vectors and because randomness
    // lessens the importance of counterexamples (see the comments in the next section).”
    let mut x_matrix = Array::random_using((n_rows, t), distribution, &mut rng);
    x_matrix.column_mut(0).fill(1. / n_rows as f64);

    // Y
    // NOTE: We are also reusing `y_matrix` when checking whether `sign_matrix` and
    // `sign_matrix_old` have any parallel columns between themselves.
    let mut y_matrix = unsafe { Array2::<f64>::uninitialized((n_rows, t)) };
    // Z
    let mut z_matrix = unsafe { Array2::<f64>::uninitialized((n_rows, t)) };

    let mut estimate = 0.0;
    let mut best_index = 0;
    let mut w = unsafe { Array1::uninitialized(n_rows) };

    // hᵢ= ‖Z(i,:)‖_∞
    let mut h = vec![unsafe { NotNan::unchecked_new(0.0) }; n_cols];

    // indᵢ= i, i:n
    let mut indices: Vec<usize> = (0..n_cols).collect();
    let mut indices_history = BTreeSet::new();

    let mut column_is_parallel = vec![false; n_cols];
    'optimization_loop: for k in 0..itmax {
        // Y = A X
        {
            let (a_slice, a_layout) = as_slice_with_layout(&a_matrix).expect("Matrix `a` not contiguous.");
            let (x_slice, x_layout) = as_slice_with_layout(&x_matrix).expect("Matrix `x` not contiguous.");
            let (y_slice, y_layout) = as_slice_with_layout_mut(&mut y_matrix).expect("Matrix `y` not contiguous.");
            assert_eq!(a_layout, x_layout);
            assert_eq!(a_layout, y_layout);
            let layout = a_layout;
            unsafe {
                cblas::dgemm(
                    layout,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    n_rows as i32,
                    t as i32,
                    n_cols as i32,
                    1.0,
                    a_slice,
                    n_rows as i32,
                    x_slice,
                    n_cols as i32,
                    0.0,
                    y_slice,
                    n_rows as i32,
                )
            }
        }

        // est = max{‖Y(:,j)‖₁ : j = 1:t}
        let (max_norm_index, max_norm) = matrix_onenorm_with_index(&y_matrix);
        // if est > est_old or k=2
        if max_norm > estimate || k == 1 {
            // ind_best = indⱼ where est = ‖Y(:,j)‖₁, w = Y(:, ind_best)
            estimate = max_norm;
            best_index = max_norm_index;
            w.assign(&y_matrix.column(best_index));
        } else if k > 1 && max_norm <= estimate {
            break 'optimization_loop
        }

        if k >= itmax {
            break 'optimization_loop
        }

        // est_old = est, Sold = S
        // NOTE: We don't “save” the old estimate, because we are using max_norm as another name
        // for the new estimate instead of overwriting/reusing est.
        sign_matrix_old.assign(&sign_matrix);

        // S = sign(Y)
        assign_signum_of_array(
            &y_matrix,
            &mut sign_matrix
        );

        // TODO: Combine the test checking for parallelity between _all_ columns between S
        // and S_old with the “if t > 1” test below.
        //
        // > If every column of S is parallel to a column of Sold, goto (6), end
        //
        // NOTE: We are reusing `y_matrix` here as a temporary value.
        if are_all_columns_parallel_between(&sign_matrix_old, &sign_matrix, &mut y_matrix) {
            break 'optimization_loop;
        }

        // FIXME: Is an explicit if condition here necessary
        if t > 1 {
            // Ensure that no column of S is parallel to another column of S
            // or to a column of Sold by replacing columns of S by rand{-1,+1}
            //
            // NOTE: We are reusing `y_matrix` here as a temporary value.
            resample_parallel_columns(
                &mut sign_matrix,
                &sign_matrix_old,
                &mut y_matrix,
                &mut column_is_parallel,
                &mut rng,
                &distribution
            );
        }

        // Z = A^T S
        let (a_slice, a_layout) = as_slice_with_layout(&a_matrix).expect("Matrix `a` is not contiguous.");
        let (sign_slice, sign_layout) = as_slice_with_layout(&sign_matrix).expect("Matrix `sign` is not contiguous.");
        let (z_slice, z_layout) = as_slice_with_layout_mut(&mut z_matrix).expect("Matrix `z` is not contiguous.");
        assert_eq!(a_layout, sign_layout);
        assert_eq!(a_layout, z_layout);
        let layout = a_layout;
        unsafe {
            cblas::dgemm(
                layout,
                cblas::Transpose::Ordinary,
                cblas::Transpose::None,
                n_cols as i32, // n_cols of Op(a)
                t as i32,
                n_rows as i32,
                1.0,
                a_slice,
                n_rows as i32,
                sign_slice,
                n_rows as i32,
                0.0,
                z_slice,
                n_cols as i32,
            )
        }

        // hᵢ= ‖Z(i,:)‖_∞
        let mut max_h = 0.0;
        for (column, h_element) in z_matrix.gencolumns().into_iter().zip(h.iter_mut()) {
            // Into is for converting f64 to NotNan
            let h = vector_maxnorm(&column);
            max_h = if h > max_h { h } else { max_h };
            *h_element = h.into();
        }

        // TODO: This test for equality needs an approximate equality test instead.
        if k > 0 && max_h == h[best_index].into() {
            break 'optimization_loop
        }

        // > Sort h so that h_1 >= ... >= h_n and re-order correspondingly.
        // NOTE: h itself doesn't need to be reordered. Only the order of
        // the indices is relevant.
        indices.sort_unstable_by(|i, j| h[*j].cmp(&h[*i]));

        x_matrix.fill(0.0);
        if t > 1 {
            // > Replace ind(1:t) by the first t indices in ind(1:n) that are not in ind_hist.
            //
            // > X(:, j) = e_ind_j, j = 1:t
            //
            // > ind_hist = [ind_hist ind(1:t)]
            //
            // NOTE: It's not actually needed to operate on the `indices` vector. What's important
            // is that the history of indices, `indices_history`, gets updated with visited indices,
            // and that each column of `x_matrix` is assigned that unit vector that is defined by the
            // respective index.
            //
            // If so many indices have already been used that `n_cols - indices_history.len() < t`
            // (which means that we have less than `t` unused indices remaining), we have to use a few
            // historical indices when filling up the columns in `x_matrix`. For that, we put the
            // historical indices after the fresh indices, but otherwise keep the order induced by `h`
            // above.
            let fresh_indices = cmp::min(t, n_cols - indices_history.len());
            if fresh_indices == 0 {
                break 'optimization_loop;
            }
            let mut current_column_fresh = 0;
            let mut current_column_historical = fresh_indices;
            let mut index_iterator = indices.iter();

            let mut all_first_t_in_history = true;

            // Iterate over the first t indices
            for i in (&mut index_iterator).take(t) {
                if !indices_history.contains(i) {
                    all_first_t_in_history = false;
                    x_matrix[(current_column_fresh, *i)] = 1.0;
                    current_column_fresh += 1;
                    indices_history.insert(*i);
                } else if current_column_historical < t {
                    x_matrix[(current_column_historical, *i)] = 1.0;
                    current_column_historical += 1;
                }
            }

            // > if ind(1:t) is contained in ind_hist, goto (6), end
            if all_first_t_in_history {
                break 'optimization_loop;
            }

            // Iterate over the remaining indices
            'fill_x: for i in index_iterator {
                if current_column_fresh > t {
                    break 'fill_x;
                }
                if !indices_history.contains(i) {
                    x_matrix[(current_column_fresh, *i)] = 1.0;
                    current_column_fresh += 1;
                    indices_history.insert(*i);
                } else if current_column_historical < t {
                    x_matrix[(current_column_historical, *i)] = 1.0;
                    current_column_historical += 1;
                }
            }
        }
    }

    estimate
}

/// Assigns the sign of matrix `a` to matrix `b`.
///
/// Panics if matrices `a` and `b` have different shape and strides, or if either underlying array is
/// non-contiguous. This is to make sure that the iteration order over the matrices is the same.
fn assign_signum_of_array<S1, S2, D>(a: &ArrayBase<S1, D>, b: &mut ArrayBase<S2, D>)
    where S1: Data<Elem=f64>,
          S2: DataMut<Elem=f64>,
          D: Dimension
{
    assert_eq!(a.strides(), b.strides());
    let (a_slice, a_layout) = as_slice_with_layout(a).expect("Matrix `a` is not contiguous.");
    let (b_slice, b_layout) = as_slice_with_layout_mut(b).expect("Matrix `b` is not contiguous.");
    assert_eq!(a_layout, b_layout);

    signum_of_slice(a_slice, b_slice);
}

fn signum_of_slice(source: &[f64], destination: &mut [f64]) {
    for (s, d) in source.iter().zip(destination) {
        *d = s.signum();
    }
}

/// Calculate the onenorm of a vector (an `ArrayBase` with dimension `Ix1`).
fn vector_onenorm<S>(a: &ArrayBase<S, Ix1>) -> f64
    where S: Data<Elem=f64>,
{
    let stride = a.strides()[0];
    assert!(stride >= 0);
    let stride = stride as usize;
    let n_elements = a.len();
    let a_slice = {
        let a = a.as_ptr();
        let total_len = n_elements * stride;
        unsafe { slice::from_raw_parts(a, total_len) }
    };

    unsafe {
        cblas::dasum(n_elements as i32, a_slice, stride as i32)
    }
}

/// Calculate the maximum norm of a vector (an `ArrayBase` with dimension `Ix1`).
fn vector_maxnorm<S>(a: &ArrayBase<S, Ix1>) -> f64
    where S: Data<Elem=f64>
{
    let stride = a.strides()[0];
    assert!(stride >= 0);
    let stride = stride as usize;
    let n_elements = a.len();
    let a_slice = {
        let a = a.as_ptr();
        let total_len = n_elements * stride;
        unsafe { slice::from_raw_parts(a, total_len) }
    };

    let idx = unsafe {
        cblas::idamax(
            n_elements as i32,
            a_slice,
            stride as i32,
        ) as usize
    };
    f64::abs(a[idx])
}

// /// Calculate the onenorm of a matrix (an `ArrayBase` with dimension `Ix2`).
// fn matrix_onenorm<S>(a: &ArrayBase<S, Ix2>) -> f64
//     where S: Data<Elem=f64>,
// {
//     let (n_rows, n_cols) = a.dim();
//     if let Some((a_slice, layout)) = as_slice_with_layout(a) {
//         let layout = match layout {
//             cblas::Layout::RowMajor => lapacke::Layout::RowMajor,
//             cblas::Layout::ColumnMajor => lapacke::Layout::ColumnMajor,
//         };
//         unsafe {
//             lapacke::dlange(
//                 layout,
//                 b'1',
//                 n_rows as i32,
//                 n_cols as i32,
//                 a_slice,
//                 n_rows as i32,
//             )
//         }
//     // Fall through case for non-contiguous arrays.
//     } else {
//         a.gencolumns().into_iter()
//             .fold(0.0, |max, column| {
//                 let onenorm = column.fold(0.0, |acc, element| { acc + f64::abs(*element) });
//                 if onenorm > max { onenorm } else { max }
//         })
//     }
// }

/// Returns the one-norm of a matrix `a` together with the index of that column for
/// which the norm is maximal.
fn matrix_onenorm_with_index<S>(a: &ArrayBase<S, Ix2>) -> (usize, f64)
    where S: Data<Elem=f64>,
{
    let mut max_norm = 0.0;
    let mut max_norm_index = 0;
    for (i, column) in a.gencolumns().into_iter().enumerate() {
        let norm = vector_onenorm(&column);
        if norm > max_norm {
            max_norm = norm;
            max_norm_index = i;
        }
    }
    (max_norm_index, max_norm)
}

/// Finds columns in the matrix `a` that are parallel to to some other column in `a`.
///
/// Assumes that all entries of `a` are either +1 or -1.
///
/// If column `j` of matrix `a` is parallel to some column `i`, `column_is_parallel[i]` is set to
/// `true`. The matrix `c` is used as an intermediate value for the matrix product `a^t * a`.
///
/// This function does not reset `column_is_parallel` to `false`. Entries that are `true` will be
/// assumed to be parallel and not checked.
///
/// Panics if arrays `a` and `c` don't have the same dimensions, or if the length of the slice
/// `column_is_parallel` is not equal to the number of columns in `a`.
fn find_parallel_columns_in<S1, S2> (
    a: &ArrayBase<S1, Ix2>,
    c: &mut ArrayBase<S2, Ix2>,
    column_is_parallel: &mut [bool]
)
    where S1: Data<Elem=f64>,
          S2: DataMut<Elem=f64>
{
    let a_dim = a.dim();
    let c_dim = c.dim();
    assert_eq!(a_dim, c_dim);

    let (n_rows, n_cols) = a_dim;

    assert_eq!(column_is_parallel.len(), n_cols);

    {
        let (a_slice, a_layout) = as_slice_with_layout(a).expect("Matrix `a` is not contiguous.");
        let (c_slice, c_layout) = as_slice_with_layout_mut(c).expect("Matrix `c` is not contiguous.");
        assert_eq!(a_layout, c_layout);
        let layout = a_layout;
        unsafe {
            // TODO: Check if this really gives the product a^t * a
            cblas::dsyrk(
                layout,
                cblas::Part::Upper,
                cblas::Transpose::Ordinary,
                n_rows as i32,
                n_cols as i32,
                1.0,
                a_slice,
                n_rows as i32,
                0.0,
                c_slice,
                n_rows as i32,
            );
        }
    }
    // c is upper triangular and contains all pair-wise vector products:
    //
    // x x x x x
    // . x x x x
    // . . x x x
    // . . . x x
    // . . . . x

    'rows: for (i, row) in c.genrows().into_iter().enumerate() {
        // Skip if the column is already found to be parallel or if we are checking
        // the last column
        if column_is_parallel[i] || i >= n_cols - 1 { continue 'rows; }
        for (j, element) in row.slice(s![i+1..]).iter().enumerate() {
            // Check if the vectors are parallel or anti-parallel
            if f64::abs(*element) == n_rows as f64 {
                column_is_parallel[i+j+1] = true;
            }
        }
    }
}

/// Checks whether any columns of the matrix `a` are parallel to any columns of `b`.
///
/// Assumes that we have parallelity only if all entries of two columns `a` and `b` are either +1
/// or -1.
///
/// `The matrix `c` is used as an intermediate value for the matrix product `a^t * b`.
///
/// `column_is_parallel[j]` is set to `true` if column `j` of matrix `a` is parallel to some column
/// `i` of the matrix `b`,
///
/// This function does not reset `column_is_parallel` to `false`. Entries that are `true` will be
/// assumed to be parallel and not checked.
///
/// Panics if arrays `a`, `b`, and `c` don't have the same dimensions, or if the length of the slice
/// `column_is_parallel` is not equal to the number of columns in `a`.
fn find_parallel_columns_between<S1, S2, S3> (
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    c: &mut ArrayBase<S3, Ix2>,
    column_is_parallel: &mut [bool],
)
    where S1: Data<Elem=f64>,
          S2: Data<Elem=f64>,
          S3: DataMut<Elem=f64>
{
    let a_dim = a.dim();
    let b_dim = b.dim();
    let c_dim = c.dim();
    assert_eq!(a_dim, b_dim);
    assert_eq!(a_dim, c_dim);

    let (n_rows, n_cols) = a_dim;

    assert_eq!(column_is_parallel.len(), n_cols);

    // Extra scope, because c_slice needs to be dropped after the dgemm
    {
        let (a_slice, a_layout) = as_slice_with_layout(&a).expect("Matrix `a` not contiguous.");
        let (b_slice, b_layout) = as_slice_with_layout(&b).expect("Matrix `b` not contiguous.");
        let (c_slice, c_layout) = as_slice_with_layout_mut(c).expect("Matrix `c` not contiguous.");

        assert_eq!(a_layout, b_layout);
        assert_eq!(a_layout, c_layout);

        let layout = a_layout;

        unsafe {
            cblas::dgemm(
                layout,
                cblas::Transpose::Ordinary,
                cblas::Transpose::None,
                n_rows as i32,
                n_cols as i32,
                n_cols as i32,
                1.0,
                a_slice,
                n_cols as i32,
                b_slice,
                n_cols as i32,
                0.0,
                c_slice,
                n_rows as i32,
            );
        }
    }

    // We are iterating over the rows because it's more memory efficient (for row-major array).  In
    // terms of logic there is no difference: we simply check if the current column of a (that's
    // the outer loop) is parallel to any column of b (inner loop). By iterating via columns we would check if
    // any column of a is parallel to the, in that case, current column of b.
    // TODO:  Implement for column major arrays.
    'rows: for (i, row) in c.genrows().into_iter().enumerate() {
        // Skip if the column is already found to be parallel the last column.
        if column_is_parallel[i] { continue 'rows; }
        for element in row {
            if f64::abs(*element) == n_rows as f64 {
                column_is_parallel[i] = true;
                continue 'rows;
            }
        }
    }
}


/// Check if every column in `a` is parallel to some column in `b`.
///
/// Assumes that we have parallelity only if all entries of two columns `a` and `b` are either +1
/// or -1.
fn are_all_columns_parallel_between<S1, S2> (
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S1, Ix2>,
    c: &mut ArrayBase<S2, Ix2>,
) -> bool
    where S1: Data<Elem=f64>,
          S2: DataMut<Elem=f64>
{
    let a_dim = a.dim();
    let b_dim = b.dim();
    let c_dim = c.dim();
    assert_eq!(a_dim, b_dim);
    assert_eq!(a_dim, c_dim);

    let (n_rows, n_cols) = a_dim;

    // Extra scope, because c_slice needs to be dropped after the dgemm
    {
        let (a_slice, a_layout) = as_slice_with_layout(&a).expect("Matrix `a` not contiguous.");
        let (b_slice, b_layout) = as_slice_with_layout(&b).expect("Matrix `b` not contiguous.");
        let (c_slice, c_layout) = as_slice_with_layout_mut(c).expect("Matrix `c` not contiguous.");

        assert_eq!(a_layout, b_layout);
        assert_eq!(a_layout, c_layout);

        let layout = a_layout;

        unsafe {
            cblas::dgemm(
                layout,
                cblas::Transpose::Ordinary,
                cblas::Transpose::None,
                n_rows as i32,
                n_cols as i32,
                n_cols as i32,
                1.0,
                a_slice,
                n_cols as i32,
                b_slice,
                n_cols as i32,
                0.0,
                c_slice,
                n_rows as i32,
            );
        }
    }

    // We are iterating over the rows because it's more memory efficient (for row-major array).  In
    // terms of logic there is no difference: we simply check if a specific column of a is parallel
    // to any column of b. By iterating via columns we would check if any column of a is parallel
    // to a specific column of b.
    'rows: for row in c.genrows() {
        for element in row {
            // If a parallel column was found, cut to the next one.
            if f64::abs(*element) == n_rows as f64 { continue 'rows; }
        }
        // This return statement should only be reached if not a single column parallel to the
        // current one was found.
        return false;
    }
    true
}

/// Find parallel columns in matrix `a` and columns in `a` that are parallel to any columns in
/// matrix `b`, and replace those with random vectors. Elements are sampled from `distribution`.
/// Returns `true` if resampling has taken place.
fn resample_parallel_columns<S1, S2, S3, D, R>(
    a: &mut ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    c: &mut ArrayBase<S3, Ix2>,
    column_is_parallel: &mut [bool],
    rng: &mut R,
    distribution: &D,
) -> bool
    where S1: DataMut<Elem=f64>,
          S2: Data<Elem=f64>,
          S3: DataMut<Elem=f64>,
          D: Distribution<f64>,
          R: Rng
{
    column_is_parallel.iter_mut().for_each(|x| {*x = false;});
    find_parallel_columns_in(a, c, column_is_parallel);
    find_parallel_columns_between(a, b, c, column_is_parallel);
    let mut has_resampled = false;
    let mut rand_iter = distribution.sample_iter(rng);
    for (i, is_parallel) in column_is_parallel.into_iter().enumerate() {
        if *is_parallel {
            a.column_mut(i).map_inplace(|x| { *x = rand_iter.next().unwrap(); });
            has_resampled = true;
        }
    }
    has_resampled
}


/// Returns slice and layout underlying an array `a`.
fn as_slice_with_layout<S, T, D>(a: &ArrayBase<S, D>) -> Option<(&[T], cblas::Layout)>
    where S: Data<Elem=T>,
          D: Dimension
{
    if let Some(a_slice) = a.as_slice() {
        Some((a_slice, cblas::Layout::RowMajor))
    } else if let Some(a_slice) = a.as_slice_memory_order() {
        Some((a_slice, cblas::Layout::ColumnMajor))
    } else {
        None
    }
}

/// Returns mutable slice and layout underlying an array `a`.
fn as_slice_with_layout_mut<S, T, D>(a: &mut ArrayBase<S, D>) -> Option<(&mut [T], cblas::Layout)>
    where S: DataMut<Elem=T>,
          D: Dimension
{
    if a.as_slice_mut().is_some() {
        Some((a.as_slice_mut().unwrap(), cblas::Layout::RowMajor))
    } else if a.as_slice_memory_order_mut().is_some() {
        Some((a.as_slice_memory_order_mut().unwrap(), cblas::Layout::ColumnMajor))
    } else {
        None
    }
    // XXX: The above is a workaround for Rust not having non-lexical lifetimes yet.
    // More information here:
    // http://smallcultfollowing.com/babysteps/blog/2016/04/27/non-lexical-lifetimes-introduction/#problem-case-3-conditional-control-flow-across-functions
    // if let Some(slice) = a.as_slice_mut() {
    //     Some((slice, cblas::Layout::RowMajor))
    // } else if let Some(slice) = a.as_slice_memory_order_mut() {
    //     Some((slice, cblas::Layout::ColumnMajor))
    // } else {
    //     None
    // }
}
