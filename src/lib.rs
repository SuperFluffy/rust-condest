//! This crate implements the matrix 1-norm estimator by [Higham and Tisseur].
//!
//! [Higham and Tisseur]: http://eprints.ma.man.ac.uk/321/1/covered/MIMS_ep2006_145.pdf
use alga::general::{ComplexField, SupersetOf};
use blas_traits::BlasScalar;
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
use num_traits::{Float, Zero};
use ordered_float::NotNan;
use rand::{
    Rng,
    SeedableRng,
    thread_rng,
};
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::BTreeSet;
use std::cmp;
use std::slice;
use std::any::TypeId;

pub struct Normest1<T: BlasScalar> where {
    n: usize,
    t: usize,
    rng: Xoshiro256StarStar,
    x_matrix: Array2<T>,
    y_matrix: Array2<T>,
    z_matrix: Array2<T>,
    w_vector: Array1<T>,
    sign_matrix: Array2<T>,
    sign_matrix_old: Array2<T>,
    column_is_parallel: Vec<bool>,
    indices: Vec<usize>,
    indices_history: BTreeSet<usize>,
    h: Vec<NotNan<f64>>,
}

/// A trait to generalize over 1-norm estimates of a matrix `A`, matrix powers `A^m`,
/// or matrix products `A1 * A2 * ... * An`.
///
/// In the 1-norm estimator, one repeatedly constructs a matrix-matrix product between some n×n
/// matrix X and some other n×t matrix Y. If one wanted to estimate the 1-norm of a matrix m times
/// itself, X^m, it might thus be computationally less expensive to repeatedly apply
/// X * ( * ( X ... ( X * Y ) rather than to calculate Z = X^m = X * X * ... * X and then apply Z *
/// Y. In the first case, one has several matrix-matrix multiplications with complexity O(m*n*n*t),
/// while in the latter case one has O(m*n*n*n) (plus one more O(n*n*t)).
///
/// So in case of t << n, it is cheaper to repeatedly apply matrix multiplication to the smaller
/// matrix on the RHS, rather than to construct one definite matrix on the LHS.  Of course, this is
/// modified by the number of iterations needed when performing the norm estimate, sustained
/// performance of the matrix multiplication method used, etc.
///
/// It is at the designation of the user to check what is more efficient: to pass in one definite
/// matrix or choose the alternative route described here.
trait LinearOperator<T: BlasScalar> {
    fn multiply_matrix<S>(&self, b: &mut ArrayBase<S, Ix2>, c: &mut ArrayBase<S, Ix2>, transpose: bool)
        where S: DataMut<Elem=T>;
}

impl<S1, T: BlasScalar> LinearOperator<T> for ArrayBase<S1, Ix2>
    where S1: Data<Elem=T>,
{
    fn multiply_matrix<S2>(&self, b: &mut ArrayBase<S2, Ix2>, c: &mut ArrayBase<S2, Ix2>, transpose: bool)
        where S2: DataMut<Elem=T>
    {
        let (n_rows, n_cols) = self.dim();
        assert_eq!(n_rows, n_cols, "Number of rows and columns does not match: `self` has to be a square matrix");
        let n = n_rows;

        let (b_n, b_t) = b.dim();
        let (c_n, c_t) = b.dim();

        assert_eq!(n, b_n, "Number of rows of b not equal to number of rows of `self`.");
        assert_eq!(n, c_n, "Number of rows of c not equal to number of rows of `self`.");

        assert_eq!(b_t, c_t, "Number of columns of b not equal to number of columns of c.");

        let t = b_t;

        let (a_slice, a_layout) = as_slice_with_layout(self).expect("Matrix `self` not contiguous.");
        let (b_slice, b_layout) = as_slice_with_layout(b).expect("Matrix `b` not contiguous.");
        let (c_slice, c_layout) = as_slice_with_layout_mut(c).expect("Matrix `c` not contiguous.");

        assert_eq!(a_layout, b_layout);
        assert_eq!(a_layout, c_layout);

        let layout = a_layout;

        let a_transpose = if transpose {
            cblas::Transpose::Conjugate     // Simple transpose in real case
        } else {
            cblas::Transpose::None
        };
        T::gemm(layout, a_transpose, cblas::Transpose::None,
                 n as i32, t as i32, n as i32,
                 T::from_subset(&1.0), a_slice, n as i32, b_slice, t as i32,
                 T::from_subset(&0.0), c_slice, t as i32,);
    }
}

impl<S1, T: BlasScalar> LinearOperator<T> for [&ArrayBase<S1, Ix2>]
    where S1: Data<Elem=T>
{
    fn multiply_matrix<S2>(&self, b: &mut ArrayBase<S2, Ix2>, c: &mut ArrayBase<S2, Ix2>, transpose: bool)
        where S2: DataMut<Elem=T>
    {
        if self.len() > 0 {
            let mut reversed;
            let mut forward;

            // TODO: Investigate, if an enum instead of a trait object might be more performant.
            // This probably doesn't matter for large matrices, but could have a measurable impact
            // on small ones.
            let a_iter: &mut dyn DoubleEndedIterator<Item=_> = if transpose {
                reversed = self.iter().rev();
                &mut reversed
            } else {
                forward = self.iter();
                &mut forward
            };
            let a = a_iter.next().unwrap(); // Ok because of if condition
            a.multiply_matrix(b, c, transpose);

            // NOTE: The swap in the loop body makes use of the fact that in all instances where
            // `multiply_matrix` is used, the values potentially stored in `b` are not required
            // anymore.
            for a in a_iter {
                std::mem::swap(b, c);
                a.multiply_matrix(b, c, transpose);
            }
        }
    }
}

impl<S1, T: BlasScalar> LinearOperator<T> for (&ArrayBase<S1, Ix2>, usize)
    where S1: Data<Elem=T>
{
    fn multiply_matrix<S2>(&self, b: &mut ArrayBase<S2, Ix2>, c: &mut ArrayBase< S2, Ix2>, transpose: bool)
        where S2: DataMut<Elem=T>
    {
        let a = self.0;
        let m = self.1;
        if m > 0 {
            a.multiply_matrix(b, c, transpose);
            for _ in 1..m {
                std::mem::swap(b, c);
                self.0.multiply_matrix(b, c, transpose);
            }
        }
    }
}

impl<T: BlasScalar> Normest1<T> {
    pub fn new(n: usize, t: usize) -> Self {
        assert!(t <= n, "Cannot have more iteration columns t than columns in the matrix.");
        let rng = Xoshiro256StarStar::from_rng(&mut thread_rng()).expect("Rng initialization failed.");
        let x_matrix = unsafe { Array2::uninitialized((n, t)) };
        let y_matrix = unsafe { Array2::uninitialized((n, t)) };
        let z_matrix = unsafe { Array2::uninitialized((n, t)) };

        let w_vector = unsafe { Array1::uninitialized(n) };

        let sign_matrix = unsafe { Array2::uninitialized((n, t)) };
        let sign_matrix_old = unsafe { Array2::uninitialized((n, t)) };

        let column_is_parallel = vec![false; t];

        let indices = (0..n).collect();
        let indices_history = BTreeSet::new();

        let h = vec![unsafe { NotNan::unchecked_new(0.0) }; n];

        Normest1 {
            n,
            t,
            rng,
            x_matrix,
            y_matrix,
            z_matrix,
            w_vector,
            sign_matrix,
            sign_matrix_old,
            column_is_parallel,
            indices,
            indices_history,
            h,
        }
    }

    fn calculate<L>(&mut self, a_linear_operator: &L, itmax: usize) -> T::RealField
        where L: LinearOperator<T> + ?Sized
    {
        assert!(itmax > 1, "normest1 is undefined for iterations itmax < 2");

        // Explicitly empty the index history; all other quantities will be overwritten at some
        // point.
        self.indices_history.clear();

        let n = self.n;
        let t = self.t;

        let sample = [-T::one(), T::one()];

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
        {
            let rng_mut = &mut self.rng;
            self.x_matrix.mapv_inplace(|_| sample[rng_mut.gen_range(0, sample.len())] );
            self.x_matrix.column_mut(0).fill(T::one());
        }

        // Resample the x_matrix to make sure no columns are parallel
        find_parallel_columns_in(&self.x_matrix, &mut self.y_matrix, &mut self.column_is_parallel);
        for (i, is_parallel) in self.column_is_parallel.iter().enumerate() {
            if *is_parallel {
                resample_column(&mut self.x_matrix, i, &mut self.rng, &sample);
            }
        }

        // Set all columns to unit vectors
        self.x_matrix.mapv_inplace(|x| (x / T::from_usize(n).unwrap()) );

        let mut estimate = T::RealField::zero();
        let mut best_index = 0;

        'optimization_loop: for k in 0..itmax {

            // Y = A X
            a_linear_operator.multiply_matrix(&mut self.x_matrix, &mut self.y_matrix, false);

            // est = max{‖Y(:,j)‖₁ : j = 1:t}
            let (max_norm_index, max_norm) = matrix_onenorm_with_index(&self.y_matrix);

            // if est > est_old or k=2
            if max_norm > estimate || k == 1 {
                // ind_best = indⱼ where est = ‖Y(:,j)‖₁, w = Y(:, ind_best)
                estimate = max_norm;
                best_index = self.indices[max_norm_index];
                self.w_vector.assign(&self.y_matrix.column(max_norm_index));
            } else if k > 1 && max_norm <= estimate {
                break 'optimization_loop
            }

            if k >= itmax {
                break 'optimization_loop
            }

            // S = sign(Y)
            assign_signum_of_array(
                &self.y_matrix,
                &mut self.sign_matrix
            );

            // TODO: Combine the test checking for parallelity between _all_ columns between S
            // and S_old with the “if t > 1” test below.
            //
            // > If every column of S is parallel to a column of Sold, goto (6), end
            //
            // NOTE: We are reusing `y_matrix` here as a temporary value.
            if are_all_columns_parallel_between(&self.sign_matrix_old, &self.sign_matrix, &mut self.y_matrix) {
                break 'optimization_loop;
            }

            // FIXME: Is an explicit if condition here necessary?
            if t > 1 {
                // > Ensure that no column of S is parallel to another column of S
                // > or to a column of Sold by replacing columns of S by rand{-1,+1}
                //
                // NOTE: We are reusing `y_matrix` here as a temporary value.
                // Note: Parallel column test can be skipped in complex case
                if TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>(){
                    resample_parallel_columns(
                        &mut self.sign_matrix,
                        &self.sign_matrix_old,
                        &mut self.y_matrix,
                        &mut self.column_is_parallel,
                        &mut self.rng,
                        &sample,
                    );
                }
            }

            // > est_old = est, Sold = S
            // NOTE: Other than in the original algorithm, we store the sign matrix at this point
            // already. This way, we can reuse the sign matrix as additional workspace which is
            // useful when performing matrix multiplication with A^m or A1 A2 ... An (see the
            // description of the LinearOperator trait for explanation).
            //
            // NOTE: We don't “save” the old estimate, because we are using max_norm as another name
            // for the new estimate instead of overwriting/reusing est.
            self.sign_matrix_old.assign(&self.sign_matrix);

            // Z = A^T S
            //
            a_linear_operator.multiply_matrix(&mut self.sign_matrix, &mut self.z_matrix, true);

            // hᵢ= ‖Z(i,:)‖_∞
            let mut max_h = T::RealField::zero();
            for (row, h_element) in self.z_matrix.genrows().into_iter().zip(self.h.iter_mut()) {
                let h : T::RealField = vector_maxnorm(&row);
                max_h = if h > max_h { h } else { max_h };
                // Convert f64 to NotNan for using sort_unstable_by below
                *h_element = h.to_subset().unwrap().into();
            }

            // TODO: This test for equality needs an approximate equality test instead.
            if k > 0 && max_h == T::RealField::from_subset(&self.h[best_index].into()) {
                break 'optimization_loop
            }

            // > Sort h so that h_1 >= ... >= h_n and re-order correspondingly.
            // NOTE: h itself doesn't need to be reordered. Only the order of
            // the indices is relevant.
            {
                let h_ref = &self.h;
                self.indices.sort_unstable_by(|i, j| h_ref[*j].cmp(&h_ref[*i]));
            }

            self.x_matrix.fill(T::zero());
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
                let fresh_indices = cmp::min(t, n - self.indices_history.len());
                if fresh_indices == 0 {
                    break 'optimization_loop;
                }
                let mut current_column_fresh = 0;
                let mut current_column_historical = fresh_indices;
                let mut index_iterator = self.indices.iter();

                let mut all_first_t_in_history = true;
                // First, iterate over the first t sorted indices.
                for i in (&mut index_iterator).take(t) {
                    if !self.indices_history.contains(i) {
                        all_first_t_in_history = false;
                        self.x_matrix[(*i, current_column_fresh)] = T::one();
                        current_column_fresh += 1;
                        self.indices_history.insert(*i);
                    } else if current_column_historical < t {
                        self.x_matrix[(*i, current_column_historical)] = T::one();
                        current_column_historical += 1;
                    }
                }

                // > if ind(1:t) is contained in ind_hist, goto (6), end
                if all_first_t_in_history {
                    break 'optimization_loop;
                }

                // Iterate over the remaining indices
                'fill_x: for i in index_iterator {
                    if current_column_fresh >= t {
                        break 'fill_x;
                    }
                    if !self.indices_history.contains(i) {
                        self.x_matrix[(*i, current_column_fresh)] = T::one();
                        current_column_fresh += 1;
                        self.indices_history.insert(*i);
                    } else if current_column_historical < t {
                        self.x_matrix[(*i, current_column_historical)] = T::one();
                        current_column_historical += 1;
                    }
                }
            }
        }

        estimate
    }

    /// Estimate the 1-norm of matrix `a` using up to `itmax` iterations.
    pub fn normest1<S>(&mut self, a: &ArrayBase<S, Ix2>, itmax: usize) -> T::RealField
        where S: Data<Elem=T>,
    {
        self.calculate(a, itmax)
    }

    /// Estimate the 1-norm of a marix `a` to the power `m` up to `itmax` iterations.
    pub fn normest1_pow<S>(&mut self, a: &ArrayBase<S, Ix2>, m: usize, itmax: usize) -> T::RealField
        where S: Data<Elem=T>,
    {
        self.calculate(&(a, m), itmax)
    }

    /// Estimate the 1-norm of a product of matrices `a1 a2 ... an` up to `itmax` iterations.
    pub fn normest1_prod<S>(&mut self, aprod: &[&ArrayBase<S, Ix2>], itmax: usize) -> T::RealField
        where S: Data<Elem=T>,
    {
        self.calculate(aprod, itmax)
    }
}

/// Estimates the 1-norm of matrix `a`.
///
/// The parameter `t` is the number of vectors that have to fulfill some bound. See [Higham,
/// Tisseur] for more information. `itmax` is the maximum number of sweeps permitted.
///
/// **NOTE:** This function allocates on every call. If you want to repeatedly estimate the
/// 1-norm on matrices of the same size, construct a [`Normest1`] first, and call its methods.
///
/// [Higham, Tisseur]: http://eprints.ma.man.ac.uk/321/1/covered/MIMS_ep2006_145.pdf
/// [`Normest1`]: struct.Normest1.html
pub fn normest1<T: BlasScalar>(a_matrix: &Array2<T>, t: usize, itmax: usize) -> T::RealField
{
    // Assume the matrix is square and take the columns as n. If it's not square, the assertion in
    // normest.calculate will fail.
    let n = a_matrix.dim().1;
    let mut normest1 : Normest1<T> = Normest1::new(n, t);
    normest1.normest1(a_matrix, itmax)
}

/// Estimates the 1-norm of a matrix `a` to the power `m`, `a^m`.
///
/// The parameter `t` is the number of vectors that have to fulfill some bound. See [Higham,
/// Tisseur] for more information. `itmax` is the maximum number of sweeps permitted.
///
/// **NOTE:** This function allocates on every call. If you want to repeatedly estimate the
/// 1-norm on matrices of the same size, construct a [`Normest1`] first, and call its methods.
///
/// [Higham, Tisseur]: http://eprints.ma.man.ac.uk/321/1/covered/MIMS_ep2006_145.pdf
pub fn normest1_pow<T: BlasScalar>(a_matrix: &Array2<T>, m: usize, t: usize, itmax: usize) -> T::RealField
{
    // Assume the matrix is square and take the columns as n. If it's not square, the assertion in
    // normest.calculate will fail.
    let n = a_matrix.dim().1;
    let mut normest1 = Normest1::new(n, t);
    normest1.normest1_pow(a_matrix, m, itmax)
}

/// Estimates the 1-norm of a product of matrices `a1`, `a2`, ..., `an` passed in as a slice of
/// references.
///
/// The parameter `t` is the number of vectors that have to fulfill some bound. See [Higham,
/// Tisseur] for more information. `itmax` is the maximum number of sweeps permitted.
///
/// **NOTE:** This function allocates on every call. If you want to repeatedly estimate the
/// 1-norm on matrices of the same size, construct a [`Normest1`] first, and call its methods.
///
/// [Higham, Tisseur]: http://eprints.ma.man.ac.uk/321/1/covered/MIMS_ep2006_145.pdf
pub fn normest1_prod<T: BlasScalar>(a_matrices: &[&Array2<T>], t: usize, itmax: usize) -> T::RealField
{
    assert!(a_matrices.len() > 0);
    let n = a_matrices[0].dim().1;
    let mut normest1 = Normest1::new(n, t);
    normest1.normest1_prod(a_matrices, itmax)
}

/// Assigns the sign of matrix `a` to matrix `b`.
///
/// Panics if matrices `a` and `b` have different shape and strides, or if either underlying array is
/// non-contiguous. This is to make sure that the iteration order over the matrices is the same.
fn assign_signum_of_array<S1, S2, D, T: BlasScalar>(a: &ArrayBase<S1, D>, b: &mut ArrayBase<S2, D>)
    where S1: Data<Elem=T>,
          S2: DataMut<Elem=T>,
          D: Dimension
{
    assert_eq!(a.strides(), b.strides());
    let (a_slice, a_layout) = as_slice_with_layout(a).expect("Matrix `a` is not contiguous.");
    let (b_slice, b_layout) = as_slice_with_layout_mut(b).expect("Matrix `b` is not contiguous.");
    assert_eq!(a_layout, b_layout);

    signum_of_slice(a_slice, b_slice);
}

fn signum_of_slice<T: BlasScalar>(source: &[T], destination: &mut [T]) {
    for (s, d) in source.iter().zip(destination) {
        *d = s.signum();
    }
}

/// Calculate the onenorm of a vector (an `ArrayBase` with dimension `Ix1`).
fn vector_onenorm<S, T: BlasScalar>(a: &ArrayBase<S, Ix1>) -> T::RealField
    where S: Data<Elem=T>,
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
    T::asum(n_elements as i32, a_slice, stride as i32)
//    unsafe {
//        cblas::dasum(n_elements as i32, a_slice, stride as i32)
//    }
}

/// Calculate the maximum norm of a vector (an `ArrayBase` with dimension `Ix1`).
fn vector_maxnorm<S, T: BlasScalar>(a: &ArrayBase<S, Ix1>) -> T::RealField
    where S: Data<Elem=T>
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
    let idx = T::amax(n_elements as i32, a_slice, stride as i32) as usize;
//    let idx = unsafe {
//        cblas::idamax(
//            n_elements as i32,
//            a_slice,
//            stride as i32,
//        ) as usize
//    };
    T::abs(a[idx])
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
fn matrix_onenorm_with_index<S, T: BlasScalar>(a: &ArrayBase<S, Ix2>) -> (usize, T::RealField)
    where S: Data<Elem=T>,
{
    //todo:
    let mut max_norm : T::RealField = <T::RealField as Zero>::zero();
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
fn find_parallel_columns_in<S1, S2, T: BlasScalar> (
    a: &ArrayBase<S1, Ix2>,
    c: &mut ArrayBase<S2, Ix2>,
    column_is_parallel: &mut [bool]
)
    where S1: Data<Elem=T>,
          S2: DataMut<Elem=T>
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

        // NOTE: When calling the wrapped Fortran dsyrk subroutine with row major layout,
        // cblas::*syrk changes `'U'` to `'L'` (`Upper` to `Lower`), and `'O'` to `'N'` (`Ordinary`
        // to `None`). Different from `cblas::*gemm`, however, it does not automatically make sure
        // that the other arguments are changed to make sense in a routine expecting column major
        // order (in `cblas::*gemm`, this happens by flipping the matrices `a` and `b` as
        // arguments).
        //
        // So while `cblas::dsyrk` changes transposition and the position of where the results are
        // written to, it passes the other arguments on to the Fortran routine as is.
        //
        // For example, in case matrix `a` is a 4x2 matrix in column major order, and we want to
        // perform the operation `a^T a` on it (resulting in a symmetric 2x2 matrix), we would pass
        // TRANS='T', N=2 (order of c), K=4 (number of rows because of 'T'), LDA=4 (max(1,k)
        // because of 'T'), LDC=2.
        //
        // But if `a` is in row major order and we want to perform the same operation, we pass
        // TRANS='T' (gets translated to 'N'), N=2, K=2 (number of columns, because we 'T' -> 'N'),
        // LDA=2 (max(1,n) because of 'N'), LDC=2.
        //
        // In other words, because of row major order, the Fortran routine actually sees our 4x2
        // matrix as a 2x4 matrix, and if we want to calculate `a^T a`, `cblas::dsyrk` makes sure
        // `'N'` is passed.
        let (k, lda) = match layout {
            cblas::Layout::ColumnMajor => (n_cols, n_rows),
            cblas::Layout::RowMajor => (n_rows, n_cols),
        };
        T::syrk(layout,
                cblas::Part::Upper,
            //All entries are assumed to be real, so herk w/ conj trans is not necessary
                cblas::Transpose::Ordinary,
                n_cols as i32,
                k as i32,
                T::from_subset(&1.0),
                a_slice,
                lda as i32,
                T::zero(),
                c_slice,
                n_cols as i32,);
    }

    // c is upper triangular and contains all pair-wise vector products:
    //
    // x x x x x
    // . x x x x
    // . . x x x
    // . . . x x
    // . . . . x

    // Don't check more rows than we have columns 
    'rows: for (i, row) in c.genrows().into_iter().enumerate().take(n_cols) {
        // Skip if the column is already found to be parallel or if we are checking
        // the last column
        if column_is_parallel[i] || i >= n_cols - 1 { continue 'rows; }
        for (j, element) in row.slice(s![i+1..]).iter().enumerate() {
            // Check if the vectors are parallel or anti-parallel
            if T::abs(*element).to_subset().unwrap() == n_rows as f64 {
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
fn find_parallel_columns_between<S1, S2, S3, T: BlasScalar> (
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    c: &mut ArrayBase<S3, Ix2>,
    column_is_parallel: &mut [bool],
)
    where S1: Data<Elem=T>,
          S2: Data<Elem=T>,
          S3: DataMut<Elem=T>
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

        T::gemm(layout, cblas::Transpose::Conjugate, cblas::Transpose::None,
                n_cols as i32, n_cols as i32, n_rows as i32,
                T::one(), a_slice, n_cols as i32, b_slice, n_cols as i32,
                T::zero(), c_slice, n_cols as i32);
//        unsafe {
//            cblas::dgemm(
//                layout,
//                cblas::Transpose::Ordinary,
//                cblas::Transpose::None,
//                n_cols as i32,
//                n_cols as i32,
//                n_rows as i32,
//                1.0,
//                a_slice,
//                n_cols as i32,
//                b_slice,
//                n_cols as i32,
//                0.0,
//                c_slice,
//                n_cols as i32,
//            );
//        }
    }

    // We are iterating over the rows because it's more memory efficient (for row-major array).  In
    // terms of logic there is no difference: we simply check if the current column of a (that's
    // the outer loop) is parallel to any column of b (inner loop). By iterating via columns we would check if
    // any column of a is parallel to the, in that case, current column of b.
    // TODO:  Implement for column major arrays.
    'rows: for (i, row) in c.genrows().into_iter().enumerate().take(n_cols) {
        // Skip if the column is already found to be parallel the last column.
        if column_is_parallel[i] { continue 'rows; }
        for element in row {
            if T::abs(*element).to_subset().unwrap() == n_rows as f64 {
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
fn are_all_columns_parallel_between<S1, S2, T: BlasScalar> (
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S1, Ix2>,
    c: &mut ArrayBase<S2, Ix2>,
) -> bool
    where S1: Data<Elem=T>,
          S2: DataMut<Elem=T>
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
        T::gemm(layout, cblas::Transpose::Conjugate, cblas::Transpose::None,
                n_cols as i32, n_cols as i32, n_rows as i32,
                T::one(), a_slice, n_cols as i32, b_slice, n_cols as i32,
                T::zero(), c_slice, n_rows as i32,);
//        unsafe {
//            cblas::dgemm(
//                layout,
//                cblas::Transpose::Ordinary,
//                cblas::Transpose::None,
//                n_cols as i32,
//                n_cols as i32,
//                n_rows as i32,
//                1.0,
//                a_slice,
//                n_cols as i32,
//                b_slice,
//                n_cols as i32,
//                0.0,
//                c_slice,
//                n_rows as i32,
//            );
//        }
    }

    // We are iterating over the rows because it's more memory efficient (for row-major array).  In
    // terms of logic there is no difference: we simply check if a specific column of a is parallel
    // to any column of b. By iterating via columns we would check if any column of a is parallel
    // to a specific column of b.
    'rows: for row in c.genrows() {
        for element in row {
            // If a parallel column was found, cut to the next one.
            if T::abs(*element).to_subset().unwrap() == n_rows as f64 { continue 'rows; }
        }
        // This return statement should only be reached if not a single column parallel to the
        // current one was found.
        return false;
    }
    true
}

/// Find parallel columns in matrix `a` and columns in `a` that are parallel to any columns in
/// matrix `b`, and replace those with random vectors. Returns `true` if resampling has taken place.
fn resample_parallel_columns<S1, S2, S3, R, T: BlasScalar>(
    a: &mut ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    c: &mut ArrayBase<S3, Ix2>,
    column_is_parallel: &mut [bool],
    rng: &mut R,
    sample: &[T],
) -> bool
    where S1: DataMut<Elem=T>,
          S2: Data<Elem=T>,
          S3: DataMut<Elem=T>,
          R: Rng
{
    column_is_parallel.iter_mut().for_each(|x| {*x = false;});
    find_parallel_columns_in(a, c, column_is_parallel);
    find_parallel_columns_between(a, b, c, column_is_parallel);
    let mut has_resampled = false;
    for (i, is_parallel) in column_is_parallel.into_iter().enumerate() {
        if *is_parallel {
            resample_column(a, i, rng, sample);
            has_resampled = true;
        }
    }
    has_resampled
}

/// Resamples column `i` of matrix `a` with elements drawn from `sample` using `rng`.
///
/// Panics if `i` exceeds the number of columns in `a`.
fn resample_column<R, S, T: BlasScalar>(
    a: &mut ArrayBase<S, Ix2>,
    i: usize, rng:
    &mut R, sample: &[T]
)
    where S: DataMut<Elem=T>,
          R: Rng
{
    assert!(i < a.dim().1, "Trying to resample column with index exceeding matrix dimensions");
    assert!(sample.len() > 0);
    a.column_mut(i).mapv_inplace(|_| sample[rng.gen_range(0, sample.len())]);
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
    //
    // if let Some(slice) = a.as_slice_mut() {
    //     Some((slice, cblas::Layout::RowMajor))
    // } else if let Some(slice) = a.as_slice_memory_order_mut() {
    //     Some((slice, cblas::Layout::ColumnMajor))
    // } else {
    //     None
    // }
}

#[cfg(test)]
mod tests {
    extern crate openblas_src;

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

    #[test]
    fn equality_between_methods() {
        let t = 2;
        let n = 100;
        let itmax = 5;

        let mut rng = Xoshiro256Plus::seed_from_u64(1234);
        let distribution = StandardNormal;

        let mut a_matrix = Array::random_using((n, n), distribution, &mut rng);
        a_matrix.mapv_inplace(|x| 1.0/x);

        let unity = Array::eye(n);

        let estimate_onlymatrix = crate::normest1(&a_matrix, t, itmax);
        let estimate_matrixpow = crate::normest1_pow(&a_matrix, 1, t, itmax);
        let estimate_matrixprod = crate::normest1_prod(&[&a_matrix, &unity], t, itmax);

        assert_eq!(estimate_onlymatrix, estimate_matrixpow);
        assert_eq!(estimate_onlymatrix, estimate_matrixprod);
    }

    #[test]
    fn pow_2_is_prod_2() {
        let t = 2;
        let n = 100;
        let itmax = 5;

        let mut rng = Xoshiro256Plus::seed_from_u64(1234);
        let distribution = StandardNormal;

        let mut a_matrix = Array::random_using((n, n), distribution, &mut rng);
        a_matrix.mapv_inplace(|x| 1.0/x);

        let estimate_matrixpow = crate::normest1_pow(&a_matrix, 2, t, itmax);
        let estimate_matrixprod = crate::normest1_prod(&[&a_matrix, &a_matrix], t, itmax);

        assert_eq!(estimate_matrixpow, estimate_matrixprod);
    }

    #[test]
    /// This performs tests inspired by Table 3 of [Higham and Tisseur].
    ///
    /// NOTE: Due to (most likely) floating point precision), the ratio `calculated/expected` (that
    /// is, the ratio of the estimated condition number to the explicitly calculated one) can
    /// exceed 1.0. However, when running the tests I have observed at most a ratio exceeding 1.0
    /// by 3 bits in the significand/mantissa. In other words, the estimated condition number appears to be
    /// within 4 ULPS of the calculated/expected one.
    ///
    /// One can probably explain this with different ordering of summation/addition/multiplication.
    ///
    /// During tests run performed by the author(s) of this library, running the tets below with
    /// `nsamples = 5000` happened to always let the test pass.
    fn table_3_t_2() {
        let t = 2;
        let n = 100;
        let itmax = 5;
        let nsamples = 5000;

        let mut calculated = Vec::with_capacity(nsamples);
        let mut expected = Vec::with_capacity(nsamples);

        let mut rng = Xoshiro256Plus::seed_from_u64(1234);
        let distribution = StandardNormal;

        for _ in 0..nsamples {
            let mut a_matrix = Array::random_using((n, n), distribution, &mut rng);
            a_matrix.mapv_inplace(|x| 1.0/x);
            let estimate = crate::normest1(&a_matrix, t, itmax);
            calculated.push(estimate);
            expected.push({
                let (a_slice, a_layout) = crate::as_slice_with_layout(&a_matrix).expect("a matrix not contiguous");
                let a_layout = match a_layout {
                    cblas::Layout::ColumnMajor => lapacke::Layout::ColumnMajor,
                    cblas::Layout::RowMajor => lapacke::Layout::RowMajor,
                };
                unsafe {
                    lapacke::dlange(
                    a_layout,
                    b'1',
                    n as i32,
                    n as i32,
                    a_slice,
                    n as i32,
                )}
            });
        }

        let calculated = Array1::from_vec(calculated);
        let expected = Array1::from_vec(expected);

        let mut underestimation_ratio = unsafe { Array1::<f64>::uninitialized(nsamples) };
        Zip::from(&calculated)
            .and(&expected)
            .and(&mut underestimation_ratio)
            .apply(|c, e, u| {
                *u = *c / *e;
        });

        let underestimation_mean = underestimation_ratio.mean_axis(Axis(0)).into_scalar();
        assert!(0.99 < underestimation_mean);
        assert!(underestimation_mean < 1.0);
    }
}
