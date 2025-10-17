// File containing struct definitions for a memory representation of MANIFolds
// Will do weird ass things if

use std::{
    array::TryFromSliceError,
    collections::HashMap,
    ops::{Add, Sub},
};

// Generic struct for N-D points based on point_nd crate
#[derive(Clone, Debug, PartialEq)]
pub struct Point<const N: usize>([f64; N]);

impl<const N: usize> Point<N> {
    pub fn from_slice(slice: &[f64]) -> Self {
        let arr: [f64; N] = slice.try_into().unwrap();
        Point::from(arr)
    }

    pub fn fill(value: f64) -> Self {
        Point::from([value; N])
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    pub fn into_arr(self) -> [f64; N] {
        self.0
    }

    pub fn apply<U>(self, modifier: fn(f64) -> f64) -> Point<N> {
        let mut arr_v = Vec::<f64>::with_capacity(N);
        let this = Vec::from(self.into_arr());

        this.into_iter().for_each(|i| {
            arr_v.push(modifier(i));
        });

        Point::from_slice(arr_v.as_slice())
    }

    pub fn apply_axis(self, axis: &[usize], modifier: fn(f64) -> f64) -> Point<N> {
        let mut arr_v = Vec::<f64>::with_capacity(N);
        let this = Vec::from(self.into_arr());

        this.into_iter().enumerate().for_each(|(n, i)| {
            if axis.contains(&n) {
                arr_v.push(modifier(i))
            } else {
                arr_v.push(i)
            };
        });

        Point::from_slice(arr_v.as_slice())
    }

    pub fn apply_with_vals<U>(self, values: [U; N], modifier: fn(f64, U) -> f64) -> Point<N> {
        let mut arr_v = Vec::<f64>::with_capacity(N);
        let vals = Vec::from(values);
        let this = Vec::from(self.into_arr());

        this.into_iter().zip(vals).for_each(|(i, v)| {
            arr_v.push(modifier(i, v));
        });

        Point::from_slice(arr_v.as_slice())
    }

    pub fn apply_point(self, other: Point<N>, modifier: fn(f64, f64) -> f64) -> Point<N> {
        self.apply_with_vals(other.into_arr(), modifier)
    }

    pub fn dot(self, other: Point<N>) -> f64 {
        let this = Vec::from(self.into_arr());
        let other = Vec::from(other.into_arr());
        let mut res: f64 = 0.0;

        this.into_iter().zip(other).for_each(|(i, v)| {
            res += i * v;
        });

        res
    }
    pub fn extend<const L: usize, const M: usize>(self, values: [f64; L]) -> Point<M> {
        let mut arr_v = Vec::<f64>::with_capacity(M);
        let this = Vec::from(self.into_arr());
        let vals = Vec::from(values);

        this.into_iter().for_each(|i| arr_v.push(i));
        vals.into_iter().for_each(|i| arr_v.push(i));

        Point::from_slice(arr_v.as_slice())
    }

    pub fn retain<const M: usize>(self, dims: &[usize; M]) -> Point<M> {
        if M > N {
            panic!("Cannot reduce dimension to a larger dimension. 💠")
        }

        let mut arr_v = Vec::<f64>::with_capacity(M);
        let this = Vec::from(self.into_arr());

        this.into_iter().enumerate().for_each(|(n, i)| {
            if dims.contains(&n) {
                arr_v.push(i);
            }
        });

        Point::from_slice(arr_v.as_slice())
    }
}

impl<const N: usize> From<[f64; N]> for Point<N> {
    fn from(array: [f64; N]) -> Self {
        Point(array)
    }
}

impl<const N: usize> TryFrom<&[f64]> for Point<N> {
    type Error = TryFromSliceError;
    fn try_from(slice: &[f64]) -> Result<Self, Self::Error> {
        let res: Result<[f64; N], _> = slice.try_into();
        match res {
            Ok(arr) => Ok(Point(arr)),
            Err(err) => Err(err),
        }
    }
}

impl<const N: usize> From<Point<N>> for [f64; N] {
    fn from(point: Point<N>) -> Self {
        point.into_arr()
    }
}

//impl<const N: usize> From<&[f64]> for Point<N> {
//    fn from(value: &[f64]) -> Self {
//        let tmp = Point::from_slice(value);
//        if tmp.dim() < N {
//            const L: usize = N;
//            let mut zeros = Vec::<f64>::with_capacity(L);
//
//            for _ in 0..(N - tmp.dim()) {
//                zeros.push(0.);
//            }
//
//            tmp.extend::<L, N>(zeros.try_into().unwrap())
//        } else if tmp.dim() > N {
//            let mut zeros: [usize; N];
//
//            for i in 0..N {
//                zeros[i] = i
//            }
//
//            tmp.retain(&zeros)
//        } else {
//            tmp
//        }
//    }
//}

impl<const N: usize> Add for Point<N> {
    type Output = Point<N>;

    fn add(self, rhs: Self) -> Self::Output {
        self.apply_point(rhs, |a, b| a + b)
    }
}

impl<const N: usize> Sub for Point<N> {
    type Output = Point<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.apply_point(rhs, |a, b| a - b)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Manifold<const N: usize> {
    vertices: HashMap<usize, Point<N>>,
    triangles: Vec<(usize, usize, usize)>,
}

impl<const N: usize> Manifold<N> {}
