use std::cmp::PartialEq;
use std::ops::{Index, IndexMut};
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub capacity: usize,
    pub stride: usize,
    pub storage: Vec<T>,
}

impl<T> Matrix<T>
where
    T: PartialEq + Copy,
{
    pub fn new(rows: usize, columns: usize, v: T) -> Matrix<T> {
        let capacity = rows * columns;
        let stride = columns;
        let storage: Vec<T> = vec![v; capacity as usize];
        Matrix {
            capacity,
            stride,
            storage,
        }
    }

    pub fn rows(&self) -> usize {
        if self.stride == 0 {
            0
        } else {
            self.capacity / self.stride
        }
    }

    pub fn cols(&self) -> usize {
        self.stride
    }

    pub fn resize(&mut self, rows: usize, cols: usize, padding: T) {
        self.capacity = rows * cols;
        self.stride = cols;
        self.storage.resize(self.capacity as usize, padding);
    }

    pub fn data(&self) -> &[T] {
        &self.storage
    }

    pub fn mut_data(&mut self) -> &mut [T] {
        &mut self.storage
    }
}

impl<T: PartialEq + Copy> PartialEq for Matrix<T> {
    fn eq(&self, other: &Matrix<T>) -> bool {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            return false;
        }
        self.storage == other.storage
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index<'a>(&'a self, (row, col): (usize, usize)) -> &'a T {
        &self.storage[row * self.stride as usize + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut<'a>(&'a mut self, (row, col): (usize, usize)) -> &'a mut T {
        &mut self.storage[row * self.stride as usize + col]
    }
}
