#![feature(static_nobundle)]

#[macro_use(s, array)]
extern crate ndarray;
extern crate kth;
extern crate libc;

use libc::{c_float, c_int, c_longlong, c_void};
use ndarray::{Array, Axis, Ix1, Ix2, ArrayBase, Data, ArrayView, ArrayViewMut};
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::ptr;
use ndarray::linalg::general_mat_mul;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray2, PyArray1};

pub type SgemmJitKernelT =
    Option<unsafe extern "C" fn(arg1: *mut c_void, arg2: *mut f32, arg3: *mut f32, arg4: *mut f32)>;

extern "C" {
    fn mkl_cblas_jit_create_sgemm(
        JITTER: *mut *mut c_void,
        layout: u32,
        transa: u32,
        transb: u32,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        lda: usize,
        ldb: usize,
        beta: f32,
        ldc: usize,
    ) -> u32;
    fn mkl_jit_get_sgemm_ptr(JITTER: *const c_void) -> SgemmJitKernelT;
    fn vmsExp(n: c_int, a: *const c_float, y: *mut c_float, mode: c_longlong);
}

static mut JITTER48_5: *mut c_void = ptr::null_mut();
static mut SGEMM48_5: SgemmJitKernelT = None;

const ALIGN: usize = 64;

pub fn align2d(input: Array<f32, Ix2>) -> Array<f32, Ix2> {
    unsafe {
        let optr = std::alloc::alloc(std::alloc::Layout::from_size_align(input.len() * 4, ALIGN).unwrap()) as *mut f32;
        let ov = Vec::from_raw_parts(optr, input.len(), input.len());
        let mut res = Array::from_shape_vec_unchecked(input.shape(), ov);
        res.assign(&input);
        res.into_dimensionality::<Ix2>().unwrap()
    } 
}

pub fn align1d(input: Array<f32, Ix1>) -> Array<f32, Ix1> {
    unsafe {
        let optr = std::alloc::alloc(std::alloc::Layout::from_size_align(input.len() * 4, ALIGN).unwrap()) as *mut f32;
        let ov = Vec::from_raw_parts(optr, input.len(), input.len());
        let mut res = Array::from_shape_vec_unchecked(input.shape(), ov);
        res.assign(&input);
        res.into_dimensionality::<Ix1>().unwrap()
    } 
}


fn initialize_jit() {
    unsafe {
        JITTER48_5 = ptr::null_mut();
        // TODO: check
        let status = mkl_cblas_jit_create_sgemm(
            &mut JITTER48_5,
            101,
            111,
            111,
            1,
            5,
            48,
            1.0,
            48,
            5,
            0.0,
            5,
        );

        SGEMM48_5 = mkl_jit_get_sgemm_ptr(JITTER48_5);
    }
}

const HIDDEN: usize = 48;

#[pyclass]
struct DecoderTab {
  tab0: Array<f32, Ix2>,
  tab1: Array<f32, Ix2>,
  tab2: Array<f32, Ix2>,
  tab3: Array<f32, Ix2>,
  tab4: Array<f32, Ix2>,
  tab5: Array<f32, Ix2>,
  tab: Array<f32, Ix2>,
}

impl DecoderTab {
  fn inner_step(&mut self, outbases: usize, base_state:isize, row: &mut ArrayViewMut<f32, Ix1>,
                out: &mut Array<f32, Ix1>) {
        unsafe {
            let tabptr = if outbases > 5 {
              self.tab.as_mut_ptr().offset((48*5+16)*base_state)
            } else if outbases == 0 {
              self.tab0.as_mut_ptr().offset((48*5+16)*base_state)
            } else if outbases == 1 {
              self.tab1.as_mut_ptr().offset((48*5+16)*base_state)
            } else if outbases == 2 {
              self.tab2.as_mut_ptr().offset((48*5+16)*base_state)
            } else if outbases == 3 {
              self.tab3.as_mut_ptr().offset((48*5+16)*base_state)
            } else if outbases == 4 {
              self.tab4.as_mut_ptr().offset((48*5+16)*base_state)
            } else {
              self.tab5.as_mut_ptr().offset((48*5+16)*base_state)
            };
            SGEMM48_5.unwrap()(JITTER48_5, row.as_mut_ptr(), tabptr,
                               out.as_mut_ptr());
            let outptr = out.as_mut_ptr();
            let biasptr = tabptr.offset((48*5)); 
            for i in 0..5 {
              *outptr.offset(i) += *biasptr.offset(i);
            }
        }
  }
}


#[pymethods]
impl DecoderTab {
  #[new]
  fn new(
         tab0: &PyArray2<f32>,
         tab1: &PyArray2<f32>,
         tab2: &PyArray2<f32>,
         tab3: &PyArray2<f32>,
         tab4: &PyArray2<f32>,
         tab5: &PyArray2<f32>,
         tab: &PyArray2<f32>) -> Self {

    unsafe {
        if JITTER48_5 == ptr::null_mut() {
            initialize_jit();
        }
    }
    DecoderTab { 
        tab0: align2d(tab0.to_owned_array()),
        tab1: align2d(tab1.to_owned_array()),
        tab2: align2d(tab2.to_owned_array()),
        tab3: align2d(tab3.to_owned_array()),
        tab4: align2d(tab4.to_owned_array()),
        tab5: align2d(tab5.to_owned_array()),
        tab: align2d(tab.to_owned_array()),
    }  
  }

  fn do_not_call_me(&mut self) {
      let a1 = Array::from_elem((100, 100), 0.0);
      let a2 = Array::from_elem((100, 100), 0.0);
      let mut a3 = Array::from_elem((100, 100), 0.0);
      general_mat_mul(1.0, &a1, &a2, 0.0, &mut a3); 
  }


  fn decode(&mut self, data: &PyArray2<f32>) -> String {
    let alphabet: Vec<char> = "NACGT".chars().collect();
    let mut out = align1d(Array::from_elem(5, 0.0));
    
    let mut outstr = String::new();
    let mut outbases = 0;
    let mut base_state = 0;

    for mut row in unsafe { data.as_array_mut() }.outer_iter_mut() {
        self.inner_step(outbases, base_state, &mut row, &mut out);
        let mut top = 0;
        unsafe {
          let outptr = out.as_ptr();
          for i in 1..5 {
            if *outptr.offset(top) < *outptr.offset(i) {
              top = i;
            }
          }
        }
        outstr.push(alphabet[top as usize]);
        if top != 0 {
          outbases += 1;
          base_state *= 4;
          base_state += top - 1;
          base_state %= 256*4*4;
        }
    }
    
    outstr
  }


    fn beam_search(&mut self, data: &PyArray2<f32>, beam_size: usize, beam_cut_threshold: f32) -> (String, String) {
        let alphabet: Vec<char> = "NACGT".chars().collect();
        // (base, what)
        let mut beam_prevs = vec![(0, 0)];
        let mut beam_max_p = vec![(0.0f32)];
        let mut beam_forward: Vec<[i32; 4]> = vec![[-1, -1, -1, -1]];
        
        let mut beam_state: Vec<(usize, isize)> = vec![(6, 0)];

        let mut cur_probs = vec![(0i32, 0.0, 1.0)];
        let mut new_probs = Vec::new();
        let mut pr = align1d(Array::from_elem(5, 0.0));
        
        for mut row in unsafe { data.as_array_mut() }.outer_iter_mut() {
//            println!("tick");
            new_probs.clear();

            for &(beam, base_prob, n_prob) in &cur_probs {
//                println!("bs {:?}", beam_state[beam as usize]);
                self.inner_step(beam_state[beam as usize].0, beam_state[beam as usize].1, &mut row, &mut pr);
                    
                unsafe {
                    let ptr = pr.as_mut_ptr();
                    let mut max = *ptr.offset(0);
                    for i in 0..5 {
                        max = max.max(*ptr.offset(i));
                    }
                    for i in 0..5 {
                        *ptr.offset(i) = *ptr.offset(i) - max;
                    }
                    vmsExp(5, ptr, ptr, 259);
                    let mut sum = 0.0;
                    for i in 0..5 {
                        sum += *ptr.offset(i)
                    }
                    for i in 0..5 {
                        *ptr.offset(i) = *ptr.offset(i) / sum;
                    }
                }

                // add N to beam
                if pr[0] > beam_cut_threshold {
                    new_probs.push((beam, 0.0, (n_prob + base_prob) * pr[0]));
                }

                for b in 1..5 {
                    if pr[b] < beam_cut_threshold {
                        continue
                    }
                    {
                        let mut new_beam = beam_forward[beam as usize][b-1];
                        if new_beam == -1 {
                            new_beam = beam_prevs.len() as i32;
                            beam_prevs.push((b, beam));
                            beam_max_p.push(pr[b]);
                            beam_forward[beam as usize][b-1] = new_beam;
                            beam_forward.push([-1, -1, -1, -1]);
                            beam_state.push((beam_state[beam as usize].0 + 1,
                                             (beam_state[beam as usize].1 * 4 + (b-1) as isize) %
                                             (256*4*4)));
                        }

                        new_probs.push((new_beam, (base_prob + n_prob) * pr[b], 0.0));
                        beam_max_p[new_beam as usize] = beam_max_p[new_beam as usize].max(pr[b]);
                    }
                }
            }
            std::mem::swap(&mut cur_probs, &mut new_probs);

            cur_probs.sort_by_key(|x| x.0);
            let mut last_key: i32 = -1;
            let mut last_key_pos = 0;
            for i in 0..cur_probs.len() {
                if cur_probs[i].0 == last_key {
                    cur_probs[last_key_pos].1 = cur_probs[last_key_pos].1 + cur_probs[i].1;
                    cur_probs[last_key_pos].2 = cur_probs[last_key_pos].2 +cur_probs[i].2;
                    cur_probs[i].0 = -1;
                } else {
                    last_key_pos = i;
                    last_key = cur_probs[i].0;
                }
            }

            cur_probs.retain(|x| x.0 != -1);
            cur_probs.sort_by(|a, b| (b.1 + b.2).partial_cmp(&(a.1 + a.2)).unwrap());
            cur_probs.truncate(beam_size);
            let top = cur_probs[0].1 + cur_probs[0].2;
            for mut x in &mut cur_probs {
                x.1 /= top;
                x.2 /= top;
            }
//            println!("tick done");
        }
//        println!("search done");

        let mut out = String::new();
        let mut out_p = String::new();
        let mut beam = cur_probs[0].0;
//        println!("beam {}", beam);
        while beam != 0 {
            out.push(alphabet[beam_prevs[beam as usize].0]);
            out_p.push(prob_to_str(beam_max_p[beam as usize]));
            beam = beam_prevs[beam as usize].1;
        }
        (out.chars().rev().collect(), out_p.chars().rev().collect())
    }
}

fn prob_to_str(x: f32) -> char {
    let q = (-(1.0-x).log10()*10.0) as u32 + 33;
    std::char::from_u32(q).unwrap()
}

#[pymodule]
fn decoder(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DecoderTab>()?;

    Ok(())
}
