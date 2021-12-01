use std::{env, path::*};
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    if !Path::new(&format!("{}/lib/libmerged.a", &out_dir)).exists() {
        Command::new("wget")
                .arg("https://anaconda.org/intel/mkl-static/2020.0/download/linux-64/mkl-static-2020.0-intel_166.tar.bz2")
//                .arg("https://anaconda.org/intel/mkl-static/2020.1/download/linux-64/mkl-static-2020.1-intel_217.tar.bz2")
                .args(&["-P", &out_dir]) 
                .status().unwrap();

        Command::new("tar")
                .arg("-xvf")
//                .arg(&format!("{}/mkl-static-2020.1-intel_217.tar.bz2", out_dir))
                .arg(&format!("{}/mkl-static-2020.0-intel_166.tar.bz2", out_dir))
                .args(&["-C", &out_dir])
                .status().unwrap();

        Command::new("ar")
                .arg("-rcT")
                .arg(&format!("{}/lib/libmerged.a", out_dir))
                .arg(&format!("{}/lib/libmkl_sequential.a", out_dir))
                .arg(&format!("{}/lib/libmkl_core.a", out_dir))
                .arg(&format!("{}/lib/libmkl_intel_ilp64.a", out_dir))
                .status().unwrap();
    }

    // TODO: make this crossplatform?

    println!("cargo:rustc-link-search={}/lib", out_dir);
    println!("cargo:rustc-link-lib=dylib=merged");
}
