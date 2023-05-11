use std::env;
use std::path::Path;
fn main() {
    let local_linklib_path = Path::new("/usr/local/lib");
    // build silent vole
    let path = env::current_dir().unwrap();
    let mut rs_vole_cpp = path.clone();
    rs_vole_cpp.push("vole_binding/rvole.cpp");
    let mut rs_aes_cpp = path.clone();
    rs_aes_cpp.push("crypto_binding/raes.cpp");
    let mut paxos_hash_cpp = path.clone();
    paxos_hash_cpp.push("crypto_binding/paxos_hash.cpp");
    let mut prng_cpp = path.clone();
    prng_cpp.push("crypto_binding/prng.cpp");
    cc::Build::new()
        .file(&rs_vole_cpp)
        .flag("-Wno-unknown-pragmas")
        .flag("-Wno-sign-compare")
        .flag("-Wno-unused-parameter")
        .flag("-msse4.1")
        .flag("-mpclmul")
        .flag("-maes")
        .flag("-std=c++17")
        .opt_level(3)
        .pic(true)
        .cpp(true)
        .compile("librvole.a");
    cc::Build::new()
        .file(&rs_aes_cpp)
        .file(&paxos_hash_cpp)
        .file(&prng_cpp)
        .flag("-Wno-unknown-pragmas")
        .flag("-Wno-sign-compare")
        .flag("-Wno-unused-parameter")
        .flag("-msse4.1")
        .flag("-mpclmul")
        .flag("-maes")
        .flag("-mavx")
        .flag("-mavx2")
        .flag("-std=c++17")
        .opt_level(3)
        .pic(true)
        .cpp(true)
        .compile("librcrypto.a");
    println!("cargo:rustc-link-lib=static=cryptoTools");
    println!("cargo:rustc-link-lib=static=libOTe");
    println!("cargo:rustc-link-lib=static=coproto");
    println!("cargo:rustc-link-lib=static=macoro");
    println!("cargo:rustc-link-lib=static=sodium");
    println!("cargo:rustc-link-lib=static=bitpolymul");
    println!("cargo:rerun-if-changed={}", rs_vole_cpp.to_str().unwrap());
    println!("cargo:rerun-if-changed={}", rs_aes_cpp.to_str().unwrap());
    println!(
        "cargo:rerun-if-changed={}",
        paxos_hash_cpp.to_str().unwrap()
    );
    println!("cargo:rerun-if-changed={}", prng_cpp.to_str().unwrap());
    println!(
        "cargo:rerun-if-changed={}",
        paxos_hash_cpp.to_str().unwrap()
    );
    println!(
        "cargo:rustc-link-search={}",
        local_linklib_path.to_str().unwrap()
    );
}
