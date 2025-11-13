{pkgs, lib, config, ...}:
let
  py = pkgs.python312Packages;
  cuda = pkgs.cudaPackages;
  lightgbm-gpu = py.lightgbm.overridePythonAttrs (old: {
    cmakeFlags = (old.cmakeFlags or []) ++ [
      "-DUSE_CUDA=ON"
      "-DCMAKE_CUDA_STANDARD=17"
      "-DCMAKE_CUDA_STANDARD_REQUIRED=ON"
    ];
    preConfigure = (old.preConfigure or "") + ''
      export CMAKE_CUDA_FLAGS="--expt-relaxed-constexpr"
      export CXXFLAGS="${(old.CXXFLAGS or "")} -std=gnu++17"
    '';
    nativeBuildInputs = (
      old.nativeBuildInputs or []
    ) ++ (with pkgs; [
      cmake
      ninja
    ]);
    buildInputs = (
      old.buildInputs or []
    ) ++ (with cuda; [
      cudatoolkit
    ]);
  });
  torchdr-gpu = py.buildPythonPackage rec {
    pname = "torchdr";
    version = "0.3";
    pyproject = true;
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "sha256-623xLK2bf7Vr8BpzemcMSk5n0yD2VjBZ4vjsy6OQTX0=";
    };
    build-system = with py; [
      setuptools-scm
      setuptools
      wheel
    ];
    propagatedBuildInputs = with py; [
      scikit-learn
      torch-bin
      numpy
    ];
    doCheck = false;
  };
  keopscore-gpu = py.buildPythonPackage rec {
    pname = "keopscore";
    version = "2.3";
    format = "setuptools";
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "sha256-6KcAwi7+5Zp3I+11FsLYirFDoRKQ3SOwfre7ocnRb+8=";
    };
    nativeBuildInputs = with py; [
      setuptools
      wheel
    ];
    doCheck = false;
  };
  pykeops-gpu = py.buildPythonPackage rec {
    pname = "pykeops";
    version = "2.3";
    format = "setuptools";
    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "sha256-RY+neYXv0hYb6ccZ6N1B5JVSbNc10yYq7AmQJQX1vbo=";
    };
    nativeBuildInputs = with py; [
      setuptools
      pybind11
      wheel
    ];
    propagatedBuildInputs = with py; [
      keopscore-gpu
      numpy
    ];
    doCheck = false;
  };
in {
  devShells.default = pkgs.mkShell {
    name = "ctc-shell";
    packages = with py; [
      pkgs.nixgl.auto.nixGLNvidia
      sentence-transformers
      pkgs.datafusion-cli
      cuda.cudatoolkit
      keopscore-gpu
      lightgbm-gpu
      scikit-learn
      torchdr-gpu
      pykeops-gpu
      setuptools
      matplotlib
      torch-bin
      seaborn
      python
      flake8
      pyyaml
      polars
      pandas
      numpy
      typer
      wheel
      shap
      pip
    ];
    CUDA_PATH = cuda.cudatoolkit;
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      cuda.cudatoolkit
    ];
    shellHook = ''
      NIXGL_BIN="$(compgen -c | grep -E '^nixGLNvidia(-[0-9.]+)?$' | head -n1)"

      alias python="$NIXGL_BIN python"
      alias python3="$NIXGL_BIN python3"
      alias torchrun="$NIXGL_BIN torchrun"
      alias CTC="$NIXGL_BIN CTC"
    '';
  };
}