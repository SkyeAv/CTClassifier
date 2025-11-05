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
in {
  devShells.default = pkgs.mkShell {
    name = "ctc-shell";
    packages = with py; [
      pkgs.nixgl.auto.nixGLNvidia
      sentence-transformers
      pkgs.datafusion-cli
      cuda.cudatoolkit
      lightgbm-gpu
      scikit-learn
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