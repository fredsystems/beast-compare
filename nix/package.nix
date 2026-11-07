{
  lib,
  buildPythonApplication,
  setuptools,
  numpy,
  polars,
  pyyaml,
  zstandard,
  pymodes,
  pytestCheckHook,
}:

buildPythonApplication {
  pname = "beast-compare";
  version = "0.1.0";
  pyproject = true;

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../pyproject.toml
      ../src
      ../tests
    ];
  };

  build-system = [ setuptools ];

  dependencies = [
    numpy
    polars
    pyyaml
    zstandard
  ];

  # pyModeS is the reference the vectorised decoders are tested against.
  nativeCheckInputs = [
    pymodes
    pytestCheckHook
  ];

  pythonImportsCheck = [ "beast_compare" ];

  meta = {
    description = "Record and compare ADS-B receivers from their Beast outputs";
    mainProgram = "beast-compare";
  };
}
