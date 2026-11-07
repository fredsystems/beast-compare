{
  lib,
  buildPythonApplication,
  setuptools,
  pyyaml,
  zstandard,
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
    pyyaml
    zstandard
  ];

  nativeCheckInputs = [ pytestCheckHook ];

  pythonImportsCheck = [ "beast_compare" ];

  meta = {
    description = "Record and compare ADS-B receivers from their Beast outputs";
    mainProgram = "beast-compare";
  };
}
