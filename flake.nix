{
  description = "Record and compare ADS-B receivers from their Beast outputs";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";

    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

    precommit-base = {
      url = "github:fredsystems/pre-commit-checks";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      precommit-base,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;

        beast-compare = python.pkgs.callPackage ./nix/package.nix { };

        precommitCheck = precommit-base.lib.mkCheck {
          inherit system;
          src = ./.;
          extraExcludes = [ "^tests/fixtures/" ];
          check_python = true;
        };

        # Everything the package needs at run time and test time, plus mypy.
        devPython = python.withPackages (
          ps:
          beast-compare.dependencies
          ++ [
            ps.mypy
            ps.pymodes
            ps.pytest
            ps.types-pyyaml
          ]
        );

        mypyCheck = pkgs.runCommand "beast-compare-mypy" { nativeBuildInputs = [ devPython ]; } ''
          cp -r ${./pyproject.toml} pyproject.toml
          cp -r ${./src} src
          cp -r ${./tests} tests
          mypy src tests
          touch $out
        '';
      in
      {
        packages = {
          default = beast-compare;
          inherit beast-compare;
        };

        checks = {
          pre-commit-check = precommitCheck;
          package = beast-compare;
          mypy = mypyCheck;
        };

        devShells.default = pkgs.mkShell {
          name = "beast-compare";
          buildInputs = precommitCheck.enabledPackages ++ [
            devPython
            pkgs.zstd
          ];
          shellHook = ''
            ${precommitCheck.shellHook}
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
          '';
        };
      }
    );
}
