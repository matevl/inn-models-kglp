{
  description = "Development environment for inn-models-kglp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
          };

          python = pkgs.python312;

          libpath =
            with pkgs;
            lib.makeLibraryPath [
              stdenv.cc.cc.lib
              zlib
              glib
              libGL
              libx11
              libxext
            ];
        in
        {
          default = pkgs.mkShell {
            name = "inn-models-dev-shell";

            packages = with pkgs; [
              python
              python.pkgs.venvShellHook
              ruff
              uv
              python.pkgs.tensorboard
              git
              curl
              bash
              glibcLocales
            ];

            venvDir = ".venv";

            shellHook = ''
              export LD_LIBRARY_PATH="${libpath}:$LD_LIBRARY_PATH"
              export PATH="$HOME/.local/bin:$PATH"

              # Force uv to use the Nix-provided Python (prevents missing Python.h)
              export UV_PYTHON_DOWNLOADS="never"

              if [ "${CI:-false}" != "true" ]; then
                # Recreate venv if Python version changed
                if [ -d "$venvDir" ]; then
                  VENV_PYTHON_VERSION=$("$venvDir/bin/python" --version 2>/dev/null || echo "none")
                  FLAKE_PYTHON_VERSION=$("${python.interpreter}" --version)
                  if [ "$VENV_PYTHON_VERSION" != "$FLAKE_PYTHON_VERSION" ]; then
                    echo "Python version mismatch (Venv: $VENV_PYTHON_VERSION, Flake: $FLAKE_PYTHON_VERSION). Recreating .venv..."
                    rm -rf "$venvDir"
                  fi
                fi

                if [ ! -d "$venvDir" ]; then
                  uv venv --python="${python.interpreter}" "$venvDir"
                fi

                source "$venvDir/bin/activate"
                uv pip install -e .
              fi
            '';
          };
        }
      );
    };
}
