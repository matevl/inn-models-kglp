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

          python = pkgs.python311;

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
              uv
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

              if [ ! -d "$venvDir" ]; then
                  uv venv --python="${python.interpreter}" "$venvDir"
              fi

              source "$venvDir/bin/activate"
              uv pip install -e .
            '';
          };
        }
      );
    };
}
