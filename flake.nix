{
  description = "Development environment for inn-models-kglp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
          };

          python = pkgs.python311;

          libpath = with pkgs; lib.makeLibraryPath [
            stdenv.cc.cc.lib
            zlib
            glib
            libGL
            xorg.libX11
            xorg.libXext
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

              if [ ! -d "${venvDir}" ]; then
                  uv venv ${venvDir}
                  source ${venvDir}/bin/activate
                  uv pip install -e .
              fi
            '';
          };
        }
      );
    };
}
