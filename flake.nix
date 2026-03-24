{
  description = "echo-vault — local meeting recorder with transcription for Obsidian";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        # Development shell — for contributors
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            ffmpeg
            pipewire.pulse
            uv
          ];

          shellHook = ''
            echo "meetrec dev shell — run 'uv sync' to install Python deps"
          '';
        };

        # User shell — for running meetrec without cloning the repo
        # Usage: nix run github:yastcher/echo-vault
        packages.default = pkgs.writeShellScriptBin "meetrec" ''
          export PATH="${pkgs.lib.makeBinPath [ pkgs.ffmpeg pkgs.pipewire.pulse ]}:$PATH"
          exec ${pkgs.uv}/bin/uvx echo-vault "$@"
        '';
      }
    );
}
