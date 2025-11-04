{
  description = "CTC (4.0.0)";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixgl.url = "github:guibou/nixGL";
  };
  nixConfig = {
    substituters = [
      "https://cache.flox.dev"
      "https://cache.nixos.org" 
    ];
    trusted-public-keys = [
      "flox-cache-public-1:7F4OyH7ZCnFhcze3fJdfyXYLQw/aV7GEed86nQ7IsOs="
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
    ];
    accept-flake-config = true;
  };
  outputs = inputs @ {self, systems, nixpkgs, flake-parts, nixgl, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import inputs.systems;
      perSystem = {pkgs, lib, config, system, ...}: {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          overlays = [inputs.nixgl.overlay];
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaCapabilities = ["6.0"];
          };
        };
        imports = [
          ./nix/shell.nix
        ];
      };
    };
}