{
  sources ? import ./lon.nix,
  pkgs ? import sources.nixpkgs { },
}:
let
  inherit (pkgs) lib;
in
{
  devShell = pkgs.mkShell rec {
    buildInputs = with pkgs; [
	  cargo
      libxkbcommon
      libGL

      # WINIT_UNIX_BACKEND=wayland
      wayland

      # WINIT_UNIX_BACKEND=x11
      xorg.libXcursor
      xorg.libXrandr
      xorg.libXi
      xorg.libX11
    ];
    LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";
    WINIT_UNIX_BACKEND = "x11";
  };
}
