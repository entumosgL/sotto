{
  description = "Sotto — невидимый ИИ-суфлёр для собеседований под Linux/Hyprland";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      # CUDA-библиотеки несвободные — разрешаем их только для сборки с CUDA.
      pkgsUnfree = import nixpkgs { inherit system; config.allowUnfree = true; };

      # Python: обычный или с faster-whisper на видеокарте NVIDIA (локальное распознавание в ~20 раз быстрее).
      mkPython = cuda:
        if !cuda then pkgs.python3 else
        pkgs.python3.override {
          packageOverrides = _self: super: {
            ctranslate2 = super.ctranslate2.override {
              ctranslate2-cpp = pkgsUnfree.ctranslate2.override { withCUDA = true; withCuDNN = true; };
            };
          };
        };

      # Python-зависимости. Список один на пакет и dev-shell, чтобы не расходились.
      pythonDeps = ps: with ps; [
        fastapi
        uvicorn
        httpx
        jinja2
        python-multipart
        pydantic
        numpy
        sherpa-onnx     # VAD, голосовые отпечатки, локальный синтез речи
        faster-whisper  # локальное распознавание речи (тренеры, офлайн)
        fastembed       # локальные эмбеддинги для семантического поиска
        sqlite-vec      # векторный индекс внутри SQLite
        pyside6         # оверлей
        telethon        # Telegram (чтение каналов, отправка по клику)
        playwright      # LinkedIn через собственный браузерный профиль
        markdown
        weasyprint      # резюме в PDF
      ];

      # Внешние программы, которые вызывает Sotto.
      runtimeTools = with pkgs; [
        pipewire        # pw-record, pw-play, pw-dump
        wireplumber     # wpctl
        ffmpeg
        wf-recorder
        grim
        slurp
        wl-clipboard
        wl-mirror
        xdg-utils
      ];

      browserEnv = {
        PLAYWRIGHT_BROWSERS_PATH = "${pkgs.playwright-driver.browsers}";
        PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = "true";
      };

      mkSotto = cuda:
        let py = mkPython cuda; in
        py.pkgs.buildPythonApplication {
          pname = "sotto";
          version = "0.1.0";
          pyproject = true;
          src = ./.;
          build-system = [ py.pkgs.setuptools ];
          dependencies = pythonDeps py.pkgs;
          nativeCheckInputs = [ py.pkgs.pytestCheckHook py.pkgs.pytest-asyncio ];
          makeWrapperArgs = [ "--prefix PATH : ${pkgs.lib.makeBinPath runtimeTools}" ]
            ++ pkgs.lib.mapAttrsToList (k: v: "--set ${k} ${v}") browserEnv;
        };

      mkShell = cuda:
        let py = mkPython cuda; in
        pkgs.mkShell {
          packages = [ (py.withPackages (ps: pythonDeps ps ++ [ ps.pytest ps.pytest-asyncio ])) ] ++ runtimeTools;
          env = browserEnv;
          shellHook = ''export PYTHONPATH=$PWD:$PYTHONPATH'';
        };
    in
    {
      packages.${system} = {
        default = mkSotto false;
        cuda = mkSotto true;
      };

      apps.${system}.default = {
        type = "app";
        program = "${mkSotto false}/bin/sotto";
      };

      devShells.${system} = {
        default = mkShell false;
        cuda = mkShell true;
      };

      # home-manager: services.sotto.enable = true; (services.sotto.cuda = true — для NVIDIA)
      homeManagerModules.default = { config, lib, ... }:
        let
          cfg = config.services.sotto;
          pkg = mkSotto cfg.cuda;
        in {
          options.services.sotto = {
            enable = lib.mkEnableOption "Sotto daemon";
            cuda = lib.mkEnableOption "локальное распознавание речи на видеокарте NVIDIA";
          };
          config = lib.mkIf cfg.enable {
            home.packages = [ pkg ];
            systemd.user.services.sotto = {
              Unit = {
                Description = "Sotto interview copilot";
                PartOf = [ "graphical-session.target" ];
                After = [ "graphical-session.target" "pipewire.service" ];
              };
              Service = {
                ExecStart = "${pkg}/bin/sotto run";
                Restart = "on-failure";
                RestartSec = 3;
              };
              Install.WantedBy = [ "graphical-session.target" ];
            };
          };
        };
    };
}
