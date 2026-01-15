from radar import MmWaveRadar
radar = MmWaveRadar(cfg_path="/home/eesh/droneFusion/radar/configs/1843_3d.cfg", already_started=False, verbose=True)
radar.connect()
radar.configure()