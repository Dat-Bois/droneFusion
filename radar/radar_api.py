import os
import sys
import time
import logging
import numpy as np
from typing import Dict

# Configure standard python logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from . import parser_lib
    from . import parseFrame
except ImportError as e:
    logger.error(f"Failed to import parsing libraries. Ensure parser_lib.py and parseFrame.py are accessible. {e}")
    sys.exit(1)

class MmWaveRadar:

    def __init__(self, cfg_path, already_started=False, verbose=True):
        """
        Initialize the Radar API object.

        Args:
            cfg_path (str): Path to the radar configuration file (.cfg).
            already_started (bool): If True, skips sending configuration commands (assumes radar is running).
            verbose (bool): If True, enables printing of status messages.
        """
        self.cfg_path = cfg_path
        self.already_started = already_started
        
        if not verbose:
            logger.setLevel(logging.WARNING)

        self.cli_com = None
        self.data_com = None
        self.parser_type = None
        self.connected = False

        if not self.already_started and (not self.cfg_path or not os.path.exists(self.cfg_path)):
            raise FileNotFoundError(f"Configuration file not found at: {self.cfg_path}")

    def connect(self):
        """
        Auto-detects and opens the CLI and Data COM ports for the radar.
        """
        logger.info("Attempting to connect to radar hardware...")
        try:
            guii_dict = {
                "alreadyStarted": "True" if self.already_started else "False"
            }
            self.parser_type, self.cli_com, self.data_com = parser_lib.get_coms_ports(guii_dict)
            logger.info(f"COM ports found. CLI: {self.cli_com.port}, Data: {self.data_com.port}")
            logger.info(f"Parser type detected: {self.parser_type}")
            self.connected = True
        except Exception as e:
            logger.error(f"Failed to connect to radar: {e}")
            self.connected = False
            raise e

    def configure(self):
        """
        Sends the configuration commands from the .cfg file to the radar via the CLI port.
        If 'already_started' was set to True in __init__, this method does nothing.
        """
        if not self.connected:
            raise RuntimeError("Radar is not connected. Call connect() first.")
        if self.already_started:
            logger.info("Radar already started. Skipping configuration.")
            return
        try:
            logger.info(f"Sending configuration from: {self.cfg_path}")
            with open(self.cfg_path, 'r') as f:
                cfg_lines = f.readlines()
            guii_dict = {"alreadyStarted": "False"}
            parser_lib.sendCfg(self.cli_com, cfg_lines, guii_dict, echo=False)
            logger.info("Configuration sent successfully.")
            time.sleep(1.0) # wait till ready
        except Exception as e:
            logger.error(f"Error during configuration: {e}")
            raise e

    def get_frame(self, timeout=None) -> Dict[str, np.ndarray] | None:
        """
        Grabs a single frame of data from the radar.

        Returns:
            dict: The parsed frame data containing 'pointCloud', 'numDetectedPoints', etc.
                  Returns None if the frame was invalid or empty (zero points).
            
            Data['pointCloud'] is an (N, 7) numpy array: **[x, y, z, doppler, snr, noise, track_id]**
        """
        if not self.connected:
            raise RuntimeError("Radar is not connected.")

        try:
            com_port = self.data_com if self.parser_type != "SingleCOMPort" else self.cli_com
            raw_frame_data = parser_lib.readAndParseUartDoubleCOMPort(com_port, self.parser_type)
            output_dict = parseFrame.parseStandardFrame(raw_frame_data)
            if output_dict.get('error', 0) != 0:
                return None
            if 'pointCloud' in output_dict and output_dict.get('numDetectedPoints', 0) > 0:
                return output_dict
            return None
        except Exception as e:
            logger.error(f"Error grabbing frame: {e}")
            return None

    def close(self):
        """
        Cleanly closes the serial ports.
        """
        logger.info("Closing radar connection...")
        if self.cli_com and hasattr(self.cli_com, 'is_open') and self.cli_com.is_open:
            self.cli_com.close()
        if self.data_com and hasattr(self.data_com, 'is_open') and self.data_com.is_open:
            self.data_com.close()
        self.connected = False
        logger.info("Disconnected.")

    def __enter__(self):
        """Context manager entry support"""
        self.connect()
        self.configure()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit support"""
        self.close()

if __name__ == "__main__":
    CONFIG_FILE = "configs/1843_short_range.cfg" 
    radar = MmWaveRadar(cfg_path=CONFIG_FILE, already_started=False)
    try:
        radar.connect()
        radar.configure()
        print("\n--- Starting Data Stream (Press Ctrl+C to stop) ---\n")
        while True:
            data = radar.get_frame()
            if data:
                num_points = data['numDetectedPoints']
                # Data['pointCloud'] is an (N, 7) numpy array: 
                # [x, y, z, doppler, snr, noise, track_id]
                points = data['pointCloud']
                print(f"Frame detected: {num_points} points")
                print(f"   First point (XYZ): {points[0, :3]}")
                
            time.sleep(0.05) # 20 Hz 

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        radar.close()