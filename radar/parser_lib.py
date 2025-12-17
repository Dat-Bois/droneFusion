from time import sleep
import pathlib

try:
    import serial
    import serial.tools.list_ports
except:
    # it's not pip install serial, but pip install pyserial...
    print("python -m pip install pyserial!!!!")
    raise

# parser lib
import sys
import os
import time
current_dir = pathlib.Path(__file__).parent.parent
package_dir = current_dir.parent / 'visualizers/Applications_Visualizer/common'
sys.path.insert(0, str(package_dir))

from . import parseTLVs
from . import parseFrame

MANUAL_OVERRIDE = True
# Set to True for Linux Users

UART_MAGIC_WORD = bytearray(b'\x02\x01\x04\x03\x06\x05\x08\x07')

DATACOM_SDK3 = "Silicon Labs Dual CP2105 USB to UART Bridge: Standard COM Port"
CLICOM_SDK3 = "Silicon Labs Dual CP2105 USB to UART Bridge: Enhanced COM Port"

CLICOM_SDK5 = "XDS110 Class Application/User UART"
DATACOM_SDK5 = "XDS110 Class Auxiliary Data Port"

CLICOM_SDK6 = "L684x"

def find_baud_rate(port):
    baud_rates = ['115200', '921600', '1250000']
    for baud_rate in baud_rates:
        try:
            ser = serial.Serial(port, baud_rate, timeout=1)
            start_time = time.time()
            data = bytearray()
            print(f"Trying baud rate {baud_rate} for port {port}")
            while time.time() - start_time < 0.5:
                chunk = ser.read(1024)
                data.extend(chunk)
                if UART_MAGIC_WORD in data:
                    print(f"\nFound magic word at baud rate {baud_rate} for port {port}")
                    return baud_rate
            ser.close()
        except serial.SerialException as e:
            print(f"Error at baud rate {baud_rate}: {e}")
    return None

def get_coms_ports(guii, echo=True):
    if MANUAL_OVERRIDE == True:
        
        PARSERTYPE = "DoubleCOMPort" 
        CFG_PORT = "/dev/ttyACM0"
        CFG_BAUD = 115200
        DATA_PORT = "/dev/ttyACM1"
        DATA_BAUD = 921600
        CFG = serial.Serial(CFG_PORT, CFG_BAUD, parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE, timeout=0.6)
        DATA = serial.Serial(DATA_PORT, DATA_BAUD, parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE, timeout=0.6)
        
    else:
        ports = serial.tools.list_ports.comports()
        #print(f"Detected ports: {[p.device + ' (' + p.description + ')' for p in ports]}")

        CFG = None
        DATA = None
        PARSERTYPE = None
        cfg_count = 0
        data_count = 0
        for p in ports:
            if p.description.find(CLICOM_SDK3) >= 0:
                if cfg_count == 0:
                    CFG = serial.Serial(p.device, 115200, parity=serial.PARITY_NONE,
                                        stopbits=serial.STOPBITS_ONE, timeout=0.6)
                    cfg_count += 1
                    break
            if p.description.find(CLICOM_SDK5) >= 0:
                if cfg_count == 0 and guii["alreadyStarted"] == "False":
                    CFG = serial.Serial(p.device, 115200, parity=serial.PARITY_NONE,
                                        stopbits=serial.STOPBITS_ONE, timeout=0.6)
                    cfg_count += 1
                break
        if guii["alreadyStarted"] == "False":
            for p in ports:
                if p.description.find(DATACOM_SDK3) >= 0:
                    DATA = serial.Serial(p.device, 921600, parity=serial.PARITY_NONE,
                                        stopbits=serial.STOPBITS_ONE, timeout=0.6)
                    data_count += 1
                    PARSERTYPE = "DoubleCOMPort"
                    break
                if p.description.find(DATACOM_SDK5) >= 0:
                    CFG.write(b"version\r\n")
                    sleep(0.1)
                    ack = CFG.readline(1024)
                    ack = CFG.readline(1024)
                    if b'L684x' in ack:
                        PARSERTYPE = "DoubleCOMPort6844"
                        DATA = serial.Serial(p.device, 1250000, parity=serial.PARITY_NONE,
                                            stopbits=serial.STOPBITS_ONE, timeout=0.6)
                        break
                    if b'WR18' in ack or b'WR16' in ack or b'WR14' in ack:
                        PARSERTYPE = "DoubleCOMPort6844"
                        DATA = serial.Serial(p.device, 921600, parity=serial.PARITY_NONE,
                                            stopbits=serial.STOPBITS_ONE, timeout=0.6)
                        break                    
                    else:
                        PARSERTYPE = "SingleCOMPort"
                        DATA = CFG
                        break
        elif guii["alreadyStarted"] == "True":
            for p in ports:
                if p.description.find(DATACOM_SDK3) >= 0:
                    baud_rate = find_baud_rate(p.device)
                    if baud_rate:
                        DATA = serial.Serial(p.device, baud_rate, parity=serial.PARITY_NONE,
                                            stopbits=serial.STOPBITS_ONE, timeout=0.6)
                        PARSERTYPE = "DoubleCOMPort"
                        CFG = "Don't Care"
                        break
                    else:
                        print("Unable to find magic word for already started")
                    data_count += 1
                if p.description.find(DATACOM_SDK5) >= 0:
                    baud_rate = find_baud_rate(p.device)
                    if baud_rate:
                        DATA = serial.Serial(p.device, baud_rate, parity=serial.PARITY_NONE,
                                            stopbits=serial.STOPBITS_ONE, timeout=0.6)
                        PARSERTYPE = "DoubleCOMPort"
                        CFG = "Don't Care"
                        break
                    data_count += 1
                if p.description.find(CLICOM_SDK5) >= 0:
                    baud_rate = find_baud_rate(p.device)
                    if baud_rate:
                        DATA = serial.Serial(p.device, baud_rate, parity=serial.PARITY_NONE,
                                            stopbits=serial.STOPBITS_ONE, timeout=0.6)
                        CFG = DATA
                        PARSERTYPE = "SingleCOMPort"
                        break
                    data_count += 1

        if PARSERTYPE == None or CFG == None or DATA == None:
            print("\nOne of the following was unable to be found and is still none: PARSERTYPE, CFG, DATA")
            print("PARSERTYPE: ", PARSERTYPE)
            print("CFG: ", CFG)
            print("DATA: ", DATA, "\n")
    return PARSERTYPE, CFG, DATA


def sendCfg(cliCom, cfg, guii, echo=True):
    """ Method from uartParser in gui_parser

    Parameters:
    ----------
    cliCom: pyserial serial object
    cfg: list of str
    each line in the cfg file is an element in the list

    """

    cfg_cleaned = []
    #print(f"Sending cfg: {cfg}")
    print("--- Sending Configuration to Radar ---")

    # --- Step 1: Clean the configuration list ---
    # This loop prepares the commands by removing comments, empty lines,
    # and ensuring each command ends with a newline character.
    for line in cfg:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('%'):
            continue
        cfg_cleaned.append(stripped_line + '\n')
        
# --- Step 2: Send commands one by one ---
    for line in cfg_cleaned:
        command = line.strip()
        print(f"Sending command: {command}")

        # Write the command to the serial port
        cliCom.write(line.encode())
        # Give the device a moment to process the command
        sleep(0.1)

        # --- Step 3: Smart, Non-Blocking Read for a Response ---
        # This is the key change. Instead of waiting forever, we quickly
        # check if the radar sent anything back.
        if cliCom.in_waiting > 0:
            # Read everything currently in the input buffer without waiting
            response = cliCom.read_all()
            if echo:
                # Decode with 'ignore' to prevent errors on weird characters
                print(f"Response: {response.decode(errors='ignore').strip()}")

        # --- Step 4: Special handling for baudRate command ---
        # This part remains the same, as it's critical for certain configs.
        if command.startswith("baudRate"):
            print("********** Changing baud rate ***************")
            try:
                _, new_baud = command.split()
                cliCom.baudrate = int(new_baud)
                print(f"Baud rate changed to {new_baud}")
            except Exception as e:
                raise ValueError(f"Error changing baud rate: {e}")

    # --- Step 5: Finalize ---
    # Give a final short amount of time for the buffer to clear before exiting
    sleep(0.1)
    cliCom.reset_input_buffer()
    print("--- Configuration successfully sent ---")


def parse_frame(frameData, parserType):
    """ parses the frame received from COM port

    Parameters:
    -----------
    frameData: bytearray

    Returns:
    --------
    outputDict: dict
    TLV into a dict, actual keys and values dependingon the lab

    Raises:
    -------
    ValueError
    """

    if (parserType in ["DoubleCOMPort", "DoubleCOMPort6844", "SingleCOMPort", "Unknown"]):
        outputDict = parseFrame.parseStandardFrame(frameData)
    else:
        raise ValueError("parserType set to wrong value")

    return outputDict


def readAndParseUartDoubleCOMPort(dataCom, parserType):
    """ adapted from gui_parser.py

    Parameters:
    ----------
    dataCom: serial.serial object
    """

    # Find magic word, and therefore the start of the frame
    index = 0
    magicByte = dataCom.read(1)
    frameData = bytearray(b'')

    while (1):
        
        # If the device doesnt transmit any data, the COMPort read function will eventually timeout
        # Which means magicByte will hold no data, and the call to magicByte[0] will produce an error
        # This check ensures we can give a meaningful error
        if (len(magicByte) < 1):
            magicByte = dataCom.read(1)

        # Found matching byte
        elif (magicByte[0] == UART_MAGIC_WORD[index]):
            index += 1
            frameData.append(magicByte[0])
            if (index == 8): # Found the full magic word
                break
            magicByte = dataCom.read(1)

        else:
            # When you fail, you need to compare your byte against that byte (ie the 4th) AS WELL AS compare it to the first byte of sequence
            # Therefore, we should only read a new byte if we are sure the current byte does not match the 1st byte of the magic word sequence
            if (index == 0): 
                magicByte = dataCom.read(1)
            index = 0  # Reset index
            frameData = bytearray(b'')  # Reset current frame data

    # Read in version from the header
    versionBytes = dataCom.read(4)

    frameData += bytearray(versionBytes)

    # Read in length from header
    lengthBytes = dataCom.read(4)
    frameData += bytearray(lengthBytes)
    frameLength = int.from_bytes(lengthBytes, byteorder='little')

    # Subtract bytes that have already been read, IE magic word, version, and length
    # This ensures that we only read the part of the frame in that we are lacking
    frameLength -= 16

    # Read in rest of the frame
    frameData += bytearray(dataCom.read(frameLength))
    
    # MCV adding sleep here for multi-threading friendliness
    sleep(0.001)

    return frameData
