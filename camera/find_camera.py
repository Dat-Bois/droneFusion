import usb.core
import usb.util
import subprocess

class find_class(object):
    def __init__(self, class_):
        self._class = class_
    def __call__(self, device):
        if device.bDeviceClass == self._class:
            return True
        for cfg in device:
            intf = usb.util.find_descriptor(cfg, bInterfaceClass=self._class)
            if intf is not None:
                return True
        return False

class FindCamera:

    def __init__(self):
        self.matches = []
        self.__find_cams()
        pass

    def __find_cams(self):
        # Find all USB devices with video interface class (14)
        devices = usb.core.find(find_all=True, custom_match=find_class(14))

        # Get the v4l2ctl data
        v4ctl = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8")
        v4ctl = v4ctl.split("\n")

        for device in devices:
            try:
                output = subprocess.check_output(f'lsusb -tvv | grep /dev/bus/usb/{device.bus:03}/{device.address:03}', shell=True).decode("utf-8")
                id = output.split("\n")[0].strip().split(" ")[0].split("1-", 1)[-1]
                for i in range(len(v4ctl)):
                    if id in v4ctl[i]:
                        self.matches.append((device.bus, device.address ,v4ctl[i+1].strip()))
                        break
            except:
                pass
    
    def find_cam(self, bus : int, address : int) -> str:
        for match in self.matches:
            if match[0] == bus and match[1] == address:
                return match[2]
        return None

    @staticmethod
    def _find_cam(bus : int, address : int) -> str:
        # Find all USB devices with video interface class (14)
        devices = usb.core.find(find_all=True, custom_match=find_class(14))

        # Get the v4l2ctl data
        v4ctl = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8")
        v4ctl = v4ctl.split("\n")

        matches = []

        for device in devices:
            try:
                output = subprocess.check_output(f'lsusb -tvv | grep /dev/bus/usb/{device.bus:03}/{device.address:03}', shell=True).decode("utf-8")
                id = output.split("\n")[0].strip().split(" ")[0].split("1-", 1)[-1]
                for i in range(len(v4ctl)):
                    if id in v4ctl[i]:
                        matches.append((device.bus, device.address ,v4ctl[i+1].strip()))
                        break
            except:
                pass
        
        for match in matches:
            if match[0] == bus and match[1] == address:
                return match[2]
        return None
    
if __name__ == "__main__":
    find = FindCamera()
    print(find.matches)