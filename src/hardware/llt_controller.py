# src/hardware/llt_controller.py

import pyllt as llt
import ctypes as ct
import time


class LLTController:
    """
    A controller class for the Micro-Epsilon scanner using a simple, synchronous model.
    This version is robust and avoids complex threading for the LLT device.
    """

    def __init__(self, exposure_time=200, idle_time=10):
        self.exposure_time = exposure_time
        self.idle_time = idle_time
        self.hLLT = None
        self.is_connected = False
        self.is_transfer_active = False
        self.resolution = 0
        self.scanner_type = ct.c_int(0)
        self.profile_buffer = None
        self.x_buffer = None
        self.z_buffer = None
        self.lost_profiles = ct.c_int()

    def connect(self):
        """Initializes and connects to the LLT device."""
        print("LLT Controller: Attempting to connect...")
        try:
            self.hLLT = llt.create_llt_device(llt.TInterfaceType.INTF_TYPE_ETHERNET)
            if not self.hLLT:
                raise ConnectionError("Failed to create LLT device instance.")

            interfaces = (ct.c_uint * 6)()
            ret = llt.get_device_interfaces_fast(self.hLLT, interfaces, len(interfaces))
            if ret < 1:
                raise ConnectionError(f"Error getting interfaces: {ret}.")

            ret = llt.set_device_interface(self.hLLT, interfaces[0], 0)
            if ret < 1:
                raise ConnectionError(f"Error setting device interface: {ret}")

            ret = llt.connect(self.hLLT)
            if ret < 1:
                raise ConnectionError(f"Error connecting to LLT device: {ret}")

            self.is_connected = True
            print("LLT Controller: Connected successfully.")
            self._configure_scanner()
            return True
        except Exception as e:
            print(f"LLT Controller: Connection failed - {e}")
            self.is_connected = False
            return False

    def _configure_scanner(self):
        """Sets up scanner parameters after connection."""
        available_resolutions = (ct.c_uint * 4)()
        ret = llt.get_resolutions(self.hLLT, available_resolutions, len(available_resolutions))
        if ret < 1: raise ValueError("Could not get resolutions.")

        self.resolution = available_resolutions[0]
        ret = llt.set_resolution(self.hLLT, self.resolution)
        if ret < 1: raise ValueError("Could not set resolution.")
        print(f"LLT Controller: Resolution set to {self.resolution}")

        llt.get_llt_type(self.hLLT, ct.byref(self.scanner_type))
        llt.set_profile_config(self.hLLT, llt.TProfileConfig.PROFILE)
        llt.set_feature(self.hLLT, llt.FEATURE_FUNCTION_TRIGGER, llt.TRIG_INTERNAL)
        llt.set_feature(self.hLLT, llt.FEATURE_FUNCTION_EXPOSURE_TIME, self.exposure_time)
        llt.set_feature(self.hLLT, llt.FEATURE_FUNCTION_IDLE_TIME, self.idle_time)

        self.profile_buffer = (ct.c_ubyte * (self.resolution * 64))()
        self.x_buffer = (ct.c_double * self.resolution)()
        self.z_buffer = (ct.c_double * self.resolution)()
        time.sleep(0.1)
        print("LLT Controller: Scanner configured.")

    def start_transfer(self):
        """Starts the profile transfer from the scanner."""
        if self.is_connected and not self.is_transfer_active:
            ret = llt.transfer_profiles(self.hLLT, llt.TTransferProfileType.NORMAL_TRANSFER, 1)
            if ret >= 1:
                self.is_transfer_active = True
                print("LLT Controller: Profile transfer started.")
            else:
                print(f"LLT Controller: Error starting profile transfer: {ret}")

    def stop_transfer(self):
        """Stops the profile transfer."""
        if self.is_connected and self.is_transfer_active:
            llt.transfer_profiles(self.hLLT, llt.TTransferProfileType.NORMAL_TRANSFER, 0)
            self.is_transfer_active = False
            print("LLT Controller: Profile transfer stopped.")

    def acquire_profile(self):
        """Acquires a single profile synchronously and returns its x and z data."""
        if not self.is_transfer_active:
            return None

        null_ptr = ct.POINTER(ct.c_ushort)()
        null_ptr_int = ct.POINTER(ct.c_uint)()

        ret = llt.get_actual_profile(self.hLLT, self.profile_buffer, len(self.profile_buffer),
                                     llt.TProfileConfig.PROFILE, ct.byref(self.lost_profiles))

        if ret == len(self.profile_buffer):
            convert_ret = llt.convert_profile_2_values(
                self.hLLT, self.profile_buffer, self.resolution,
                llt.TProfileConfig.PROFILE, self.scanner_type, 0, 1,
                null_ptr, null_ptr, null_ptr,
                self.x_buffer, self.z_buffer, null_ptr_int, null_ptr_int
            )
            if convert_ret & llt.CONVERT_X and convert_ret & llt.CONVERT_Z:
                x_data = [self.x_buffer[i] for i in range(self.resolution)]
                z_data = [self.z_buffer[i] for i in range(self.resolution)]
                return (x_data, z_data)
        return None

    def disconnect(self):
        """Stops transfer and disconnects from the device."""
        print("LLT Controller: Disconnecting...")
        if self.is_connected:
            self.stop_transfer()
            llt.disconnect(self.hLLT)
            llt.del_device(self.hLLT)
            self.is_connected = False
        print("LLT Controller: Disconnected.")