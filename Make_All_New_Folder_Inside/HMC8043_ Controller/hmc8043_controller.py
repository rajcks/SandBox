#!/usr/bin/env python3
"""
HMC8043 Controller
-------------------
A PySide6 + PyVISA GUI to discover VISA instruments (USB/LAN/Serial),
connect to a Rohde & Schwarz HMC8043 triple-channel power supply,
and control/monitor all three channels.

Requirements:
    pip install PySide6 pyvisa pyvisa-py

On Windows you will typically also want the NI-VISA or R&S VISA runtime
installed so pyvisa can see USBTMC devices. If you don't have a vendor
VISA installed, pyvisa-py (pure python backend) will be used instead.

Run:
    python hmc8043_controller.py
"""

import sys
import traceback
import pyvisa
import PySide6.QtCore
import PySide6.QtWidgets
import PySide6.QtGui


# --------------------------------------------------------------------------
# Instrument wrapper
# --------------------------------------------------------------------------

class HMC8043:
    """Thin SCPI wrapper around a pyvisa resource for the HMC8043."""

    NUM_CHANNELS = 3

    def __init__(self, resource):
        self.inst = resource
        self.inst.timeout = 3000  # ms
        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"

    def idn(self):
        return self.inst.query("*IDN?").strip()

    def select_channel(self, ch: int):
        self.inst.write(f"INST OUT{ch}")

    def set_voltage(self, ch: int, volts: float):
        self.select_channel(ch)
        self.inst.write(f"VOLT {volts:.3f}")

    def set_current(self, ch: int, amps: float):
        self.select_channel(ch)
        self.inst.write(f"CURR {amps:.3f}")

    def output_on(self, ch: int):
        self.select_channel(ch)
        self.inst.write("OUTP ON")

    def output_off(self, ch: int):
        self.select_channel(ch)
        self.inst.write("OUTP OFF")

    def output_state(self, ch: int) -> bool:
        self.select_channel(ch)
        resp = self.inst.query("OUTP?").strip()
        return resp in ("1", "ON")

    def measure_voltage(self, ch: int) -> float:
        self.select_channel(ch)
        return float(self.inst.query("MEAS:VOLT?"))

    def measure_current(self, ch: int) -> float:
        self.select_channel(ch)
        return float(self.inst.query("MEAS:CURR?"))

    def all_outputs_off(self):
        for ch in range(1, self.NUM_CHANNELS + 1):
            self.output_off(ch)

    def close(self):
        try:
            self.inst.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Channel widget
# --------------------------------------------------------------------------

class ChannelPanel(PySide6.QtWidgets.QGroupBox):
    """UI panel for a single HMC8043 output channel."""

    def __init__(self, channel_number: int, parent=None):
        super().__init__(f"CHANNEL {channel_number}", parent)
        self.channel_number = channel_number
        self._build_ui()

    def _build_ui(self):
        layout = PySide6.QtWidgets.QVBoxLayout()

        # Voltage setpoint
        v_row = PySide6.QtWidgets.QHBoxLayout()
        v_row.addWidget(PySide6.QtWidgets.QLabel("Voltage (V):"))
        self.voltage_spin = PySide6.QtWidgets.QDoubleSpinBox()
        self.voltage_spin.setRange(0.0, 32.0)
        self.voltage_spin.setDecimals(3)
        self.voltage_spin.setSingleStep(0.1)
        self.voltage_spin.setValue(0.0)
        v_row.addWidget(self.voltage_spin)
        layout.addLayout(v_row)

        # Current setpoint
        c_row = PySide6.QtWidgets.QHBoxLayout()
        c_row.addWidget(PySide6.QtWidgets.QLabel("Current (A):"))
        self.current_spin = PySide6.QtWidgets.QDoubleSpinBox()
        self.current_spin.setRange(0.0, 10.0)
        self.current_spin.setDecimals(3)
        self.current_spin.setSingleStep(0.1)
        self.current_spin.setValue(0.0)
        c_row.addWidget(self.current_spin)
        layout.addLayout(c_row)

        # Apply button
        self.apply_btn = PySide6.QtWidgets.QPushButton("Apply Setpoints")
        layout.addWidget(self.apply_btn)

        # Output on/off buttons
        out_row = PySide6.QtWidgets.QHBoxLayout()
        self.on_btn = PySide6.QtWidgets.QPushButton("OUTPUT ON")
        self.on_btn.setStyleSheet("background-color:#2e7d32;color:white;font-weight:bold;")
        self.off_btn = PySide6.QtWidgets.QPushButton("OUTPUT OFF")
        self.off_btn.setStyleSheet("background-color:#c62828;color:white;font-weight:bold;")
        out_row.addWidget(self.on_btn)
        out_row.addWidget(self.off_btn)
        layout.addLayout(out_row)

        # Status line
        self.status_label = PySide6.QtWidgets.QLabel("Output: UNKNOWN")
        self.status_label.setAlignment(PySide6.QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Separator
        line = PySide6.QtWidgets.QFrame()
        line.setFrameShape(PySide6.QtWidgets.QFrame.HLine)
        layout.addWidget(line)

        # Measurements
        meas_font = PySide6.QtGui.QFont()
        meas_font.setPointSize(11)
        meas_font.setBold(True)

        self.meas_voltage_label = PySide6.QtWidgets.QLabel("-- V")
        self.meas_voltage_label.setFont(meas_font)
        self.meas_current_label = PySide6.QtWidgets.QLabel("-- A")
        self.meas_current_label.setFont(meas_font)

        meas_grid = PySide6.QtWidgets.QGridLayout()
        meas_grid.addWidget(PySide6.QtWidgets.QLabel("Measured V:"), 0, 0)
        meas_grid.addWidget(self.meas_voltage_label, 0, 1)
        meas_grid.addWidget(PySide6.QtWidgets.QLabel("Measured I:"), 1, 0)
        meas_grid.addWidget(self.meas_current_label, 1, 1)
        layout.addLayout(meas_grid)

        layout.addStretch()
        self.setLayout(layout)

    def set_output_status(self, is_on: bool):
        if is_on:
            self.status_label.setText("Output: ON")
            self.status_label.setStyleSheet("color:#2e7d32;font-weight:bold;")
        else:
            self.status_label.setText("Output: OFF")
            self.status_label.setStyleSheet("color:#c62828;font-weight:bold;")

    def set_measurements(self, volts: float, amps: float):
        self.meas_voltage_label.setText(f"{volts:.3f} V")
        self.meas_current_label.setText(f"{amps:.3f} A")


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(PySide6.QtWidgets.QMainWindow):
    POLL_INTERVAL_MS = 1000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HMC8043 Controller")
        self.resize(900, 560)

        self.rm = None
        self.hmc = None
        self.channel_panels = []

        self._build_ui()
        self._refresh_devices()

        self.poll_timer = PySide6.QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self._poll_measurements)

    # ---------------------------------------------------------------- UI --

    def _build_ui(self):
        central = PySide6.QtWidgets.QWidget()
        root = PySide6.QtWidgets.QVBoxLayout()

        # --- Connection bar ---
        conn_box = PySide6.QtWidgets.QGroupBox("Device")
        conn_layout = PySide6.QtWidgets.QHBoxLayout()

        self.device_combo = PySide6.QtWidgets.QComboBox()
        self.device_combo.setMinimumWidth(400)
        conn_layout.addWidget(self.device_combo, stretch=1)

        self.refresh_btn = PySide6.QtWidgets.QPushButton("Refresh Devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        conn_layout.addWidget(self.refresh_btn)

        self.connect_btn = PySide6.QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect_clicked)
        conn_layout.addWidget(self.connect_btn)

        self.disconnect_btn = PySide6.QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self._disconnect_clicked)
        self.disconnect_btn.setEnabled(False)
        conn_layout.addWidget(self.disconnect_btn)

        conn_box.setLayout(conn_layout)
        root.addWidget(conn_box)

        # --- Status line ---
        status_row = PySide6.QtWidgets.QHBoxLayout()
        self.status_dot = PySide6.QtWidgets.QLabel("●")
        self.status_dot.setStyleSheet("color:red;font-size:16px;")
        self.status_text = PySide6.QtWidgets.QLabel("Disconnected")
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_text)
        status_row.addStretch()
        root.addLayout(status_row)

        # --- Channel panels ---
        channels_box = PySide6.QtWidgets.QGroupBox("Channels")
        channels_layout = PySide6.QtWidgets.QHBoxLayout()
        for ch in range(1, HMC8043.NUM_CHANNELS + 1):
            panel = ChannelPanel(ch)
            panel.setEnabled(False)
            panel.apply_btn.clicked.connect(lambda _=False, c=ch: self._apply_setpoints(c))
            panel.on_btn.clicked.connect(lambda _=False, c=ch: self._set_output(c, True))
            panel.off_btn.clicked.connect(lambda _=False, c=ch: self._set_output(c, False))
            channels_layout.addWidget(panel)
            self.channel_panels.append(panel)
        channels_box.setLayout(channels_layout)
        root.addWidget(channels_box, stretch=1)

        # --- Global controls ---
        global_row = PySide6.QtWidgets.QHBoxLayout()
        self.all_off_btn = PySide6.QtWidgets.QPushButton("ALL OUTPUTS OFF")
        self.all_off_btn.setStyleSheet("background-color:#b71c1c;color:white;font-weight:bold;padding:8px;")
        self.all_off_btn.setEnabled(False)
        self.all_off_btn.clicked.connect(self._all_outputs_off)
        global_row.addStretch()
        global_row.addWidget(self.all_off_btn)
        global_row.addStretch()
        root.addLayout(global_row)

        central.setLayout(root)
        self.setCentralWidget(central)

    # ------------------------------------------------------------- VISA --

    def _refresh_devices(self):
        self.device_combo.clear()
        if pyvisa is None:
            self.device_combo.addItem("pyvisa not installed")
            return
        try:
            if self.rm is None:
                self.rm = pyvisa.ResourceManager()
            resources = self.rm.list_resources()
        except Exception as exc:
            PySide6.QtWidgets.QMessageBox.warning(self, "VISA Error", f"Could not list resources:\n{exc}")
            resources = []

        if not resources:
            self.device_combo.addItem("No devices found")
        else:
            for r in resources:
                self.device_combo.addItem(r)

    def _connect_clicked(self):
        if pyvisa is None:
            PySide6.QtWidgets.QMessageBox.critical(self, "Missing dependency", "pyvisa is not installed.\nRun: pip install pyvisa pyvisa-py")
            return

        resource_name = self.device_combo.currentText()
        if not resource_name or "No devices" in resource_name:
            PySide6.QtWidgets.QMessageBox.warning(self, "No device", "Select a VISA resource first.")
            return

        try:
            if self.rm is None:
                self.rm = pyvisa.ResourceManager()
            resource = self.rm.open_resource(resource_name)
            hmc = HMC8043(resource)
            idn = hmc.idn()
        except Exception as exc:
            PySide6.QtWidgets.QMessageBox.critical(self, "Connection failed", f"Could not connect:\n{exc}")
            traceback.print_exc()
            return

        if "HMC8043" not in idn.upper():
            reply = PySide6.QtWidgets.QMessageBox.question(
                self, "Unexpected device",
                f"Connected device identifies as:\n\n{idn}\n\n"
                "This does not look like an HMC8043. Continue anyway?",
                PySide6.QtWidgets.QMessageBox.Yes | PySide6.QtWidgets.QMessageBox.No
            )
            if reply == PySide6.QtWidgets.QMessageBox.No:
                hmc.close()
                return

        self.hmc = hmc
        self.status_dot.setStyleSheet("color:#2e7d32;font-size:16px;")
        self.status_text.setText(f"Connected: {idn}")
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.all_off_btn.setEnabled(True)
        for panel in self.channel_panels:
            panel.setEnabled(True)

        self._read_all_output_states()
        self.poll_timer.start(self.POLL_INTERVAL_MS)

    def _disconnect_clicked(self):
        self.poll_timer.stop()
        if self.hmc:
            self.hmc.close()
            self.hmc = None
        self.status_dot.setStyleSheet("color:red;font-size:16px;")
        self.status_text.setText("Disconnected")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.all_off_btn.setEnabled(False)
        for panel in self.channel_panels:
            panel.setEnabled(False)
            panel.set_measurements(0.0, 0.0)
            panel.status_label.setText("Output: UNKNOWN")
            panel.status_label.setStyleSheet("")

    # ---------------------------------------------------------- Actions --

    def _apply_setpoints(self, channel: int):
        if not self.hmc:
            return
        panel = self.channel_panels[channel - 1]
        volts = panel.voltage_spin.value()
        amps = panel.current_spin.value()
        try:
            self.hmc.set_voltage(channel, volts)
            self.hmc.set_current(channel, amps)
        except Exception as exc:
            PySide6.QtWidgets.QMessageBox.critical(self, "Write error", f"Channel {channel}:\n{exc}")

    def _set_output(self, channel: int, on: bool):
        if not self.hmc:
            return
        try:
            if on:
                self.hmc.output_on(channel)
            else:
                self.hmc.output_off(channel)
            panel = self.channel_panels[channel - 1]
            panel.set_output_status(on)
        except Exception as exc:
            PySide6.QtWidgets.QMessageBox.critical(self, "Write error", f"Channel {channel}:\n{exc}")

    def _all_outputs_off(self):
        if not self.hmc:
            return
        try:
            self.hmc.all_outputs_off()
            for panel in self.channel_panels:
                panel.set_output_status(False)
        except Exception as exc:
            PySide6.QtWidgets.QMessageBox.critical(self, "Write error", f"All outputs off:\n{exc}")

    def _read_all_output_states(self):
        if not self.hmc:
            return
        for ch in range(1, HMC8043.NUM_CHANNELS + 1):
            try:
                state = self.hmc.output_state(ch)
                self.channel_panels[ch - 1].set_output_status(state)
            except Exception:
                pass

    # ------------------------------------------------------------- Poll --

    def _poll_measurements(self):
        if not self.hmc:
            return
        for ch in range(1, HMC8043.NUM_CHANNELS + 1):
            try:
                v = self.hmc.measure_voltage(ch)
                i = self.hmc.measure_current(ch)
                self.channel_panels[ch - 1].set_measurements(v, i)
            except Exception:
                # Skip a bad poll cycle rather than crashing the GUI
                pass

    def closeEvent(self, event):
        self.poll_timer.stop()
        if self.hmc:
            self.hmc.close()
        event.accept()


# --------------------------------------------------------------------------

def main():
    app = PySide6.QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
