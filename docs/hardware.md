# AayuSense Hardware Guide — ESP32 Sensor Array

## Device Overview

The AayuSense hardware is built around an **ESP32 microcontroller** that reads
an 8-sensor electrochemical array and transmits JSON payloads over serial/MQTT.

Device IDs used in the project: `ESP32-01`, `ESP32-02`

## Sensor Array

| # | Sensor | Parameter | Field Name | Range | Notes |
|---|--------|-----------|-----------|-------|-------|
| 1 | pH Electrode | Acidity | `pH` | 0–14 | 3-point calibration (pH 4, 7, 10 buffers) |
| 2 | TDS Probe | Total Dissolved Solids | `TDS` | 0–1000 ppm | EC conversion factor ~0.5 |
| 3 | ORP Sensor | Oxidation-Reduction Potential | `orp_mV` | −500 to +500 mV | Platinum electrode |
| 4 | Turbidity Sensor | Optical clarity | `turbidity` | 0–1.0 | Normalised from NTU |
| 5 | Reduction Sensor | Reduction activity | `Reduction_value` | 0–1.0 | Derived from ORP slope |
| 6 | Ionic Sensor | Ionic concentration | `Ionic_value` | 0–1.0 | Conductivity-derived |
| 7 | Salt Content | Salt loading | `Salt_content` | 0–1.0 | Computed from TDS + Ionic |
| 8 | Temperature | Solution temperature | `temp_c` | 15–45 °C | Compensates other sensors |

## ESP32 Firmware Notes

- Sampling rate: configurable (default 1 reading per test session)
- Output: JSON string over USB serial (115200 baud) or MQTT
- Sample ID format: `S{YYYYMMDD}-{NN}` (e.g., `S20250916-01`)
- Timestamp: UTC ISO 8601

## Sensor Calibration

### pH Electrode
1. Prepare buffer solutions: pH 4.0, 7.0, 10.0
2. Rinse electrode with DI water between buffers
3. Record ADC values at each buffer
4. Apply 3-point linear interpolation in firmware

### ORP Sensor
1. Use ORP calibration solution (e.g., Zobell solution, +228 mV at 25°C)
2. Adjust offset in firmware until reading matches reference

### TDS Probe
1. Use 1413 µS/cm conductivity standard
2. Apply temperature compensation using `temp_c` reading
3. Default conversion factor: EC (µS/cm) × 0.5 = TDS (ppm)

### Temperature
- DS18B20 digital sensor (no calibration required)
- Used for automatic temperature compensation of pH and ORP readings

## Sample Preparation Protocol

1. Weigh 1g of dried herbal powder
2. Add 50 mL deionized water
3. Stir for 5 minutes at room temperature
4. Allow sediment to settle for 2 minutes
5. Insert sensor array into supernatant
6. Wait 30 seconds for readings to stabilize
7. Trigger measurement on ESP32

## Connecting to Python Backend

```python
# Read from serial port
import serial, json
from src.esp32_interface import ESP32DataParser

parser = ESP32DataParser()
ser = serial.Serial('COM3', 115200, timeout=5)  # adjust port
line = ser.readline().decode('utf-8').strip()
reading = parser.parse(line)
print(reading.sensor_vector())
```
