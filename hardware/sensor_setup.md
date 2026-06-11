# Sensor Hardware Setup Guide

## AayuSense E-Tongue Hardware Configuration

### Required Components

| Component | Specification | Purpose |
|-----------|---------------|--------|
| Microcontroller | Arduino Uno / Raspberry Pi | Data acquisition controller |
| pH Electrode | Analog pH sensor + BNC adapter | Acidity measurement |
| Conductivity Probe | EC/TDS sensor module | Ionic concentration |
| Turbidity Sensor | Optical turbidity module | Particulate matter |
| ORP Sensor | ORP electrode + amplifier | Redox potential measurement |
| ADC Module | ADS1115 (16-bit) | Analog-to-digital conversion |
| Power Supply | 5V regulated DC | System power |

### Wiring Diagram

```
[Sensor Array] --> [ADC Module ADS1115] --> [Microcontroller] --> [PC / Cloud]
     |                     |                        |
   pH, EC,              I2C Bus              Serial / Wi-Fi
   Turbidity, ORP                              Data Stream
```

### Calibration Procedure

#### pH Sensor Calibration
1. Prepare pH 4.0, 7.0, and 10.0 buffer solutions
2. Immerse electrode in pH 7.0 buffer; adjust potentiometer to read 7.00
3. Rinse with distilled water; immerse in pH 4.0 buffer; verify reading
4. Record calibration coefficients in `config/calibration.json`

#### Conductivity Calibration
1. Use 1413 µS/cm standard conductivity solution
2. Measure and record sensor voltage output
3. Compute K-factor: `K = standard_value / measured_voltage`

#### ORP Calibration
1. Use Zobell's solution (228 mV at 25°C)
2. Adjust offset in firmware to match reference value

### Data Acquisition Script (Arduino)

The Arduino sketch reads all sensors at 1 Hz and transmits JSON over Serial:

```json
{"timestamp": 1234567890, "ph": 6.82, "conductivity": 1.24, "turbidity": 43.2, "orp": 181.5, "label": ""}
```

### Safety Notes

- Always rinse electrodes with distilled water between samples
- Store pH electrode in KCl storage solution when not in use
- Avoid cross-contamination between sample types
- Dry electrodes thoroughly before storage
