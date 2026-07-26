"""
Validate the EDF writer end-to-end on synthetic telemetry, with no DSI
dependency. Builds ECG (1 kHz), BP (500 Hz), and temperature (1 Hz) channels,
writes EDF+, reads it back, and checks amplitudes/timing survive to 16-bit.
"""
import numpy as np
import pyedflib
from datetime import datetime
from edf_writer import Channel, write_edf

OUT = "test_out.edf"


def build_channels(duration_s=20):
    t_ecg = np.arange(0, duration_s, 1 / 1000.0)
    ecg = 1.2 * np.sin(2 * np.pi * 6 * t_ecg) + 0.15 * np.sin(2 * np.pi * 50 * t_ecg)
    t_bp = np.arange(0, duration_s, 1 / 500.0)
    bp = 100 + 20 * np.sin(2 * np.pi * 6 * t_bp)
    t_temp = np.arange(0, duration_s, 1 / 1.0)
    temp = 37 + 0.2 * np.sin(2 * np.pi * 0.01 * t_temp)
    return [
        Channel("ECG", ecg, 1000.0, "mV"),
        Channel("BP", bp, 500.0, "mmHg"),
        Channel("Temp", temp, 1.0, "degC"),
    ], (ecg, bp, temp)


def main():
    channels, (ecg, bp, temp) = build_channels()
    write_edf(
        OUT, channels,
        start_time=datetime(2026, 6, 23, 9, 0, 0),
        patient_code="MOUSE_07", patient_name="Subject 7",
        recording_additional="synthetic test",
        annotations=[(0.0, 0.0, "Recording start"), (10.0, 0.0, "marker")],
    )

    f = pyedflib.EdfReader(OUT)
    try:
        labels = f.getSignalLabels()
        print("Signals:", labels)
        print("Sample rates:", [f.getSampleFrequency(i) for i in range(f.signals_in_file)])
        print("Start datetime:", f.getStartdatetime())
        print("Patient code:", f.getPatientCode().strip())
        idx = {lab.strip(): i for i, lab in enumerate(labels)}
        r_ecg = f.readSignal(idx["ECG"])
        r_bp = f.readSignal(idx["BP"])
        r_temp = f.readSignal(idx["Temp"])
        ann = f.readAnnotations()
        print("Annotations:", list(zip(ann[0], ann[2])))
    finally:
        f.close()

    def maxerr(orig, read, prange):
        n = min(len(orig), len(read))
        step = prange / 65535.0
        return np.max(np.abs(orig[:n] - read[:n])), step

    e_ecg, s_ecg = maxerr(ecg, r_ecg, ecg.max() - ecg.min())
    e_bp, s_bp = maxerr(bp, r_bp, bp.max() - bp.min())
    e_temp, s_temp = maxerr(temp, r_temp, temp.max() - temp.min())
    print("\nECG  max abs err = %.3e mV   (q-step %.3e)" % (e_ecg, s_ecg))
    print("BP   max abs err = %.3e mmHg (q-step %.3e)" % (e_bp, s_bp))
    print("Temp max abs err = %.3e degC (q-step %.3e)" % (e_temp, s_temp))
    ok = (e_ecg <= 1.2 * s_ecg) and (e_bp <= 1.2 * s_bp) and (e_temp <= 1.2 * s_temp)
    print("\nROUND-TRIP WITHIN 16-BIT QUANTIZATION:", "PASS" if ok else "FAIL")
    assert ok, "Round-trip error exceeds 16-bit quantization!"


if __name__ == "__main__":
    main()
