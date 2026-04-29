from src.nosso_mar.digital_twin.sensor_interface import WaveBuoySensor
from src.nosso_mar.digital_twin.anomaly_detector import ExtremeWaveDetector
import torch


def test_buoy_read():
    obs = WaveBuoySensor("b01",(5000.,6000.)).read()
    assert obs.variables["Hs"] > 0

def test_extreme_wave_detector():
    eta = torch.zeros(32,32); eta[16,16] = 10.0
    mask = ExtremeWaveDetector(2.0).detect(eta, Hs=2.0)
    assert mask[16,16].item()
