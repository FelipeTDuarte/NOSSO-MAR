"""
Launch the NOSSO-MAR digital twin for a wave energy farm site.

Usage:
    python scripts/run_digital_twin.py --config configs/digital_twin/site_atlantic.yaml
"""
import argparse, yaml, asyncio, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.nosso_mar.digital_twin.twin_manager import DigitalTwinManager
    from src.nosso_mar.digital_twin.sensor_interface import WaveBuoySensor

    twin = DigitalTwinManager(cfg)

    # Register sensors from config
    for s in cfg.get("sensors", []):
        twin.register_sensor(WaveBuoySensor(s["id"], tuple(s["position"])))

    logger.info("Starting digital twin ...")
    loop = asyncio.get_event_loop()
    twin.start()
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        twin.stop()
        logger.info("Digital twin stopped.")


if __name__ == "__main__":
    main()
