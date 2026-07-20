# Validation record

Version: 0.1.0

The reference run must satisfy five release gates:

1. zero-signal astrometry false-detection fraction <= 0.10;
2. high-amplitude astrometry recovery fraction >= 0.85;
3. rare-event balanced accuracy >= 0.75;
4. bolide-class precision >= 0.65;
5. no operational group appears in both training and test data.

The benchmark deliberately reports class-balanced and calibration metrics because raw accuracy is misleading in a rare-event setting. All committed outputs are hashed in `product_manifest.json`.
