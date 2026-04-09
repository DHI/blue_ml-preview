"""Category definitions for wave data processing."""

from enum import Enum


class ModelScope(Enum):
    """Enumeration of model scope categories for ocean and atmospheric data."""

    Waves = 1
    Ocean = 2
    Atmosphere = 3


class Magnitude(Enum):
    """Enumeration of physical magnitude types for measurements."""

    SignWaveHeight = 1
    WaveDirection = 2
    WavePeriod = 3
    WindSpeed = 4
    WindDirection = 5
    CurrentSpeed = 6
    CurrentDirection = 7
    WaterLevel = 8
    AirPressure = 9
    DirectionalStdDev = 10


class WaveSource(Enum):
    """Enumeration of wave source types."""

    WindSea = 1
    Swell = 2
    Combined = 3


class WaveFeature(Enum):
    """Enumeration of wave feature statistics."""

    Mean = 1
    Peak = 2
    ZeroCrossing = 3


class WindHeight(Enum):
    """Enumeration of wind measurement heights in meters."""

    Ten = 1
    Hundred = 2
