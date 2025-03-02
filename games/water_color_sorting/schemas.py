from typing import Dict
from pydantic import BaseModel, field_validator


class PositionInScreen(BaseModel):
    x: int = 0
    y: int = 0


class RgbColorCode(BaseModel):
    r: int = 0
    g: int = 0
    b: int = 0

    @field_validator('r','g','b')
    @classmethod
    def validate_rgb(cls, value:int):
        """Validate that RGB values are between 0 and 255"""
        if not (0 <= value <= 255): raise ValueError("RGB values must be between 0 and 255")
        return value


class Color(BaseModel):
    name: str = "unknown"
    rgb_code: RgbColorCode = RgbColorCode()
    position: PositionInScreen = PositionInScreen()


class Tube(BaseModel):
    """Model representing a tube in the water sorting game"""

    position: PositionInScreen
    color_1: Color = Color()
    color_2: Color = Color()
    color_3: Color = Color()
    color_4: Color = Color()


class GameState(BaseModel):
    """Model representing the complete game state of the water sorting puzzle"""
    tubes: dict[str, Tube]



