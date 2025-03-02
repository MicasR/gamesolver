from typing import Dict
from pydantic import BaseModel, field_validator


class PositionInScreen(BaseModel):
    x: int
    y: int


class RgbColorCode(BaseModel):
    r: int
    g: int
    b: int

    @field_validator('r','g','b')
    @classmethod
    def validate_rgb(cls, value:int):
        """Validate that RGB values are between 0 and 255"""
        if not (0 <= value <= 255): raise ValueError("RGB values must be between 0 and 255")
        return value


class Color(BaseModel):
    name: str = "unknown"
    rgb_code: RgbColorCode
    position: PositionInScreen


class Tube(BaseModel):
    """Model representing a tube in the water sorting game"""
    position: PositionInScreen
    color_1: Color
    color_2: Color
    color_3: Color
    color_4: Color


class GameState(BaseModel):
    """Model representing the complete game state of the water sorting puzzle"""
    tubes: dict[str, Tube]


# Example of a game state
# game_state = GameState(tubes={
#     "tube_1": Tube(
#         position=PositionInScreen(x=0, y=0),
#         color_1=Color(rgb_code=RgbColorCode(r=0, g=0, b=0), position=PositionInScreen(x=0, y=0)),
#         color_2=Color(rgb_code=RgbColorCode(r=0, g=0, b=0), position=PositionInScreen(x=0, y=0)),
#         color_3=Color(rgb_code=RgbColorCode(r=0, g=0, b=0), position=PositionInScreen(x=0, y=0)),
#         color_4=Color(rgb_code=RgbColorCode(r=0, g=0, b=0), position=PositionInScreen(x=0, y=0)),
#     ),
#     "tube_2": Tube(
#         position=PositionInScreen(x=1, y=0),
#         color_1=Color(rgb_code=RgbColorCode(r=0, g=0, b=0), position=PositionInScreen(x=1, y=0)),
#         color_2=Color(rgb_code=RgbColorCode(r=0, g=0, b=0), position=PositionInScreen(x=1, y=0)),
#         color_3=Color(rgb_code=RgbColorCode(r=0, g=0, b=0), position=PositionInScreen(x=1, y=0)),
#         color_4=Color(rgb_code=RgbColorCode(r=0, g=0, b=0), position=PositionInScreen(x=1, y=0)),
#     )
# })

