import cv2
import winsound
import numpy as np
from cv2.typing import MatLike
from games.water_color_sorting import schemas as s


def identify_tubes(image_path: str) -> s.GameState:
    """
    Identify test tubes in the water sort puzzle game screenshot.

    Args:
        image_path (str): Path to the game screenshot image

    Returns:
        GameState with number and position of tubes. All color have default values.
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None: raise FileNotFoundError(f"Could not load image from {image_path}")

    # Convert to RGB for visualization (OpenCV loads as BGR) --- Delete Line
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) --- Delete Line

    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding instead of global thresholding
    # This works better for images with varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find tubes
    tube_contours = []
    for contour in contours:
        # Calculate contour area and aspect ratio
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0

        # Test tubes are typically tall and narrow (high aspect ratio)
        # Adjust the area and aspect ratio thresholds based on the specific image
        if area > 500 and aspect_ratio > 1.2 and h > 50:
            tube_contours.append((x, y, w, h))

    # Debug: Draw contours on a copy of the original image
    debug_img = image.copy()
    for x, y, w, h in tube_contours:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("./temp/contours_debug.png", debug_img)

    # Sort tubes from left to right, top to bottom
    # First, group tubes by rows (tubes that are roughly on the same y-level)
    # We consider tubes to be in the same row if their y-coordinates are within 50 pixels
    rows = {}
    for i, (x, y, w, h) in enumerate(tube_contours):
        assigned = False
        for row_y in rows.keys():
            if abs(y - row_y) < 50:  # Tubes in the same row
                rows[row_y].append((i, x, y, w, h))
                assigned = True
                break
        if not assigned:
            rows[y] = [(i, x, y, w, h)]

    # Sort rows by y-coordinate (top to bottom)
    sorted_rows = sorted(rows.items(), key=lambda item: item[0])

    # Sort tubes within each row by x-coordinate (left to right)
    tubes = {}
    tube_index = 1

    for _, row_tubes in sorted_rows:
        # Sort tubes in this row from left to right
        row_tubes.sort(key=lambda tube: tube[1])  # Sort by x-coordinate

        for _, x, y, w, h in row_tubes:
            # Calculate center position of the tube
            center_x = x + w // 2
            center_y = y + h // 2

            tubes[f"tube {tube_index}"] = {
                "position": {"x":center_x, "y":center_y}
            }
            tube_index += 1

    return s.GameState.model_validate(({"tubes": tubes}))



# Dictionary to store discovered colors that aren't in the known list
DISCOVERED_COLORS = {}

def get_closest_color(rgb_value: tuple[int, int, int]) -> tuple[str, tuple[int, int, int]]:
    """
    Find the closest known color to the given RGB value.
    If the color is not recognized, it will be added to discovered colors.

    Args:
        rgb_value (tuple): RGB value to match

    Returns:
        tuple: (color_name, rgb_value) of the closest known color
    """
    min_distance = float('inf')
    closest_color = None

    # First check against known colors

    for know_color in s.WebColor:
        # Calculate Euclidean distance between colors
        distance = np.sqrt(sum((np.array(rgb_value) - np.array(know_color.value)) ** 2))
        if distance < min_distance:
            min_distance = distance
            closest_color = (know_color._name_, know_color.value)

    # Then check against previously discovered colors
    if DISCOVERED_COLORS:
        for color_name, color_rgb in DISCOVERED_COLORS.items():
            distance = np.sqrt(sum((np.array(rgb_value) - np.array(color_rgb)) ** 2))

            if distance < min_distance:
                min_distance = distance
                closest_color = (color_name, color_rgb)

    # If the distance is too large, it might be a new color
    if min_distance > 50:  # Threshold for considering it a new color
        # Create a unique name for this new color
        new_color_id = len(DISCOVERED_COLORS) + 1
        new_color_name = f"unknown-{new_color_id}"

        # Store this new color for future reference
        DISCOVERED_COLORS[new_color_name] = rgb_value

        print(f"⚠️ - Discovered new color: {new_color_name} with RGB value {rgb_value}")

        return (new_color_name, rgb_value)

    return closest_color


def add_color_rgb(image_rgb:MatLike, color: s.Color) -> s.Color:
    """Sample RGB values from image based on position and adds to color"""

    # print(f"{type(color.position.y) = }  {color.position.y = }\n")

    pos_x = max(0, min(color.position.x, image_rgb.shape[1] - 1))
    pos_y = max(0, min(color.position.y, image_rgb.shape[0] - 1))

    # Sample a small region around the point to get a more stable color
    region_size = 4
    x_start = max(0, pos_x - region_size)
    x_end = min(image_rgb.shape[1], pos_x + region_size + 1)
    y_start = max(0, pos_y - region_size)
    y_end = min(image_rgb.shape[0], pos_y + region_size + 1)

    region = image_rgb[y_start:y_end, x_start:x_end]
    avg_color = tuple(map(int, np.mean(region, axis=(0, 1))))
    color_name, color_rgb = get_closest_color(avg_color)

    # Store color information
    color.name = color_name
    color.rgb_code.r, color.rgb_code.g, color.rgb_code.b = color_rgb

    return color


def add_color_details(image_path:str, state: s.GameState) -> s.GameState:
    """Adds color positions to game state"""
    # For each tube, sample colors at 4 vertical positions
    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    debug_img = image_rgb.copy()

    for tube in state.tubes:
        # Calculate the positions for sampling colors
        tube_height = 200  # Estimated height in pixels
        center_x = state.tubes[tube].position.x
        center_y = state.tubes[tube].position.y
        cv2.circle(debug_img, (center_x, center_y), 5, (255, 0, 0), -1)


        # Color 1 - Top position
        state.tubes[tube].color_1.position.x = center_x
        state.tubes[tube].color_1.position.y = center_y - tube_height//3
        add_color_rgb(image_rgb, state.tubes[tube].color_1)
        cv2.circle(debug_img, (state.tubes[tube].color_1.position.x, state.tubes[tube].color_1.position.y), 4, (255, 255, 255), -1)

        # Color 2 - Second from top
        state.tubes[tube].color_2.position.x = center_x
        state.tubes[tube].color_2.position.y = center_y - tube_height//6 + 15 #15 as adjustment
        add_color_rgb(image_rgb, state.tubes[tube].color_2)
        cv2.circle(debug_img, (state.tubes[tube].color_2.position.x, state.tubes[tube].color_2.position.y), 4, (255, 255, 255), -1)

        # Color 3 -  Third from top
        state.tubes[tube].color_3.position.x = center_x
        state.tubes[tube].color_3.position.y = center_y + tube_height//6
        add_color_rgb(image_rgb, state.tubes[tube].color_3)
        cv2.circle(debug_img, (state.tubes[tube].color_3.position.x, state.tubes[tube].color_3.position.y), 4, (255, 255, 255), -1)

        #Color 4 -  Bottom position
        state.tubes[tube].color_4.position.x = center_x
        state.tubes[tube].color_4.position.y = center_y + tube_height//3 + 15
        add_color_rgb(image_rgb, state.tubes[tube].color_4)

        cv2.circle(debug_img, (state.tubes[tube].color_4.position.x, state.tubes[tube].color_4.position.y), 4, (255, 255, 255), -1)


    cv2.imwrite(
        "games/water_color_sorting/temp/color_sampling_debug.png",
        cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
    )
    return state


def is_valid_state(state: s.GameState, fst_state: bool = True) -> s.GameStateReport:
    """Checks if the initial game state is valid"""
    report = s.GameStateReport()
    report.is_game_start = fst_state

    for tube_name in state.tubes.keys():

        # add 1 to tube count
        if "tube" in tube_name: report.tubes += 1

        c1_name = state.tubes[tube_name].color_1.name
        c2_name = state.tubes[tube_name].color_2.name
        c3_name = state.tubes[tube_name].color_3.name
        c4_name = state.tubes[tube_name].color_4.name

        # add to color repetition count
        if c1_name and c1_name != "Empty": report.colors[c1_name] += 1
        if c2_name and c2_name != "Empty": report.colors[c2_name] += 1
        if c3_name and c3_name != "Empty": report.colors[c3_name] += 1
        if c4_name and c4_name != "Empty": report.colors[c4_name] += 1

        if c1_name == "Empty" and c2_name == "Empty" and c3_name == "Empty" and c4_name == "Empty":
            report.empty_tubes += 1

        if (c1_name and c1_name != "Empty") and(
            c2_name and c2_name != "Empty") and(
            c3_name and c3_name != "Empty") and(
            c4_name and c4_name != "Empty"): report.full_tubes += 1

    report.unique_colors = len(report.colors)

    # Rule 1: Each color appears 4 times.
    rule_1 = "Error: Rule 1: Each color appears 4 times."
    for color in report.colors:
        if report.colors[color] != 4:
            report.err_msg.append(rule_1 + f" color: {color}: {report.colors[color]}.")

    # Rule 2: for n tubes, n-2 is equal to the number of unique colors not counting empty.
    rule_2 = "Error: Rule 2: for n tubes, n-2 is equal to the number of unique colors not counting empty."
    if report.tubes - 2 != report.unique_colors:
        report.err_msg.append(rule_2 + f" tubes: {report.tubes}, unique_colors: {report.unique_colors}.")

    # Rule 3: For n tubes, n-2 tubes must be full at the start of the game.
    rule_3 = "Error: Rule 3: For n tubes, n-2 tubes must be full at the start of the game."
    if fst_state and report.tubes - 2 != report.full_tubes:
        report.err_msg.append(rule_3 + f" tubes: {report.tubes}, full_tubes: {report.full_tubes}.")


    # Rule 4: At game start 2 tubes must be empty.
    rule_4 = "Error: Rule 4: At game start 2 tubes must be empty"
    if fst_state and  report.empty_tubes != 2:
        report.err_msg.append(rule_4 + f" empty_tubes: {report.empty_tubes}.")

    # is state valid?
    if not report.err_msg: report.is_valid_state = True

    return report


def extract_from_image(image_path:str) -> s.GameState:
    """
    Extract the game state by identifying tubes and their colors.

    Args:
        image_path (str): Path to the game screenshot image

    Returns:
        GameState
    """

    # First identify tubes using the existing function
    # Returns GameState with correct number of tubes but with default colors.
    state = identify_tubes(image_path)
    # Colors mus be added to the state next.

    # Load the image for color analysis
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # add color details
    state = add_color_details(image_path, state)

    # validate state
    report = is_valid_state(state)

    # add report to state
    state.report = report


    if not state.report.is_valid_state:
        print(state.model_dump())
        winsound.Beep(1000, 1000)
        raise Exception("State is not valid. Read errors above: ☝️")

    return state







