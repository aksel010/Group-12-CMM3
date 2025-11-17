from blinkstick import blinkstick
import time

# Define hex color codes for LED states using RGB values
OFF = "#000000"  # LED off (black)
RED = "#100000"  # LED red, indicating Stop/Danger
AMBER = "#100600"  # LED amber, indicating a transitional state (caution)
GREEN = "#001000"  # LED green, indicating Go/Safe

# Initialize the BlinkStick device by finding the first available one
bs = blinkstick.find_first()
assert bs is not None, "BlinkStick device not found."  # Raise an error if no device is found
print(bs)  # Output BlinkStick device info for debugging purposes

# List of specific LED indices that correspond to traffic signals for different roads
indices = [5, 7, 0, 3]  # Indices represent road and pedestrian signals in order of operation

def turn_off(delay):
    """
    Turns off all LEDs by setting their color to OFF (black).
    
    Args:
        delay (float): The time to wait (in seconds) after turning off all LEDs.

    This function iterates over the 8 LEDs (assuming the device has 8 LEDs), 
    setting each one to the OFF state. The function then pauses execution 
    for a specified duration using `time.sleep()`.
    
    References:
        - PEP8: Function and Variable Names (https://www.python.org/dev/peps/pep-0008/)
    """
    for i in range(8):  # Assumes there are 8 LEDs
        bs.set_color(index=i, hex=OFF)  # Set each LED to the OFF state
    time.sleep(delay)  # Delay for the specified duration

def set_red(delay):
    """
    Sets specific LEDs (based on the `indices` list) to RED, simulating 
    stop/danger lights for traffic signals.
    
    Args:
        delay (float): The time to wait (in seconds) after setting the LEDs to red.
    
    This function first calls `turn_off()` to ensure all LEDs are off, then 
    sets the LEDs defined in the `indices` list to RED. A delay is added 
    at the end to pause the execution.
    """
    turn_off(0)  # Turn off all LEDs before setting them to red
    for i in indices:  # Iterate through the list of indices to set specific LEDs
        bs.set_color(index=i, hex=RED)  # Set LED at index `i` to red
    time.sleep(delay)  # Pause for the specified delay

def set_color(index, color, delay):
    """
    Sets the color of a specific LED and pauses for a specified duration.
    
    Args:
        index (int): The index of the LED to change.
        color (str): Hexadecimal string representing the color to set the LED to.
        delay (float): The time to wait (in seconds) after changing the LED color.
    
    This function allows direct control of any LED by specifying its index 
    and the desired color. After setting the color, a delay is added to allow 
    for a transition effect.
    
    References:
        - RGB Color Model (https://en.wikipedia.org/wiki/RGB_color_model)
    """
    bs.set_color(index=index, hex=color)  # Set the LED at `index` to the specified `color`
    time.sleep(delay)  # Pause for the specified delay

def animate(r):
    """
    Animates specific traffic light sequences based on the index `r`, 
    simulating a toucan crossing system with red, amber, and green lights.
    
    Args:
        r (int): The current LED index being animated.
    
    The animation logic varies depending on the index `r`, which represents 
    specific road traffic signals:
    - If `r == 0`: Controls lights for Esslemount Road.
    - If `r in [5, 7]`: Controls lights for West Mains Road and Mayfield Road.
    - Otherwise: Handles the pedestrian crossing signal.
    
    References:
        - Chapter 6 "Traffic Control" of the UK Traffic Signs Manual, 2019
          (https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/851465/
          dft-traffic-signs-manual-chapter-6.pdf)
    """
    if r == 0:  # Logic for Esslemount Road
        set_color(r + 1, AMBER, 2)  # Set LED at `r+1` to amber for 2 seconds
        set_color(r, OFF, 0)  # Turn off LED at index `r`
        set_color(r + 1, GREEN, 10)  # Set LED at `r+1` to green for 10 seconds
        set_color(r + 1, AMBER, 3)  # Set LED at `r+1` back to amber for 3 seconds
        set_red(2)  # Set all LEDs in `indices` to red with a 2-second delay

    elif r in [5, 7]:  # Logic for West Mains Road and Mayfield Road
        set_color(r - 1, AMBER, 2)  # Set LED at `r-1` to amber for 2 seconds
        set_color(r, OFF, 0)  # Turn off LED at index `r`
        set_color(r - 1, GREEN, 10)  # Set LED at `r-1` to green for 10 seconds
        set_color(r - 1, AMBER, 3)  # Set LED at `r-1` back to amber for 3 seconds
        set_red(r - 5)  # Set red with a delay that depends on the road index

    else:  # Logic for pedestrian crossing
        set_color(r, OFF, 0)  # Turn off LED at index `r`
        set_color(r - 1, GREEN, 10)  # Set LED at `r-1` to green for 10 seconds
        set_red(9)  # Set all LEDs in `indices` to red with a 9-second delay

# Initial setup: turn off all LEDs and set specific ones to red
turn_off(2)  # Turn off all LEDs with a 2-second delay before starting
set_red(4)  # Set the red lights with a 4-second delay

while True:
    """
    Infinite loop to continuously run the LED animation sequence.
    
    This loop ensures that the traffic light system runs indefinitely, 
    repeating the `animate()` function for each LED index in the `indices` list.
    
    References:
        - Infinite Loop Pattern (https://en.wikipedia.org/wiki/Infinite_loop)
    """
    for r in indices:  # Iterate over each LED index in the `indices` list
        animate(r)  # Animate the traffic light system based on the current index `r`