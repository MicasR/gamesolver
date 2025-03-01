import uiautomator2 as u2

d = u2.connect("R9YR810XVMX")  # Connect to device
print(d.device_info)  # Check device info

game = "com.stacity.sort.color.water.drink"
d.app_start(game)

d.screenshot("test_result.png")