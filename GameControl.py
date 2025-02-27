import vgamepad as vg


def reset(gamepad):
    gamepad.reset()
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.update()

def play(gamepad,joystick,backward,forward):
    gamepad.reset()
    gamepad.left_joystick(x_value=int( (joystick-1)*20000 ), y_value=0)
    gamepad.left_trigger(value=int( 255 * backward))
    gamepad.right_trigger(value=int( 255 * forward))
    gamepad.update()

def createGamepad():
    return vg.VX360Gamepad()