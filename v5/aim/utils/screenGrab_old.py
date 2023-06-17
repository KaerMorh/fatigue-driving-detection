import mss
import cv2
import numpy as np


def grab_screen(region = None):
    if region:
        region = region # 没写加参
    else:
        sct = mss.mss()
        screen_width = 1920
        screen_height = 1080
        GM_X, GM_Y = screen_width // 2, screen_height//2  # //整数除法 游戏内窗口区域
        RESIZE_WIDTH , RESIZE_HEIGHT = screen_width//2 , screen_height//2


    monitor = {
        "top": GM_Y,
        "left": GM_X,
        "width": RESIZE_WIDTH,
        "height": RESIZE_HEIGHT
    }



    window_name = 'test'
    while True:
        img = sct.grab(monitor=monitor)
        img = np.array(img)                # imshow只能传array？

        cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)        # 新建一个窗口，方便之后resize，并根据窗口大小调节图片大小（NORRMAL）
        cv2.resizeWindow(window_name,RESIZE_WIDTH,RESIZE_HEIGHT)
        cv2.imshow(window_name, img)
        k = cv2.waitKey(1)
        if k % 256 == 27:                   # 返回键的Ascii为27（ESC）
                cv2.destroyAllWindows()
                exit('结束img_show进程中')


grab_screen()