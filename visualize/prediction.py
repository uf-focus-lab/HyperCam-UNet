#!python
# Python Packages
import sys
from os.path import isfile, isdir
from functools import cache
# PIP Packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Custom Packages
import util.env as env
# Components
import cvtb
from cvtb.UI import KEYCODE
from lib.Context import Context
from util.param import LED_LIST, REF_BANDS
from .util.view import view, trim, text
from .util.plot import plot

LED_BANDS = [led.bandwidth for led in LED_LIST]
led_bands = np.array(LED_BANDS).copy().astype(np.float64)

def diff(X: np.ndarray, Y: np.ndarray):
    B, G, R = [np.zeros(X.shape, dtype=np.uint8) for _ in range(3)]
    R = (X > Y) * (X - Y)
    B = (Y > X) * (Y - X)
    return np.stack([B, G, R], axis=2)

def UI_LOOP(ctx: Context, REF, PRD, RAW, DIF, prefix="Unknown"):
    # Interpolate spectral raw
    def raw(i):
        wavelength = REF_BANDS[i]
        w = cvtb.fx.gaussian(wavelength, 5)(led_bands)
        idx = np.argmax(w)
        gray = np.average(cvtb.types.F32(RAW), axis=2, weights=w)
        print(gray.shape)
        gray = cvtb.types.U8(gray)
        gray = cv2.resize(gray, (307, 307))
        return np.stack([gray] * 3, axis=2)
    def dif(i):
        return cvtb.types.U8(DIF[:, :, :, i])
    # Mouse callback
    plot_list = []
    def mouseCallback(e, x, y, flags, param):
        id, pos = update_cursor((x, y), e == cv2.EVENT_LBUTTONUP)
        if id is not None:
            plot_list.append((id, pos))
        if pos is not None:
            # Update band plot
            render_plot(id, pos)
            pass


    render_plot = plot(
        "Pixel Spectrum",
        [
            (REF, ("Ground Truth", REF_BANDS, 'r-')),
            (PRD, ("Prediction"  , REF_BANDS, 'b-')),
            (RAW, ("Raw Input"   , LED_BANDS, 'cx')),
        ],
        size=[3, 5]
    )


    ref = cvtb.types.U8(REF)
    prd = cvtb.types.U8(PRD)

    rasterize, update_cursor, render = view(
        f"Inspecting {prefix}",
        [
            (ref, "Ground Truth"          , ( 64,  64, 255)),
            (prd, "Model Prediction"      , ( 64, 255, 255)),
            (raw, "Raw image (our camera)", (200, 255,  64)),
            (dif, "Prediction error"      , (128, 255, 128)),
        ],
        callback=mouseCallback
    )

    while True:
        key = cv2.waitKey(100)
        if key < 0:
            continue
        elif key == KEYCODE["escape"] or key == ord("q"):
            return False
        elif key == KEYCODE["enter"] or key == ord("s"):
            path = ctx.path
            print("Saving...")
            # Generate grid image
            update_cursor(None)
            grid_img, band_number = rasterize()
            cv2.imwrite(str(path / "grid.png"), grid_img)
            # Generate band plots
            if len(plot_list) == 0: return True
            imgs = []
            for i, (id, pos) in zip(range(len(plot_list)), plot_list):
                img = render_plot(id, pos, draw_legend=(i + 1 == len(plot_list)))
                filename = f"band_{id}.png"
                cv2.imwrite(str(path / filename), img)
                if i > 0:
                    # img[190:-174, :158] = 255
                    img[190:-150, :118] = 255
                    img = img[:, 80:]
                if i + 1 < len(plot_list):
                    img[190:-150, -80:] = 255
                    img = img[:, :-40]
                imgs.append(img)
            img = np.concatenate(imgs, axis=1)
            cv2.imwrite(str(path / "bands.png"), img)
            # Merge the results
            SIZE = 1236
            pad_top = 255 * np.ones((190, SIZE, 3), dtype=np.uint8)
            pad_bottom = 255 * np.ones((174, SIZE, 3), dtype=np.uint8)

            blank = np.ones((256, SIZE, 3), dtype=np.uint8) * 255
            txt = trim(text(
                blank,
                "Sample Points",
                (0, 128), color=(0, 0, 0), scale=1.4, w=2
            ))
            # Add text into blank space
            h, w, _ = txt.shape
            rx = [int((SIZE - w) / 2), None]
            rx[1] = rx[0] + w
            ry = [120 - h, 120]
            pad_bottom[ry[0]:ry[1], rx[0]:rx[1]] = txt

            grid_img = np.concatenate((pad_top, cv2.resize(grid_img, (SIZE, SIZE)), pad_bottom), axis=0)

            merged_img = trim(np.concatenate(
                (grid_img, 255 * np.ones((img.shape[0], 20, 3), dtype=np.uint8), img),
                axis=1
            ))
            cv2.imwrite(str(path) + '.png', merged_img)
            return True
        else:
            return key


def normalizeIndex(idx: int, arr: list):
    if idx >= len(arr): return len(arr) - 1
    if idx < 0: return 0
    return idx

@cache
def load(fileID: str, type="raw"):
    path = env.DATA_PATH / type / f"{fileID}.npy"
    assert isfile(path), path
    return np.load(path)

activeDatasetIdx = 0
activePointIdx = 0

def main(ctx: Context, **kwargs: tuple[list[str], np.ndarray]):
    global activeDatasetIdx, activePointIdx
    dataset_list = list(kwargs.items())
    # Normalize DataSet Index
    activeDatasetIdx = normalizeIndex(activeDatasetIdx, dataset_list)
    # Get the current dataset
    prefix, (idList, predList) = dataset_list[activeDatasetIdx]
    # Validate the index
    activePointIdx = normalizeIndex(activePointIdx, idList)
    # Drive current displayed point
    fileID = idList[activePointIdx]
    PRD = predList[activePointIdx]
    RAW = load(fileID, 'raw')
    REF = load(fileID, 'ref')
    DIF = diff(PRD, REF)
    # Start main UI loop
    key = UI_LOOP(ctx, REF, PRD, RAW, DIF, prefix=f"{fileID} @ Run {ctx.id} ({prefix})")
    # Match key to action
    if (not key) or key == ord("q"): return False
    # Arrow keys (up & down) switch the dataset
    elif key == KEYCODE["arrow_up"]: activeDatasetIdx -= 1
    elif key == KEYCODE["arrow_down"]: activeDatasetIdx += 1
    # Arrow keys (left & right) navigate the dataset
    elif key == KEYCODE["arrow_left"]: activePointIdx -= 1
    elif key == KEYCODE["arrow_right"]: activePointIdx += 1
    else: return True
    cv2.destroyAllWindows()
    cv2.waitKey(10)
    return True

def load_pred_list(path, npyName = "U_Net.prediction.npy", listName = "list.txt"):
    assert isfile(path / listName), f"Prediction file {listName} does not exist under {path}"
    idList = open(path / listName, 'r').read().strip().split('\n')
    assert isfile(path / npyName), f"Prediction file {npyName} does not exist under {path}"
    dataset = np.load(path / npyName).swapaxes(1, 3)
    return idList, dataset

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("runID", type=str, help="The runID to inspect")
    runID = parser.parse_args().runID
    runPath = env.RUN_PATH / runID
    assert(isdir(runPath)), f"Run {runID} does not exist"
    with Context(runID, runPath / "visualize") as ctx:
        train = load_pred_list(runPath / "train")
        test = load_pred_list(runPath / "test")
        while main(ctx, train=train, test=test): pass
