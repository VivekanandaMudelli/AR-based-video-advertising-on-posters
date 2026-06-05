import cv2
import numpy as np
import sys

# ── CONFIG ─────────────────────────────────────────────────────────

LOGO_IMAGE       = "logo1.jpeg"
OVERLAY_VIDEO    = "overlay.mp4"
MIN_LOGO_MATCHES = 25
MIN_AREA_RATIO   = 0.05
MAX_AREA_RATIO   = 0.95
APPROX_EPSILON   = 0.02

PHONE_CAM_URL = "http://172.31.100.13:4747/video"


# ── Poster detection ───────────────────────────────────────────────

def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def detect_poster_corners(frame):
    h, w = frame.shape[:2]
    area = w * h

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0

    for cnt in contours:
        a = cv2.contourArea(cnt)

        if not (area * MIN_AREA_RATIO < a < area * MAX_AREA_RATIO):
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPSILON * peri, True)

        if len(approx) == 4 and a > best_area:
            best_area = a
            best = order_corners(approx)

    return best


# ── Logo detection ────────────────────────────────────────────────

def build_logo_detector(logo_path):
    logo = cv2.imread(logo_path)

    if logo is None:
        print("Logo not found")
        sys.exit(1)

    gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

    try:
        det = cv2.SIFT_create(nfeatures=300)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        print("Using SIFT")
    except:
        det = cv2.ORB_create(nfeatures=300)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        print("Using ORB")

    kp, desc = det.detectAndCompute(gray, None)

    return det, matcher, kp, desc


def logo_present(frame, corners, det, matcher, kp_logo, desc_logo):
    W, H = 640, 480

    dst = np.float32([[0,0],[W,0],[W,H],[0,H]])
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (W, H))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    kp_frame, desc_frame = det.detectAndCompute(gray, None)

    if desc_frame is None or len(kp_frame) < MIN_LOGO_MATCHES:
        return False

    matches = matcher.knnMatch(desc_logo, desc_frame, k=2)

    good = [m for m, n in matches if m.distance < 0.65 * n.distance]

    return len(good) >= MIN_LOGO_MATCHES


# ── Overlay ───────────────────────────────────────────────────────

def warp_and_composite(frame, vframe, corners):
    h, w = frame.shape[:2]
    vh, vw = vframe.shape[:2]

    src = np.float32([[0,0],[vw,0],[vw,vh],[0,vh]])

    H, _ = cv2.findHomography(src, corners)

    if H is None:
        return frame

    warped = cv2.warpPerspective(vframe, H, (w, h))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)

    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    result = np.where(mask3 == 255, warped, frame)

    return result


# ── MAIN ──────────────────────────────────────────────────────────

def main():
    det, matcher, kp_logo, desc_logo = build_logo_detector(LOGO_IMAGE)

    vid = cv2.VideoCapture(OVERLAY_VIDEO)

    if not vid.isOpened():
        print("Video not found")
        sys.exit(1)

    # PHONE CAMERA HERE
    cap = cv2.VideoCapture(PHONE_CAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Cannot connect to phone camera")
        sys.exit(1)

    print("Using phone camera...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received from phone")
            break

        corners = detect_poster_corners(frame)

        if corners is not None:
            if logo_present(frame, corners, det, matcher, kp_logo, desc_logo):

                ret_v, vframe = vid.read()
                if not ret_v:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_v, vframe = vid.read()

                output = warp_and_composite(frame, vframe, corners)

                text = "Logo detected → AR ON"
                color = (0,255,0)

            else:
                output = frame.copy()
                pts = corners.astype(np.int32).reshape((-1,1,2))
                cv2.polylines(output, [pts], True, (0,255,255), 2)

                text = "Poster detected, no logo"
                color = (0,255,255)

        else:
            output = frame.copy()
            text = "No poster"
            color = (0,0,255)

        cv2.putText(output, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("AR Overlay (Phone Cam)", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()