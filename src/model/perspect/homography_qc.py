import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from typing import Optional

class HomographyQC:
    """
    Quality Control for improve query and plot.
    """

    def __init__(self, interactive=True):
        self.interactive = interactive
        self.frame_count = 0
        self.rows = []   # list of dicts, later -> DataFrame

    def inspect(
        self,
        frame_index: int,
        H: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        err: Optional[float] = None,
        image: Optional[np.ndarray] = None,
    ):
        print(f"\n🧪 QC FRAME {frame_index}")
        print(f"→ Homography matrix:\n{H}")
        if err is not None:
            print(f"→ MSRE: {err:.4f}")
        print(f"→ # Points: {len(src)}")

        if image is not None:
            self._show_image(image, src, dst)

        if self.interactive:
            input("Press Enter to continue...")

    def _show_image(self, image, src, dst):
        vis = image.copy()
        h, w = vis.shape[:2]
        src_px = (src * np.array([[w, h]])).astype(int)
        dst_px = (dst * np.array([[w, h]])).astype(int)

        for pt in src_px:
            cv2.circle(vis, tuple(pt), 4, (255, 0, 0), -1)
        for pt in dst_px:
            cv2.circle(vis, tuple(pt), 4, (0, 255, 0), 1)

        cv2.imshow("QC Points", vis)
        cv2.waitKey(1)


    def add(self, frame_id: int,
            H:        np.ndarray | None,
            n_pts:    int,
            coverage: float,
            msre:     float | None,
            inliers:  int | None,
            tag: str  = ""):
        """
        Parameters
        ----------
        frame_id  : integer index or timestamp
        H         : 3×3 np.float32 *after* your normalisation; None if fallback
        n_pts     : #points >= conf_th
        coverage  : convex-hull area in [0,1]
        msre      : mean symmetric reprojection error (None if skipped)
        inliers   : #RANSAC inliers (None if skipped)
        tag       : free string (e.g. "ACCEPTED", "det≈0", etc.)
        """
        row = dict(frame=frame_id, n_pts=n_pts, cover=coverage,
                   msre=msre, inl=inliers, tag=tag)

        if H is not None:
            row.update(
                det        = float(np.linalg.det(H)),
                cond       = float(np.linalg.cond(H)),
                tx         = float(H[0,2]),    # crude translation mags
                ty         = float(H[1,2]),
                h22        = float(H[2,2]),
            )
        else:
            row.update(det=np.nan, cond=np.nan, tx=np.nan,
                       ty=np.nan, h22=np.nan)
        self.rows.append(row)

    def df(self) -> pd.DataFrame:
        """Return a pandas DataFrame with all collected rows."""
        return pd.DataFrame(self.rows)

    def describe(self):
        """Print basic stats grouped by your tag."""
        df = self.df()
        print(df.groupby("tag")[["msre", "det", "cond", "n_pts", "cover"]]
              .describe(percentiles=[.25,.5,.75]))

    def plot_hist(self, col: str, bins: int = 50, xlim=None):
        import matplotlib.pyplot as plt
        df = self.df()
        plt.figure()
        for tag in df.tag.unique():
            sel = df[df.tag == tag][col].dropna()
            plt.hist(sel, bins=bins, alpha=0.5, label=tag, density=True)
        plt.title(f"Histogram of {col}")
        plt.legend(); 
        if xlim: plt.xlim(*xlim)
        plt.show()

    def select_good(self,
                    max_msre   = 0.05,
                    det_range  = (0.5, 2.0),
                    max_cond   = 1e4):
        """
        Return boolean mask of rows that pass all criteria.
        You can tune the defaults interactively.
        """
        df = self.df()
        good =  (df.msre <= max_msre) & \
                (df.det.between(*det_range)) & \
                (df.cond <= max_cond) & \
                (df.n_pts >= 4)
        return good.values
