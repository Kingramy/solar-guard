# dl/drone_segmenter.py
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dl.radiometric import order_quad_corners

CORE_SHRINK_ALPHA = 0.72
CANVAS_H, CANVAS_W = 256, 384


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PanelUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(64 + 64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(32 + 32, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(16 + 16, 16)

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


def generate_wild_drone_frame(
    X_crops: np.ndarray,
    y_crops: np.ndarray,
    classes: np.ndarray,
    rows: int = 3,
    cols: int = 5,
    angle_deg: Optional[float] = None,
    force_critical: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Synthesizes an aerial drone thermal frame where:
      - Desert ground has hot thermal sand waves (38°C - 54°C) + hot rocks.
      - Each solar panel sits inside an outer aluminum rack border (so the inner
        24x40 solar crop is never overwritten by a synthetic black frame).
    """
    low_res_sand = np.random.uniform(38.0, 54.0, (16, 24)).astype(np.float32)
    temp_map = cv2.resize(low_res_sand, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_CUBIC)
    temp_map += np.random.normal(0, 0.8, (CANVAS_H, CANVAS_W)).astype(np.float32)

    visual_img = np.clip((temp_map - 30.0) * 4.5, 20, 145).astype(np.uint8)
    for _ in range(np.random.randint(4, 10)):
        rx, ry = np.random.randint(10, CANVAS_W - 10), np.random.randint(10, CANVAS_H - 10)
        rad = np.random.randint(3, 8)
        cv2.circle(visual_img, (rx, ry), rad, int(np.random.randint(110, 175)), -1)
        cv2.circle(temp_map, (rx, ry), rad, float(np.random.uniform(45.0, 56.0)), -1)

    core_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    gt_panels = []

    if angle_deg is None:
        angle_deg = float(np.random.uniform(-20.0, 20.0))

    mod_w, mod_h = 36, 54
    gap_x, gap_y = 8, 12
    total_w = cols * mod_w + (cols - 1) * gap_x
    total_h = rows * mod_h + (rows - 1) * gap_y
    start_x = (CANVAS_W - total_w) / 2.0
    start_y = (CANVAS_H - total_h) / 2.0

    cx_canvas, cy_canvas = CANVAS_W / 2.0, CANVAS_H / 2.0
    M_rot = cv2.getRotationMatrix2D((cx_canvas, cy_canvas), angle_deg, 1.0)

    temp_ranges = {
        "No-Anomaly": (39.0, 44.0),
        "Soiling": (46.0, 54.0),
        "Offline-Module": (62.0, 76.0),
        "Diode": (64.0, 78.0),
        "Cell": (75.0, 96.0),
        "Hot-Spot": (85.0, 105.0),
    }
    healthy_idx = list(classes).index("No-Anomaly")

    for r in range(rows):
        for c in range(cols):
            x0 = start_x + c * (mod_w + gap_x)
            y0 = start_y + r * (mod_h + gap_y)

            local_quad = np.array([
                [x0, y0],
                [x0 + mod_w, y0],
                [x0 + mod_w, y0 + mod_h],
                [x0, y0 + mod_h]
            ], dtype=np.float32)

            ones = np.ones((4, 1), dtype=np.float32)
            pts_homo = np.hstack([local_quad, ones])
            rot_quad = order_quad_corners((M_rot @ pts_homo.T).T.astype(np.float32))
            centroid = np.mean(rot_quad, axis=0, keepdims=True)

            # 1. Draw outer aluminum mounting frame (1.08x outside the solar crop)
            outer_frame_quad = centroid + 1.08 * (rot_quad - centroid)
            cv2.fillConvexPoly(visual_img, outer_frame_quad.astype(np.int32), 25)

            # 2. Deterministically select panel class for reproducible demos
            if force_critical:
                if r == 0 and c == cols - 1:
                    target_cls = "No-Anomaly"
                    override_spike = 95.5  # Demo Step 5: Healthy visual crop with 95.5°C spike
                elif r == 1 and c == 2:
                    target_cls = "Cell"
                    override_spike = None
                elif r == 2 and c == 1:
                    target_cls = "Diode"
                    override_spike = None
                elif r == 2 and c == 3:
                    target_cls = "Offline-Module"
                    override_spike = None
                else:
                    target_cls = "No-Anomaly"
                    override_spike = None
            else:
                if (r == 0 and c == 1) or (r == 1 and c == 4):
                    target_cls = "Soiling"
                elif r == 2 and c in [1, 2, 3]:
                    target_cls = "Offline-Module"
                elif r == 1 and c == 2:
                    target_cls = "Diode"
                else:
                    target_cls = "No-Anomaly"
                override_spike = None

            cls_id = list(classes).index(target_cls) if target_cls in classes else healthy_idx
            matching = np.where(y_crops == cls_id)[0]
            crop = X_crops[np.random.choice(matching)]  # Unmodified (40, 24) RaptorMaps crop!

            # 3. Warp untouched crop into rot_quad
            src_corners = np.array([[0, 0], [23, 0], [23, 39], [0, 39]], dtype=np.float32)
            H_mat = cv2.getPerspectiveTransform(src_corners, rot_quad)
            cv2.warpPerspective(
                crop, H_mat, (CANVAS_W, CANVAS_H),
                dst=visual_img, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
            )

            # 4. Synthesize panel temperature and warp into temp_map
            t_low, t_high = temp_ranges.get(target_cls, (40.0, 50.0))
            base_t = np.random.uniform(t_low, t_high)
            crop_temp = (crop.astype(np.float32) / 255.0) * 10.0 + (base_t - 5.0)
            if override_spike is not None:
                crop_temp[15:25, 7:17] = override_spike
            cv2.warpPerspective(
                crop_temp, H_mat, (CANVAS_W, CANVAS_H),
                dst=temp_map, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
            )

            # 5. Shrunk Core Polygon (alpha = 0.72) for U-Net segmentation target
            shrunk_quad = centroid + CORE_SHRINK_ALPHA * (rot_quad - centroid)
            cv2.fillConvexPoly(core_mask, shrunk_quad.astype(np.int32), 1.0)

            gt_panels.append({
                "row": r,
                "col": c,
                "polygon": rot_quad.astype(int).tolist(),
                "true_class": target_cls,
                "injected_override": override_spike is not None
            })

    return visual_img, temp_map, core_mask, gt_panels


def detect_tilted_panels(
    visual_img: np.ndarray,
    model: PanelUNet,
    device: torch.device,
    prob_threshold: float = 0.50
) -> Tuple[List[np.ndarray], np.ndarray]:
    model.eval()
    h_orig, w_orig = visual_img.shape[:2]
    if visual_img.ndim == 3 and visual_img.shape[2] == 3:
        visual_gray = cv2.cvtColor(visual_img, cv2.COLOR_BGR2GRAY)
    else:
        visual_gray = visual_img
    resized = cv2.resize(visual_gray, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_LINEAR)

    norm_img = (resized.astype(np.float32) - np.mean(resized, dtype=np.float32)) / (np.std(resized, dtype=np.float32) + 1e-6)
    tensor_in = torch.from_numpy(norm_img).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_in)
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()

    if (h_orig, w_orig) != (CANVAS_H, CANVAS_W):
        prob_map = cv2.resize(prob_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    binary_mask = (prob_map >= prob_threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_quads = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 250:
            continue
        (cx, cy), (w_core, h_core), angle = cv2.minAreaRect(cnt)
        w_full = w_core / CORE_SHRINK_ALPHA
        h_full = h_core / CORE_SHRINK_ALPHA
        box_pts = cv2.boxPoints(((cx, cy), (w_full, h_full), angle))
        detected_quads.append(order_quad_corners(box_pts))

    detected_quads.sort(key=lambda q: (int(np.mean(q[:, 1]) // 45), int(np.mean(q[:, 0]) // 35)))
    return detected_quads, prob_map


class SyntheticDroneDataset(Dataset):
    def __init__(self, imgs: np.ndarray, masks: np.ndarray):
        self.imgs = imgs
        self.masks = masks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        img = (img - np.mean(img, dtype=np.float32)) / (np.std(img, dtype=np.float32) + 1e-6)
        mask = self.masks[idx].astype(np.float32)
        return torch.from_numpy(img).float().unsqueeze(0), torch.from_numpy(mask).float().unsqueeze(0)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Retraining Drone PanelUNet with Outer Frame Geometry on: {device}")

    data = np.load("data/processed/ism.npz", allow_pickle=True)
    X_tr, y_tr = data["X"][data["tr"]], data["y"][data["tr"]]
    X_te, y_te = data["X"][data["te"]], data["y"][data["te"]]
    classes = data["classes"]

    train_imgs, train_masks = [], []
    for _ in range(200):
        r = int(np.random.choice([2, 3]))
        c = int(np.random.choice([4, 5, 6]))
        v_img, _, c_mask, _ = generate_wild_drone_frame(X_tr, y_tr, classes, rows=r, cols=c)
        train_imgs.append(v_img)
        train_masks.append(c_mask)

    val_imgs, val_masks = [], []
    for _ in range(40):
        v_img, _, c_mask, _ = generate_wild_drone_frame(X_te, y_te, classes, rows=3, cols=5)
        val_imgs.append(v_img)
        val_masks.append(c_mask)

    train_loader = DataLoader(SyntheticDroneDataset(np.stack(train_imgs), np.stack(train_masks)), batch_size=16, shuffle=True)
    val_loader = DataLoader(SyntheticDroneDataset(np.stack(val_imgs), np.stack(val_masks)), batch_size=16, shuffle=False)

    model = PanelUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    best_dice = 0.0
    for epoch in range(1, 16):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = bce_dice_loss(model(x_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                probs = (torch.sigmoid(model(x_b)) >= 0.5).float()
                inter = (probs * y_b).sum(dim=(1, 2, 3))
                union = probs.sum(dim=(1, 2, 3)) + y_b.sum(dim=(1, 2, 3))
                val_dice += ((2.0 * inter + 1e-5) / (union + 1e-5)).mean().item()
        val_dice /= len(val_loader)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"state_dict": model.state_dict(), "val_dice": best_dice}, "models/drone_unet.pt")

    print(f"Saved updated PanelUNet (Val Core Dice: {best_dice:.4f})")


if __name__ == "__main__":
    main()