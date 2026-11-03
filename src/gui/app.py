"""
app.py

GUI for the Handwritten Number Recognition System (HNRS).

Image acquisition (two ways, as required by the brief):
    1. Load Image...                  - open a photo / scan of a number from a file
    2. Build From Digit Folder...     - create the image of a number automatically from a
                                        folder of images of INDIVIDUAL digits
                                        (uses number_builder.py from Task 2)

Settings:
    Model          which trained digit classifier reads the digits (Task 3)
    Segmenter      which trained segmentation network cuts the number into digits (Task 2):
                   U-Net, ResNet-18 FCN, MobileNetV3 LR-ASPP or FC-DenseNet
    Preprocessing  adaptive thresholding (copes with shadows) or Otsu (one global threshold)

Run from the project root:
    python src\\gui\\app.py
"""

import os
import sys
import time
from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# Setup paths to import from the other tasks
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "integration"))
sys.path.insert(0, str(ROOT / "src" / "segmentation"))
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from pipeline import NumberRecognizer, MODEL_NAMES, SEGMENTERS, annotate        # noqa: E402
from number_builder import (DIGIT_FOLDER, build_number, export_digit_folder,      # noqa: E402
                            load_digit_folder)

# Only the four trained segmentation networks are offered in the GUI.
# (The rule-based methods are still in pipeline.py for the evaluation scripts.)
SEGMENTER_LABELS = {
    "unet": "U-Net",
    "resnet": "ResNet-18 FCN",
    "mobilenet": "MobileNetV3 LR-ASPP",
    "fcdensenet": "FC-DenseNet",
}
SEGMENTER_KEYS = [k for k in SEGMENTER_LABELS if k in SEGMENTERS]
KEY_FROM_LABEL = {SEGMENTER_LABELS[k]: k for k in SEGMENTER_KEYS}

# Gap between digits when a number is built from the digit folder.
# Negative = the digits overlap, which is the hard case for segmentation.
SPACING_CHOICES = {
    "Normal (4-10 px gap)": (4, 10),
    "Close (1-3 px gap)": (1, 3),
    "Touching (digits overlap)": (-3, -3),
}

MAX_DIGITS_IN_NUMBER = 12


class HNRSGui:
    MAX_W, MAX_H = 600, 500          # size of the preview area

    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Number Recognition System (HNRS)")
        self.root.geometry("1000x740")
        self.root.minsize(800, 640)

        self.current_image_path = None   # set when an image was loaded from a file
        self.original_image = None       # BGR image currently being recognised
        self.image_source = ""           # text shown in the status bar
        self.build_settings = None       # settings of the last number built from a folder
        self.last_result = None
        self.last_ms = None
        self.recognizer = None

        self._create_widgets()
        self._init_recognizer()

    # ------------------------------------------------------------------ setup
    def _init_recognizer(self):
        """Initialise the recognizer with default options."""
        try:
            self.recognizer = NumberRecognizer(
                model=self.model_var.get(),
                segmenter=self._segmenter_key(),
                preprocessing=self.prep_var.get()
            )
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))

    def _segmenter_key(self):
        return KEY_FROM_LABEL[self.segmenter_var.get()]

    def _create_widgets(self):
        # ---- Pipeline settings -------------------------------------------------
        control_frame = ttk.LabelFrame(self.root, text=" Pipeline Settings ", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_var = tk.StringVar(value="resnet")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                   values=list(MODEL_NAMES), state="readonly", width=10)
        model_combo.grid(row=0, column=1, padx=5)
        model_combo.bind("<<ComboboxSelected>>", self._update_settings)

        ttk.Label(control_frame, text="Segmenter:").grid(row=0, column=2, sticky=tk.W, padx=(15, 5))
        default_seg = SEGMENTER_LABELS["unet"] if "unet" in SEGMENTER_KEYS else SEGMENTER_LABELS[SEGMENTER_KEYS[0]]
        self.segmenter_var = tk.StringVar(value=default_seg)
        seg_combo = ttk.Combobox(control_frame, textvariable=self.segmenter_var,
                                 values=[SEGMENTER_LABELS[k] for k in SEGMENTER_KEYS],
                                 state="readonly", width=22)
        seg_combo.grid(row=0, column=3, padx=5)
        seg_combo.bind("<<ComboboxSelected>>", self._update_settings)

        ttk.Label(control_frame, text="Preprocessing:").grid(row=0, column=4, sticky=tk.W, padx=(15, 5))
        self.prep_var = tk.StringVar(value="adaptive")
        prep_combo = ttk.Combobox(control_frame, textvariable=self.prep_var,
                                  values=("adaptive", "otsu"), state="readonly", width=10)
        prep_combo.grid(row=0, column=5, padx=5)
        prep_combo.bind("<<ComboboxSelected>>", self._update_settings)

        # ---- Image acquisition -------------------------------------------------
        source_frame = ttk.LabelFrame(self.root, text=" Image Source ", padding=10)
        source_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Button(source_frame, text="Load Image...", command=self._load_image).grid(row=0, column=0, padx=5)
        ttk.Button(source_frame, text="Build From Digit Folder...",
                   command=self._open_build_dialog).grid(row=0, column=1, padx=5)
        self.rebuild_btn = ttk.Button(source_frame, text="Rebuild (new digits)",
                                      command=self._rebuild, state=tk.DISABLED)
        self.rebuild_btn.grid(row=0, column=2, padx=5)
        self.save_btn = ttk.Button(source_frame, text="Save Image...",
                                   command=self._save_image, state=tk.DISABLED)
        self.save_btn.grid(row=0, column=3, padx=5)

        ttk.Label(source_frame, text="View:").grid(row=0, column=4, sticky=tk.E, padx=(20, 5))
        self.view_var = tk.StringVar(value="annotated")
        ttk.Radiobutton(source_frame, text="Annotated", variable=self.view_var, value="annotated",
                        command=self._refresh_view).grid(row=0, column=5, padx=2)
        ttk.Radiobutton(source_frame, text="Preprocessed", variable=self.view_var, value="binary",
                        command=self._refresh_view).grid(row=0, column=6, padx=2)

        # ---- Status bar (packed before the content so it stays at the bottom) --
        self.status_var = tk.StringVar(value="Load an image or build a number from a folder of digits.")
        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                           anchor=tk.W, padding=(8, 3))
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # ---- Main content: left = image, right = results -----------------------
        content_frame = ttk.Frame(self.root, padding=10)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        img_frame = ttk.LabelFrame(content_frame, text=" Preview / Annotated Image ", padding=5)
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.canvas_label = ttk.Label(img_frame, text="No image loaded.", anchor=tk.CENTER)
        self.canvas_label.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(content_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))

        result_box_frame = ttk.LabelFrame(right_frame, text=" Recognition Result ", padding=10)
        result_box_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.result_var = tk.StringVar(value="—")
        result_label = ttk.Label(result_box_frame, textvariable=self.result_var,
                                 font=("Helvetica", 24, "bold"), foreground="#2a78d6")
        result_label.pack(anchor=tk.CENTER, pady=10)

        details_frame = ttk.LabelFrame(right_frame, text=" Digit Breakdown & Confidence ", padding=10)
        details_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(details_frame, columns=("digit", "confidence"), show="headings", height=8)
        self.tree.heading("digit", text="Digit")
        self.tree.heading("confidence", text="Confidence")
        self.tree.column("digit", width=80, anchor=tk.CENTER)
        self.tree.column("confidence", width=150, anchor=tk.CENTER)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --------------------------------------------------------------- settings
    def _update_settings(self, event=None):
        if self.recognizer:
            try:
                self.recognizer.segmenter = self._segmenter_key()
                self.recognizer.preprocessing = self.prep_var.get()
                self.recognizer.set_model(self.model_var.get())
                if self.original_image is not None:
                    self._run_recognition()
            except Exception as e:
                messagebox.showerror("Configuration Error", str(e))

    # ------------------------------------------------- image acquisition: file
    def _load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Number Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not read the selected image file.")
                return
            self.current_image_path = file_path
            self.original_image = image
            self.build_settings = None
            self.rebuild_btn.state(["disabled"])
            self.image_source = f"File: {os.path.basename(file_path)}"
            self._run_recognition()

    # ------------------------------------ image acquisition: folder of digits
    def _open_build_dialog(self):
        prev = self.build_settings or {}
        dlg = tk.Toplevel(self.root)
        dlg.title("Build Number From Digit Folder")
        dlg.transient(self.root)
        dlg.resizable(False, False)

        frm = ttk.Frame(dlg, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        number_var = tk.StringVar(value=prev.get("number", "4071"))
        folder_var = tk.StringVar(value=prev.get("folder", str(DIGIT_FOLDER)))
        spacing_var = tk.StringVar(value=prev.get("spacing", next(iter(SPACING_CHOICES))))
        paper_var = tk.BooleanVar(value=prev.get("paper", False))

        ttk.Label(frm, text="Number to build:").grid(row=0, column=0, sticky=tk.W, pady=4)
        number_entry = ttk.Entry(frm, textvariable=number_var, width=16)
        number_entry.grid(row=0, column=1, sticky=tk.W, pady=4)
        ttk.Label(frm, text=f"(digits only, up to {MAX_DIGITS_IN_NUMBER})").grid(
            row=0, column=2, columnspan=2, sticky=tk.W, padx=8)

        ttk.Label(frm, text="Digit folder:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(frm, textvariable=folder_var, width=46).grid(row=1, column=1, columnspan=2,
                                                               sticky=tk.W, pady=4)

        def browse():
            chosen = filedialog.askdirectory(parent=dlg, title="Folder with sub-folders 0-9",
                                             initialdir=folder_var.get() if os.path.isdir(folder_var.get()) else None)
            if chosen:
                folder_var.set(chosen)

        ttk.Button(frm, text="Browse...", command=browse).grid(row=1, column=3, padx=(8, 0))

        ttk.Label(frm, text="The folder needs sub-folders named 0 to 9, each holding images of that digit.",
                  foreground="#555555").grid(row=2, column=0, columnspan=4, sticky=tk.W)

        def create_sample_folder():
            if self._create_sample_folder(dlg):
                folder_var.set(str(DIGIT_FOLDER))

        ttk.Button(frm, text="Create sample folder from MNIST",
                   command=create_sample_folder).grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=(6, 4))

        ttk.Label(frm, text="Spacing:").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(frm, textvariable=spacing_var, values=list(SPACING_CHOICES),
                     state="readonly", width=28).grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=4)

        ttk.Checkbutton(frm, text="Photo-style paper background (dark ink on uneven light paper)",
                        variable=paper_var).grid(row=5, column=0, columnspan=4, sticky=tk.W, pady=4)

        buttons = ttk.Frame(frm)
        buttons.grid(row=6, column=0, columnspan=4, sticky=tk.E, pady=(12, 0))

        def on_build(event=None):
            settings = {"number": number_var.get().strip(), "folder": folder_var.get().strip(),
                        "spacing": spacing_var.get(), "paper": bool(paper_var.get())}
            if self._build_and_show(settings, parent=dlg):
                dlg.destroy()

        ttk.Button(buttons, text="Build", command=on_build).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT)

        dlg.bind("<Return>", on_build)
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        number_entry.focus_set()
        dlg.grab_set()

    def _create_sample_folder(self, parent):
        """Fill the default digit folder with 20 MNIST images of each digit."""
        if os.path.isdir(DIGIT_FOLDER) and any(os.scandir(DIGIT_FOLDER)):
            return True
        try:
            parent.config(cursor="watch")
            self.status_var.set("Creating the digit folder from MNIST (this may download MNIST the first time)...")
            self.root.update_idletasks()
            export_digit_folder()
            return True
        except Exception as e:
            messagebox.showerror("Could not create the digit folder", str(e), parent=parent)
            return False
        finally:
            parent.config(cursor="")

    def _make_number_image(self, number, folder, gap, paper):
        """Pick a random image of each digit from the folder and glue them into one number image."""
        bank = load_digit_folder(folder)
        missing = sorted({ch for ch in number if int(ch) not in bank})
        if missing:
            raise ValueError("The folder has no images for the digit(s): " + ", ".join(missing))
        rng = np.random.default_rng()
        chosen = []
        for ch in number:
            options = bank[int(ch)]
            chosen.append(options[int(rng.integers(len(options)))])
        image, _owner = build_number(chosen, gap=gap, paper=paper, rng=rng)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def _build_and_show(self, settings, parent=None):
        """Build a number image from the digit folder and recognise it. True if it worked."""
        parent = parent or self.root
        number, folder = settings["number"], settings["folder"]
        if not number.isdigit() or not (1 <= len(number) <= MAX_DIGITS_IN_NUMBER):
            messagebox.showerror("Invalid number",
                                 f"Enter 1 to {MAX_DIGITS_IN_NUMBER} digits (0-9) only.", parent=parent)
            return False
        if not os.path.isdir(folder):
            if os.path.abspath(folder) == os.path.abspath(DIGIT_FOLDER):
                if not messagebox.askyesno("Digit folder not found",
                                           "The sample digit folder does not exist yet.\n"
                                           "Create it from MNIST now?", parent=parent):
                    return False
                if not self._create_sample_folder(parent):
                    return False
            else:
                messagebox.showerror("Folder not found", f"Cannot find the folder:\n{folder}", parent=parent)
                return False
        try:
            image = self._make_number_image(number, folder, SPACING_CHOICES[settings["spacing"]],
                                            settings["paper"])
        except FileNotFoundError:
            messagebox.showerror("No digit images",
                                 f"No digit images were found in:\n{folder}\n\n"
                                 "Use 'Create sample folder from MNIST' or choose another folder.",
                                 parent=parent)
            return False
        except Exception as e:
            messagebox.showerror("Could not build the number", str(e), parent=parent)
            return False

        self.current_image_path = None
        self.original_image = image
        self.build_settings = settings
        self.rebuild_btn.state(["!disabled"])
        self.image_source = f"Built from digit folder: {number} ({settings['spacing'].split(' (')[0].lower()} spacing" \
                            f"{', paper look' if settings['paper'] else ''})"
        self._run_recognition()
        return True

    def _rebuild(self):
        if self.build_settings:
            self._build_and_show(self.build_settings)

    def _save_image(self):
        if self.original_image is None:
            return
        path = filedialog.asksaveasfilename(title="Save the current number image",
                                            defaultextension=".png",
                                            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")])
        if path:
            if cv2.imwrite(path, self.original_image):
                self.status_var.set(f"Saved image to {path}")
            else:
                messagebox.showerror("Error", "Could not save the image.")

    # ------------------------------------------------------------ recognition
    def _run_recognition(self):
        if self.original_image is None:
            return
        self.save_btn.state(["!disabled"])
        try:
            start = time.perf_counter()
            result = self.recognizer.recognise(self.original_image)
            self.last_ms = (time.perf_counter() - start) * 1000
            self.last_result = result

            # Update main result text
            self.result_var.set(result.number if result.number else "No digits found")

            # Update treeview
            for row in self.tree.get_children():
                self.tree.delete(row)
            for idx, (dig, conf) in enumerate(zip(result.digits, result.confidences), start=1):
                self.tree.insert("", tk.END, values=(f"Digit {idx}:  {dig}", f"{conf:.1%}"))

            self._refresh_view()
            self._update_status()

        except Exception as e:
            messagebox.showerror("Pipeline Error", f"An error occurred during recognition:\n{e}")

    def _update_status(self):
        parts = [self.image_source,
                 f"Model: {self.model_var.get()}",
                 f"Segmenter: {self.segmenter_var.get()}",
                 f"Preprocessing: {self.prep_var.get()}"]
        if self.last_ms is not None:
            parts.append(f"{self.last_ms:.0f} ms")
        self.status_var.set("   |   ".join(p for p in parts if p))

    # ---------------------------------------------------------------- display
    def _refresh_view(self):
        """Show either the annotated image or the preprocessed (binary) image."""
        if self.original_image is None:
            return
        result = self.last_result
        h, w = self.original_image.shape[:2]
        # Scale to fit the preview; small images (e.g. a built number) are enlarged, not left tiny.
        k = min(self.MAX_W / w, self.MAX_H / h, 8.0)

        if self.view_var.get() == "binary" and result is not None:
            shown = cv2.cvtColor(result.binary, cv2.COLOR_GRAY2BGR)
            shown = cv2.resize(shown, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
        else:
            interp = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
            shown = cv2.resize(self.original_image, None, fx=k, fy=k, interpolation=interp)
            if result is not None and result.boxes:
                boxes = [(int(round(x * k)), int(round(y * k)),
                          max(1, int(round(bw * k))), max(1, int(round(bh * k))))
                         for (x, y, bw, bh) in result.boxes]
                shown = annotate(shown, replace(result, boxes=boxes))
        self._display_image(shown)

    def _display_image(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.photo_img = ImageTk.PhotoImage(pil_img)      # keep a reference so it is not garbage-collected
        self.canvas_label.config(image=self.photo_img, text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = HNRSGui(root)
    root.mainloop()