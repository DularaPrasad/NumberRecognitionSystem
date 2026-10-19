"""
app.py

GUI for the Handwritten Number Recognition System (HNRS).
Allows users to load an image, select models/settings, run recognition,
and view bounding boxes and digit breakdown.

Run from the project root:
    python src\\gui\\app.py
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk

# Setup paths to import from pipeline
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "integration"))
sys.path.insert(0, str(ROOT / "src" / "segmentation"))
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from pipeline import NumberRecognizer, MODEL_NAMES, SEGMENTERS, annotate


class HNRSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Number Recognition System (HNRS)")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        self.current_image_path = None
        self.original_image = None
        self.recognizer = None

        self._create_widgets()
        self._init_recognizer()

    def _init_recognizer(self):
        """Initialise the recognizer with default options."""
        try:
            self.recognizer = NumberRecognizer(
                model=self.model_var.get(),
                segmenter=self.segmenter_var.get(),
                preprocessing=self.prep_var.get()
            )
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))

    def _create_widgets(self):
        # Top Control Frame
        control_frame = ttk.LabelFrame(self.root, text=" Pipeline Settings ", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Model Selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_var = tk.StringVar(value="resnet")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, values=list(MODEL_NAMES), state="readonly", width=10)
        model_combo.grid(row=0, column=1, padx=5)
        model_combo.bind("<<ComboboxSelected>>", self._update_settings)

        # Segmenter Selection
        ttk.Label(control_frame, text="Segmenter:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.segmenter_var = tk.StringVar(value="unet")
        seg_combo = ttk.Combobox(control_frame, textvariable=self.segmenter_var, values=list(SEGMENTERS), state="readonly", width=18)
        seg_combo.grid(row=0, column=3, padx=5)
        seg_combo.bind("<<ComboboxSelected>>", self._update_settings)

        # Preprocessing Selection
        ttk.Label(control_frame, text="Preprocessing:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.prep_var = tk.StringVar(value="adaptive")
        prep_combo = ttk.Combobox(control_frame, textvariable=self.prep_var, values=("adaptive", "otsu"), state="readonly", width=10)
        prep_combo.grid(row=0, column=5, padx=5)
        prep_combo.bind("<<ComboboxSelected>>", self._update_settings)

        # Load Image Button
        load_btn = ttk.Button(control_frame, text="Load Image...", command=self._load_image)
        load_btn.grid(row=0, column=6, padx=15)

        # Main Content Area (Split: Left = Image, Right = Results & Digits)
        content_frame = ttk.Frame(self.root, padding=10)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left Frame: Image Display
        img_frame = ttk.LabelFrame(content_frame, text=" Preview / Annotated Image ", padding=5)
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.canvas_label = ttk.Label(img_frame, text="No image loaded.", anchor=tk.CENTER)
        self.canvas_label.pack(fill=tk.BOTH, expand=True)

        # Right Frame: Results & Breakdown
        right_frame = ttk.Frame(content_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))

        result_box_frame = ttk.LabelFrame(right_frame, text=" Recognition Result ", padding=10)
        result_box_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.result_var = tk.StringVar(value="—")
        result_label = ttk.Label(result_box_frame, textvariable=self.result_var, font=("Helvetica", 24, "bold"), foreground="#2a78d6")
        result_label.pack(anchor=tk.CENTER, pady=10)

        details_frame = ttk.LabelFrame(right_frame, text=" Digit Breakdown & Confidence ", padding=10)
        details_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Treeview for detailed breakdown
        self.tree = ttk.Treeview(details_frame, columns=("digit", "confidence"), show="headings", height=8)
        self.tree.heading("digit", text="Digit")
        self.tree.heading("confidence", text="Confidence")
        self.tree.column("digit", width=80, anchor=tk.CENTER)
        self.tree.column("confidence", width=150, anchor=tk.CENTER)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _update_settings(self, event=None):
        if self.recognizer:
            try:
                self.recognizer.segmenter = self.segmenter_var.get()
                self.recognizer.preprocessing = self.prep_var.get()
                self.recognizer.set_model(self.model_var.get())
                if self.current_image_path:
                    self._run_recognition()
            except Exception as e:
                messagebox.showerror("Configuration Error", str(e))

    def _load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Number Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self.current_image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Could not read the selected image file.")
                return
            self._run_recognition()

    def _run_recognition(self):
        if self.original_image is None:
            return
        try:
            result = self.recognizer.recognise(self.original_image)
            
            # Update main result text
            if result.number:
                self.result_var.set(result.number)
            else:
                self.result_var.set("No digits found")

            # Update treeview
            for row in self.tree.get_children():
                self.tree.delete(row)
            
            for idx, (dig, conf) in enumerate(zip(result.digits, result.confidences), start=1):
                self.tree.insert("", tk.END, values=(f"Digit {idx}:  {dig}", f"{conf:.1%}"))

            # Update annotated image view
            annotated_img = annotate(self.original_image, result) if result.boxes else self.original_image
            self._display_image(annotated_img)

        except Exception as e:
            messagebox.showerror("Pipeline Error", f"An error occurred during recognition:\n{e}")

    def _display_image(self, cv_img):
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Resize to fit display area while maintaining aspect ratio
        max_w, max_h = 600, 500
        pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        self.photo_img = ImageTk.PhotoImage(pil_img)
        self.canvas_label.config(image=self.photo_img, text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = HNRSGui(root)
    root.mainloop()