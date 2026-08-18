# Tkinter desktop UI for the local AI detector.
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ai_detector.detectors import create_detectors, detector_catalog
from ai_detector.engine import DetectorEngine
from ai_detector.formatting import label_text, percent
from ai_detector.models import AnalysisReport
from ai_detector.text import TextExtractionError, load_text_from_path


class DetectorApp(tk.Tk):
    # Create the main window and UI state.
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Detector")
        self.geometry("1040x720")
        self.minsize(860, 600)

        self._queue: queue.Queue[AnalysisReport | Exception] = queue.Queue()
        self._detector_vars: dict[str, tk.BooleanVar] = {}
        self._detector_cards: dict[str, dict[str, tk.StringVar]] = {}
        self._last_report: AnalysisReport | None = None

        self._build_ui()

    # Build the toolbar, text box, four detector boxes, and notes area.
    def _build_ui(self) -> None:
        # Tkinter keeps the first desktop version dependency-light and easy to package.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=(12, 10))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(8, weight=1)

        ttk.Button(toolbar, text="Open File", command=self._open_file).grid(row=0, column=0, padx=(0, 8))
        self.analyze_button = ttk.Button(toolbar, text="Analyze", command=self._start_analysis)
        self.analyze_button.grid(row=0, column=1, padx=(0, 8))
        ttk.Button(toolbar, text="Clear", command=self._clear).grid(row=0, column=2, padx=(0, 16))
        ttk.Button(toolbar, text="Save Report", command=self._save_report).grid(row=0, column=3, padx=(0, 16))

        detector_frame = ttk.LabelFrame(toolbar, text="Detectors")
        detector_frame.grid(row=0, column=4, columnspan=4, sticky="w")
        for index, info in enumerate(detector_catalog()):
            var = tk.BooleanVar(value=True)
            self._detector_vars[info.key] = var
            ttk.Checkbutton(detector_frame, text=info.key, variable=var).grid(row=0, column=index, padx=6, pady=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(toolbar, textvariable=self.status_var, anchor="e").grid(row=0, column=8, sticky="e")

        body = ttk.PanedWindow(self, orient=tk.VERTICAL)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        text_frame = ttk.Frame(body)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.text = tk.Text(text_frame, wrap="word", undo=True, height=14)
        text_scroll = ttk.Scrollbar(text_frame, command=self.text.yview)
        self.text.configure(yscrollcommand=text_scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        text_scroll.grid(row=0, column=1, sticky="ns")
        body.add(text_frame, weight=3)

        result_frame = ttk.Frame(body)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(2, weight=1)

        self.verdict_var = tk.StringVar(value="No analysis yet")
        ttk.Label(result_frame, textvariable=self.verdict_var, font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", pady=(8, 8)
        )

        self.loading_frame = ttk.Frame(result_frame, padding=(8, 6))
        self.loading_var = tk.StringVar(value="")
        ttk.Label(self.loading_frame, textvariable=self.loading_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.loading_bar = ttk.Progressbar(self.loading_frame, mode="indeterminate")
        self.loading_bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.loading_frame.columnconfigure(0, weight=1)

        self.cards_frame = ttk.Frame(result_frame)
        self.cards_frame.grid(row=2, column=0, sticky="nsew")
        self.cards_frame.columnconfigure(0, weight=1)
        self.cards_frame.columnconfigure(1, weight=1)
        for index, info in enumerate(detector_catalog()):
            self._build_detector_card(self.cards_frame, info, index)

        note_frame = ttk.Frame(result_frame)
        note_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        note_frame.columnconfigure(0, weight=1)
        self.notes_var = tk.StringVar(value="Results are advisory and should be reviewed by a human.")
        ttk.Label(note_frame, textvariable=self.notes_var, wraplength=980).grid(row=0, column=0, sticky="w")

        body.add(result_frame, weight=2)
        self._reset_detector_cards()

    # Build one fixed detector result box.
    def _build_detector_card(self, parent: ttk.Frame, info, index: int) -> None:
        row, column = divmod(index, 2)
        frame = ttk.LabelFrame(parent, text=info.name, padding=10)
        frame.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0 if column == 1 else 6), pady=(0, 8))
        parent.rowconfigure(row, weight=1)
        frame.columnconfigure(1, weight=1)

        values = {"status": tk.StringVar(), "result": tk.StringVar(), "probability": tk.StringVar(), "detail": tk.StringVar()}
        ttk.Label(frame, text="Status").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(frame, textvariable=values["status"]).grid(row=0, column=1, sticky="w")
        ttk.Label(frame, text="Result").grid(row=1, column=0, sticky="w", padx=(0, 10))
        ttk.Label(frame, textvariable=values["result"], font=("Segoe UI", 11, "bold")).grid(row=1, column=1, sticky="w")
        ttk.Label(frame, text="AI likelihood").grid(row=2, column=0, sticky="w", padx=(0, 10))
        ttk.Label(frame, textvariable=values["probability"]).grid(row=2, column=1, sticky="w")
        ttk.Label(frame, textvariable=values["detail"], wraplength=440).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._detector_cards[info.key] = values

    # Load supported local files into the text box.
    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open text document",
            filetypes=[
                ("Supported documents", "*.txt *.md *.markdown *.docx *.pdf"),
                ("Text files", "*.txt *.md *.markdown"),
                ("Word documents", "*.docx"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            content = load_text_from_path(path)
        except TextExtractionError as exc:
            messagebox.showerror("Could not open file", str(exc))
            return
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        self.status_var.set(f"Loaded {Path(path).name}")

    # Validate input and start analysis without blocking the UI.
    def _start_analysis(self) -> None:
        # Run detection in a worker thread so model loading does not freeze the window.
        content = self.text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("No text", "Paste text or open a file first.")
            return
        keys = [key for key, var in self._detector_vars.items() if var.get()]
        if not keys:
            messagebox.showinfo("No detectors", "Select at least one detector.")
            return

        self._set_busy(True)
        self._show_loading_state(keys)
        self.verdict_var.set("Analyzing...")
        self.notes_var.set("Loading detectors can take a while the first time.")

        thread = threading.Thread(target=self._run_analysis, args=(content, keys), daemon=True)
        thread.start()
        self.after(150, self._poll_queue)

    # Run the detector engine in a background thread.
    def _run_analysis(self, content: str, keys: list[str]) -> None:
        try:
            engine = DetectorEngine(create_detectors(keys))
            self._queue.put(engine.analyze(content))
        except Exception as exc:  # pragma: no cover - UI safety net
            self._queue.put(exc)

    # Check whether the background analysis has finished.
    def _poll_queue(self) -> None:
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            self.after(150, self._poll_queue)
            return

        self._set_busy(False)
        self._hide_loading_state()
        if isinstance(item, Exception):
            messagebox.showerror("Analysis failed", str(item))
            self.status_var.set("Analysis failed")
            self.verdict_var.set("Analysis failed")
            return
        self._show_report(item)

    # Display an analysis report in the four detector boxes.
    def _show_report(self, report: AnalysisReport) -> None:
        self._last_report = report
        self.verdict_var.set(
            f"{label_text(report.verdict)} - {report.verdict_detail} "
            f"(consensus: {percent(report.consensus_ai_probability)})"
        )
        self._reset_detector_cards("Not selected")
        for result in report.detector_results:
            self._set_detector_card(result.key, result.status.title(), label_text(result.label), percent(result.ai_probability), result.detail)
        self.notes_var.set(" ".join(report.notes))
        self.status_var.set(f"Analyzed {report.word_count} words in {report.chunk_count} chunk(s)")

    # Reset every detector box to an idle or skipped state.
    def _reset_detector_cards(self, status: str = "Waiting") -> None:
        for values in self._detector_cards.values():
            values["status"].set(status)
            values["result"].set("-")
            values["probability"].set("-")
            values["detail"].set("Run analysis to see this detector output.")

    # Put selected detector boxes into loading state.
    def _show_loading_state(self, keys: list[str]) -> None:
        self.loading_var.set("Analyzing selected detectors...")
        self.loading_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.loading_bar.start(12)
        self._reset_detector_cards("Not selected")
        for key in keys:
            self._set_detector_card(key, "Loading", "Waiting for result", "-", "Model is loading or analyzing this text.")

    # Hide the loading screen after analysis finishes.
    def _hide_loading_state(self) -> None:
        self.loading_bar.stop()
        self.loading_frame.grid_remove()

    # Update one detector result box.
    def _set_detector_card(self, key: str, status: str, result: str, probability: str, detail: str) -> None:
        values = self._detector_cards.get(key)
        if not values:
            return
        values["status"].set(status)
        values["result"].set(result)
        values["probability"].set(probability)
        values["detail"].set(detail or "-")

    # Save the most recent analysis report as JSON.
    def _save_report(self) -> None:
        # JSON is simple for now and can feed a nicer PDF/HTML report later.
        if self._last_report is None:
            messagebox.showinfo("No report", "Run an analysis first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save report",
            defaultextension=".json",
            filetypes=[("JSON report", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self._last_report.as_dict(), indent=2), encoding="utf-8")
        self.status_var.set(f"Saved {Path(path).name}")

    # Reset the input and result state.
    def _clear(self) -> None:
        self.text.delete("1.0", tk.END)
        self._hide_loading_state()
        self._reset_detector_cards()
        self._last_report = None
        self.verdict_var.set("No analysis yet")
        self.notes_var.set("Results are advisory and should be reviewed by a human.")
        self.status_var.set("Ready")

    # Toggle controls while analysis is running.
    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.analyze_button.configure(state=state)
        self.status_var.set("Analyzing..." if busy else "Ready")


# Launch the desktop app.
def main() -> int:
    app = DetectorApp()
    app.mainloop()
    return 0


# Start the desktop app when this file is executed directly.
if __name__ == "__main__":
    raise SystemExit(main())
