from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.config import load_config
from app.controller import AppController
from app.tracking import create_backend
from app.ui import MainWindow


def main() -> None:
    config = load_config()
    backend = create_backend(config.tracking)
    controller = AppController(config, backend)
    app = QApplication(sys.argv)
    window = MainWindow(config)
    controller.bind_ui(window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
