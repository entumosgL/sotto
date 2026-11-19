"""Выделение области и разбор хоткеев.

RegionSelector — обычный Qt-виджет: гоняем его событиями QTest и без настоящей мыши.
"""

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sotto.region import MIN_SIDE, RegionSelector  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv[:1])


def drag(selector, start: QPoint, end: QPoint) -> None:
    QTest.mousePress(selector, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(selector, end)
    QTest.mouseRelease(selector, Qt.MouseButton.LeftButton, pos=end)


def test_selects_rectangle(app):
    selector = RegionSelector(QRect(0, 0, 1200, 800))
    selector.show()
    QTest.qWaitForWindowExposed(selector)
    drag(selector, QPoint(100, 120), QPoint(400, 320))
    # конец выделения включается в прямоугольник — так же он и рисуется
    assert selector.selection == (100, 120, 301, 201)


def test_tiny_selection_is_cancelled(app):
    selector = RegionSelector(QRect(0, 0, 1200, 800))
    selector.show()
    QTest.qWaitForWindowExposed(selector)
    drag(selector, QPoint(10, 10), QPoint(10 + MIN_SIDE - 2, 10 + MIN_SIDE - 2))
    assert selector.selection is None


def test_escape_cancels(app):
    selector = RegionSelector(QRect(0, 0, 1200, 800))
    selector.show()
    QTest.qWaitForWindowExposed(selector)
    QTest.keyClick(selector, Qt.Key.Key_Escape)
    assert selector.selection is None
    assert not selector.isVisible()


def test_negative_coordinates_of_left_monitor(app):
    selector = RegionSelector(QRect(-1920, 0, 1920, 1080))
    selector.show()
    QTest.qWaitForWindowExposed(selector)
    drag(selector, QPoint(50, 60), QPoint(250, 260))
    assert selector.selection == (-1870, 60, 201, 201)


def test_hotkey_parsing_windows():
    winsys = pytest.importorskip("sotto.winsys", reason="разбор хоткеев — Windows (RegisterHotKey)")
    mods, vk = winsys.parse_hotkey("CTRL + ALT + A")
    assert mods & winsys.MOD_CONTROL and mods & winsys.MOD_ALT
    assert vk == ord("A")
    assert winsys.parse_hotkey("SUPER + SHIFT + F5") == (winsys.MOD_WIN | winsys.MOD_SHIFT
                                                         | winsys.MOD_NOREPEAT, 0x74)
    assert winsys.parse_hotkey("") is None
