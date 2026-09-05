# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Height-for-width layouts shared by the Config and Tools forms."""

import math
from PySide6.QtCore import QEvent, QRect, QSize, Qt, QTimer
from PySide6.QtWidgets import QLayout, QPushButton


class ResponsiveFieldLayout(QLayout):
    """Lay out existing label/control pairs horizontally, or as readable rows.

    Height-for-width lets the scroll area negotiate vertical space before painting.
    Only geometry changes on resize: widgets, focus and signal connections stay put.
    """

    def __init__(self, parent, pairs, ratios, *, field_ratios=False,
                 trailing=False, spacing=30, column_spacing=None,
                 equal_fields=False, control_stretches=None, wrap_labels=True):
        super().__init__(parent)
        self.wrap_labels = wrap_labels
        self.pairs = pairs
        self.ratios = ratios
        self.field_ratios = field_ratios
        self.trailing = trailing
        self.column_spacing = spacing if column_spacing is None else column_spacing
        self.equal_fields = equal_fields
        self.control_stretches = control_stretches
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)
        if parent is not None:
            parent.installEventFilter(self)
        for label, control in pairs:
            self.addWidget(label)
            self.addWidget(control)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < self.count() else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < self.count() else None

    def eventFilter(self, watched, event):
        # Re-negotiate the form height after a child is shown/hidden or its
        # minimum changes, even when this row's width has not changed.
        if event.type() == QEvent.Type.LayoutRequest:
            watched.updateGeometry()
        return super().eventFilter(watched, event)

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self):
        return True

    @staticmethod
    def _minimum(widget):
        # Explicit minima and compound-control layouts remain authoritative,
        # including controls which used Ignored in the old fixed-width rows.
        return widget.minimumSizeHint().expandedTo(widget.minimumSize()).boundedTo(
            widget.maximumSize()
        )

    def minimumSize(self):
        width = max(self._minimum(w).width() for pair in self.pairs for w in pair)
        if not self.wrap_labels:
            width = (max(self._label_width(label) for label, _ in self.pairs)
                     + self.spacing()
                     + max(self._minimum(control).width() for _, control in self.pairs))
        return QSize(width, max(self._minimum(w).height()
                               for pair in self.pairs for w in pair))

    def sizeHint(self):
        width = self._wide_width()
        return QSize(width, self.heightForWidth(width))

    def _label_width(self, label):
        return max(label.minimumWidth(), label.sizeHint().width())

    def _column_minima(self):
        gap = self.spacing()
        return [self._label_width(label) + gap + self._minimum(control).width()
                for label, control in self.pairs]

    def _wide_width(self):
        minima = self._column_minima()
        gap = self.spacing()
        if self.equal_fields:
            control_width = max(self._minimum(control).width() for _, control in self.pairs)
            return (sum(self._label_width(label) + gap + control_width
                        for label, _ in self.pairs)
                    + self.column_spacing * (len(minima) - 1))
        if self.trailing:
            return sum(minima) + self.column_spacing * (len(minima) - 1)
        prefix = 0
        if self.field_ratios:
            prefix = self._label_width(self.pairs[0][0]) + gap
            minima[0] -= prefix
        unit = max(math.ceil(width / ratio)
                   for width, ratio in zip(minima, self.ratios))
        return prefix + unit * sum(self.ratios) + self.column_spacing * (len(minima) - 1)

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, max(1, width), 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)
        parent = self.parentWidget()
        height = self.heightForWidth(rect.width())
        if parent is not None and getattr(self, "_last_height", None) != height:
            self._last_height = height
            QTimer.singleShot(0, parent.updateGeometry)

    def _arrange(self, rect, *, apply):
        gap = self.spacing()
        row_gap = 12
        wide = rect.width() >= self._wide_width()
        if apply:
            self.parentWidget().setProperty("stacked", not wide)
        shared_label = max(self._label_width(label) for label, _ in self.pairs)

        def place_pair(index, x, y, width, label_width):
            label_item, control_item = self._items[2 * index:2 * index + 2]
            control_min = self._minimum(self.pairs[index][1]).width()
            stacked_label = self.wrap_labels and label_width + gap + control_min > width
            input_width = width if stacked_label else width - label_width - gap
            label_height = label_item.sizeHint().height()
            control_height = (control_item.heightForWidth(input_width)
                              if control_item.hasHeightForWidth()
                              else control_item.sizeHint().height())
            height = (label_height + row_gap + control_height if stacked_label
                      else max(label_height, control_height))
            if apply:
                label_item.setGeometry(QRect(
                    x, y, label_width, label_height if stacked_label else height,
                ))
                control_item.setGeometry(QRect(
                    x if stacked_label else x + label_width + gap,
                    y + label_height + row_gap if stacked_label
                    else y + (height - control_height) // 2,
                    input_width, control_height,
                ))
            return height

        if not wide:
            y = rect.y()
            for index in range(len(self.pairs)):
                y += place_pair(index, rect.x(), y, rect.width(), shared_label)
                y += row_gap
            return y - rect.y() - row_gap

        widths = self._column_minima()
        available = rect.width() - self.column_spacing * (len(widths) - 1)
        if self.equal_fields:
            input_space = available - sum(self._label_width(label) + gap
                                          for label, _ in self.pairs)
            widths = []
            for index, (label, _) in enumerate(self.pairs):
                share = input_space // (len(self.pairs) - index)
                widths.append(self._label_width(label) + gap + share)
                input_space -= share
        elif self.trailing:
            flexible = [i for i, (_, control) in enumerate(self.pairs)
                        if not isinstance(control, QPushButton)]
            weights = (self.control_stretches if self.control_stretches is not None
                       else tuple(int(i in flexible) for i in range(len(widths))))
            flexible = [i for i, weight in enumerate(weights) if weight > 0]
            extra = available - sum(widths)
            remaining_weight = sum(weights)
            for index in flexible:
                share = extra * weights[index] // remaining_weight
                remaining_weight -= weights[index]
                widths[index] += share
                extra -= share
        else:
            prefix = (self._label_width(self.pairs[0][0]) + gap
                      if self.field_ratios else 0)
            remaining = available - prefix
            ratio_sum = sum(self.ratios)
            widths = []
            for ratio in self.ratios:
                width = remaining * ratio // ratio_sum
                widths.append(width)
                remaining -= width
                ratio_sum -= ratio
            widths[0] += prefix
        x = rect.x()
        height = 0
        for index, ((label, _), width) in enumerate(zip(self.pairs, widths)):
            height = max(height, place_pair(
                index, x, rect.y(), width, self._label_width(label)
            ))
            x += width + self.column_spacing
        return height


class ResponsiveFlowLayout(QLayout):
    """Wrap visible controls in order, retaining their minimum usable sizes."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self._stretches = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)
        if parent is not None:
            parent.installEventFilter(self)

    def addItem(self, item):
        self._items.append(item)
        self._stretches.append(0)

    def addWidget(self, widget, stretch=0):
        super().addWidget(widget)
        self._stretches[-1] = stretch

    def stretch(self, index):
        return self._stretches[index]

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < self.count() else None

    def takeAt(self, index):
        if 0 <= index < self.count():
            self._stretches.pop(index)
            return self._items.pop(index)
        return None

    def eventFilter(self, watched, event):
        # Re-negotiate the form height after a child is shown/hidden or its
        # minimum changes, even when this row's width has not changed.
        if event.type() == QEvent.Type.LayoutRequest:
            watched.updateGeometry()
        return super().eventFilter(watched, event)

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self):
        return True

    def _visible(self):
        return [i for i, item in enumerate(self._items) if not item.isEmpty()]

    def _minimum(self, index):
        widget = self._items[index].widget()
        return widget.minimumSizeHint().expandedTo(widget.minimumSize()).boundedTo(
            widget.maximumSize()
        )

    def minimumSize(self):
        sizes = [self._minimum(i) for i in self._visible()]
        return QSize(max((s.width() for s in sizes), default=0),
                     max((s.height() for s in sizes), default=0))

    def sizeHint(self):
        visible = self._visible()
        width = (sum(self._minimum(i).width() for i in visible)
                 + self.spacing() * max(0, len(visible) - 1))
        return QSize(width, self.heightForWidth(width))

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, max(1, width), 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)
        parent = self.parentWidget()
        height = self.heightForWidth(rect.width())
        if parent is not None and getattr(self, "_last_height", None) != height:
            self._last_height = height
            parent.setMinimumHeight(height)
            QTimer.singleShot(0, parent.updateGeometry)

    def _arrange(self, rect, *, apply):
        rows = []
        row = []
        used = 0
        for index in self._visible():
            width = self._minimum(index).width()
            if row and used + self.spacing() + width > rect.width():
                rows.append(row)
                row = []
                used = 0
            used += (self.spacing() if row else 0) + width
            row.append(index)
        if row:
            rows.append(row)
        y = rect.y()
        for row in rows:
            widths = [self._minimum(i).width() for i in row]
            extra = max(0, rect.width() - sum(widths) - self.spacing() * (len(row) - 1))
            weight = sum(self._stretches[i] for i in row)
            for pos, index in enumerate(row):
                if self._stretches[index] and weight:
                    share = extra * self._stretches[index] // weight
                    widths[pos] += share
                    extra -= share
                    weight -= self._stretches[index]
            height = max(self._items[i].sizeHint().height() for i in row)
            x = rect.x()
            for index, width in zip(row, widths):
                item = self._items[index]
                if apply:
                    h = item.sizeHint().height()
                    item.setGeometry(QRect(x, y + (height - h) // 2, width, h))
                x += width + self.spacing()
            y += height + self.spacing()
        return max(0, y - rect.y() - self.spacing())


class ResponsiveSelectorLayout(ResponsiveFlowLayout):
    """Selector, optional name, folder button; keep the folder beside the selector."""

    def sizeHint(self):
        width = self._minimum(0).width()
        if 1 in self._visible():
            width = 2 * max(width, self._minimum(1).width()) + self.spacing()
        width += self.spacing() + self._minimum(2).width()
        return QSize(width, self.heightForWidth(width))

    def minimumSize(self):
        selector_width = self._minimum(0).width() + self.spacing() + self._minimum(2).width()
        name_width = self._minimum(1).width() if 1 in self._visible() else 0
        return QSize(max(selector_width, name_width), super().minimumSize().height())

    def _arrange(self, rect, *, apply):
        visible = self._visible()
        if not visible:
            return 0
        has_name = 1 in visible
        folder_width = self._minimum(2).width()
        gap = self.spacing()
        available = rect.width() - folder_width - gap
        split = has_name and (available - gap) // 2 < max(
            self._minimum(0).width(), self._minimum(1).width()
        )
        height = max(self._items[i].sizeHint().height() for i in (0, 2))
        selector_width = (available - gap) // 2 if has_name and not split else available
        if apply:
            self._items[0].setGeometry(QRect(rect.x(), rect.y(), selector_width, height))
            self._items[2].setGeometry(QRect(rect.right() - folder_width + 1,
                                            rect.y(), folder_width, height))
        if has_name:
            name_height = self._items[1].sizeHint().height()
            if apply:
                self._items[1].setGeometry(QRect(
                    rect.x() if split else rect.x() + selector_width + gap,
                    rect.y() + height + gap if split else rect.y(),
                    rect.width() if split else available - selector_width - gap,
                    name_height,
                ))
            height = height + gap + name_height if split else max(height, name_height)
        return height
