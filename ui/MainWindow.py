# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1021, 946)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 1011, 491))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 30, 481, 451))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.Phase_Layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.Phase_Layout.setContentsMargins(0, 0, 0, 0)
        self.Phase_Layout.setObjectName("Phase_Layout")
        self.horizontalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_2)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 30, 481, 451))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.Ronch_Layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.Ronch_Layout.setContentsMargins(0, 0, 0, 0)
        self.Ronch_Layout.setObjectName("Ronch_Layout")
        self.horizontalLayout.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 500, 591, 411))
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox_3)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 30, 571, 381))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        self.P5_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.P5_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.P5_not_label.setObjectName("P5_not_label")
        self.gridLayout.addWidget(self.P5_not_label, 8, 1, 1, 1)
        self.A2_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A2_label.setObjectName("A2_label")
        self.gridLayout.addWidget(self.A2_label, 2, 0, 1, 1)
        self.Q4_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Q4_label.setObjectName("Q4_label")
        self.gridLayout.addWidget(self.Q4_label, 6, 0, 1, 1)
        self.A6_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A6_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.A6_not_label.setObjectName("A6_not_label")
        self.gridLayout.addWidget(self.A6_not_label, 13, 1, 1, 1)
        self.O2_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.O2_label.setObjectName("O2_label")
        self.gridLayout.addWidget(self.O2_label, 1, 0, 1, 1)
        self.O3_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.O3_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.O3_not_label.setObjectName("O3_not_label")
        self.gridLayout.addWidget(self.O3_not_label, 5, 1, 1, 1)
        self.R5_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.R5_label.setObjectName("R5_label")
        self.gridLayout.addWidget(self.R5_label, 9, 0, 1, 1)
        self.O2_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.O2_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.O2_not_label.setObjectName("O2_not_label")
        self.gridLayout.addWidget(self.O2_not_label, 1, 1, 1, 1)
        self.A5_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A5_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.A5_not_label.setObjectName("A5_not_label")
        self.gridLayout.addWidget(self.A5_not_label, 10, 1, 1, 1)
        self.A6_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A6_label.setObjectName("A6_label")
        self.gridLayout.addWidget(self.A6_label, 13, 0, 1, 1)
        self.A4_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A4_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.A4_not_label.setObjectName("A4_not_label")
        self.gridLayout.addWidget(self.A4_not_label, 7, 1, 1, 1)
        self.P5_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.P5_label.setObjectName("P5_label")
        self.gridLayout.addWidget(self.P5_label, 8, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 3, 1, 1)
        self.A3_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A3_label.setObjectName("A3_label")
        self.gridLayout.addWidget(self.A3_label, 4, 0, 1, 1)
        self.P3_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.P3_label.setObjectName("P3_label")
        self.gridLayout.addWidget(self.P3_label, 3, 0, 1, 1)
        self.A5_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A5_label.setObjectName("A5_label")
        self.gridLayout.addWidget(self.A5_label, 10, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 4, 1, 1)
        self.Q4_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Q4_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Q4_not_label.setObjectName("Q4_not_label")
        self.gridLayout.addWidget(self.Q4_not_label, 6, 1, 1, 1)
        self.O6_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.O6_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.O6_not_label.setObjectName("O6_not_label")
        self.gridLayout.addWidget(self.O6_not_label, 11, 1, 1, 1)
        self.ang_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.ang_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ang_label.setObjectName("ang_label")
        self.gridLayout.addWidget(self.ang_label, 0, 5, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 2, 1, 1)
        self.A3_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A3_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.A3_not_label.setObjectName("A3_not_label")
        self.gridLayout.addWidget(self.A3_not_label, 4, 1, 1, 1)
        self.P3_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.P3_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.P3_not_label.setObjectName("P3_not_label")
        self.gridLayout.addWidget(self.P3_not_label, 3, 1, 1, 1)
        self.O3_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.O3_label.setObjectName("O3_label")
        self.gridLayout.addWidget(self.O3_label, 5, 0, 1, 1)
        self.R6_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.R6_label.setObjectName("R6_label")
        self.gridLayout.addWidget(self.R6_label, 12, 0, 1, 1)
        self.R6_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.R6_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.R6_not_label.setObjectName("R6_not_label")
        self.gridLayout.addWidget(self.R6_not_label, 12, 1, 1, 1)
        self.O6_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.O6_label.setObjectName("O6_label")
        self.gridLayout.addWidget(self.O6_label, 11, 0, 1, 1)
        self.R5_not_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.R5_not_label.setAlignment(QtCore.Qt.AlignCenter)
        self.R5_not_label.setObjectName("R5_not_label")
        self.gridLayout.addWidget(self.R5_not_label, 9, 1, 1, 1)
        self.A4_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A4_label.setObjectName("A4_label")
        self.gridLayout.addWidget(self.A4_label, 7, 0, 1, 1)
        self.A2_label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.A2_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.A2_label_2.setObjectName("A2_label_2")
        self.gridLayout.addWidget(self.A2_label_2, 2, 1, 1, 1)
        self.O2_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.O2_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.O2_doubleSpinBox.setDecimals(3)
        self.O2_doubleSpinBox.setMinimum(-1000.0)
        self.O2_doubleSpinBox.setMaximum(1000.0)
        self.O2_doubleSpinBox.setObjectName("O2_doubleSpinBox")
        self.gridLayout.addWidget(self.O2_doubleSpinBox, 1, 3, 1, 1)
        self.A2_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A2_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A2_ang_doubleSpinBox.setMinimum(-180.0)
        self.A2_ang_doubleSpinBox.setMaximum(180.0)
        self.A2_ang_doubleSpinBox.setObjectName("A2_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.A2_ang_doubleSpinBox, 2, 5, 1, 1)
        self.A2_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A2_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A2_doubleSpinBox.setDecimals(3)
        self.A2_doubleSpinBox.setMinimum(-1000.0)
        self.A2_doubleSpinBox.setMaximum(1000.0)
        self.A2_doubleSpinBox.setObjectName("A2_doubleSpinBox")
        self.gridLayout.addWidget(self.A2_doubleSpinBox, 2, 3, 1, 1)
        self.A3_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A3_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A3_ang_doubleSpinBox.setMinimum(-180.0)
        self.A3_ang_doubleSpinBox.setMaximum(180.0)
        self.A3_ang_doubleSpinBox.setObjectName("A3_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.A3_ang_doubleSpinBox, 4, 5, 1, 1)
        self.O3_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.O3_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.O3_doubleSpinBox.setDecimals(3)
        self.O3_doubleSpinBox.setMinimum(-1000.0)
        self.O3_doubleSpinBox.setMaximum(1000.0)
        self.O3_doubleSpinBox.setObjectName("O3_doubleSpinBox")
        self.gridLayout.addWidget(self.O3_doubleSpinBox, 5, 3, 1, 1)
        self.O6_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.O6_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.O6_doubleSpinBox.setDecimals(3)
        self.O6_doubleSpinBox.setMinimum(-1000.0)
        self.O6_doubleSpinBox.setMaximum(1000.0)
        self.O6_doubleSpinBox.setObjectName("O6_doubleSpinBox")
        self.gridLayout.addWidget(self.O6_doubleSpinBox, 11, 3, 1, 1)
        self.P3_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.P3_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.P3_ang_doubleSpinBox.setMinimum(-180.0)
        self.P3_ang_doubleSpinBox.setMaximum(180.0)
        self.P3_ang_doubleSpinBox.setObjectName("P3_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.P3_ang_doubleSpinBox, 3, 5, 1, 1)
        self.P3_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.P3_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.P3_doubleSpinBox.setDecimals(3)
        self.P3_doubleSpinBox.setMinimum(-1000.0)
        self.P3_doubleSpinBox.setMaximum(1000.0)
        self.P3_doubleSpinBox.setObjectName("P3_doubleSpinBox")
        self.gridLayout.addWidget(self.P3_doubleSpinBox, 3, 3, 1, 1)
        self.P5_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.P5_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.P5_ang_doubleSpinBox.setMinimum(-180.0)
        self.P5_ang_doubleSpinBox.setMaximum(180.0)
        self.P5_ang_doubleSpinBox.setObjectName("P5_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.P5_ang_doubleSpinBox, 8, 5, 1, 1)
        self.A6_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A6_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A6_doubleSpinBox.setDecimals(3)
        self.A6_doubleSpinBox.setMinimum(-1000.0)
        self.A6_doubleSpinBox.setMaximum(1000.0)
        self.A6_doubleSpinBox.setObjectName("A6_doubleSpinBox")
        self.gridLayout.addWidget(self.A6_doubleSpinBox, 13, 3, 1, 1)
        self.A5_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A5_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A5_doubleSpinBox.setDecimals(3)
        self.A5_doubleSpinBox.setMinimum(-1000.0)
        self.A5_doubleSpinBox.setMaximum(1000.0)
        self.A5_doubleSpinBox.setObjectName("A5_doubleSpinBox")
        self.gridLayout.addWidget(self.A5_doubleSpinBox, 10, 3, 1, 1)
        self.A6_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A6_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A6_ang_doubleSpinBox.setMinimum(-180.0)
        self.A6_ang_doubleSpinBox.setMaximum(180.0)
        self.A6_ang_doubleSpinBox.setObjectName("A6_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.A6_ang_doubleSpinBox, 13, 5, 1, 1)
        self.P5_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.P5_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.P5_doubleSpinBox.setDecimals(3)
        self.P5_doubleSpinBox.setMinimum(-1000.0)
        self.P5_doubleSpinBox.setMaximum(1000.0)
        self.P5_doubleSpinBox.setObjectName("P5_doubleSpinBox")
        self.gridLayout.addWidget(self.P5_doubleSpinBox, 8, 3, 1, 1)
        self.Q4_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.Q4_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.Q4_ang_doubleSpinBox.setMinimum(-180.0)
        self.Q4_ang_doubleSpinBox.setMaximum(180.0)
        self.Q4_ang_doubleSpinBox.setObjectName("Q4_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.Q4_ang_doubleSpinBox, 6, 5, 1, 1)
        self.Q4_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.Q4_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.Q4_doubleSpinBox.setDecimals(3)
        self.Q4_doubleSpinBox.setMinimum(-1000.0)
        self.Q4_doubleSpinBox.setMaximum(1000.0)
        self.Q4_doubleSpinBox.setObjectName("Q4_doubleSpinBox")
        self.gridLayout.addWidget(self.Q4_doubleSpinBox, 6, 3, 1, 1)
        self.R5_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.R5_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.R5_ang_doubleSpinBox.setMinimum(-180.0)
        self.R5_ang_doubleSpinBox.setMaximum(180.0)
        self.R5_ang_doubleSpinBox.setObjectName("R5_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.R5_ang_doubleSpinBox, 9, 5, 1, 1)
        self.R6_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.R6_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.R6_ang_doubleSpinBox.setMinimum(-180.0)
        self.R6_ang_doubleSpinBox.setMaximum(180.0)
        self.R6_ang_doubleSpinBox.setObjectName("R6_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.R6_ang_doubleSpinBox, 12, 5, 1, 1)
        self.R5_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.R5_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.R5_doubleSpinBox.setDecimals(3)
        self.R5_doubleSpinBox.setMinimum(-1000.0)
        self.R5_doubleSpinBox.setMaximum(1000.0)
        self.R5_doubleSpinBox.setObjectName("R5_doubleSpinBox")
        self.gridLayout.addWidget(self.R5_doubleSpinBox, 9, 3, 1, 1)
        self.R6_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.R6_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.R6_doubleSpinBox.setDecimals(3)
        self.R6_doubleSpinBox.setMinimum(-1000.0)
        self.R6_doubleSpinBox.setMaximum(1000.0)
        self.R6_doubleSpinBox.setObjectName("R6_doubleSpinBox")
        self.gridLayout.addWidget(self.R6_doubleSpinBox, 12, 3, 1, 1)
        self.A3_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A3_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A3_doubleSpinBox.setDecimals(3)
        self.A3_doubleSpinBox.setMinimum(-1000.0)
        self.A3_doubleSpinBox.setMaximum(1000.0)
        self.A3_doubleSpinBox.setObjectName("A3_doubleSpinBox")
        self.gridLayout.addWidget(self.A3_doubleSpinBox, 4, 3, 1, 1)
        self.A4_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A4_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A4_ang_doubleSpinBox.setMinimum(-180.0)
        self.A4_ang_doubleSpinBox.setMaximum(180.0)
        self.A4_ang_doubleSpinBox.setObjectName("A4_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.A4_ang_doubleSpinBox, 7, 5, 1, 1)
        self.A4_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A4_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A4_doubleSpinBox.setDecimals(3)
        self.A4_doubleSpinBox.setMinimum(-1000.0)
        self.A4_doubleSpinBox.setMaximum(1000.0)
        self.A4_doubleSpinBox.setObjectName("A4_doubleSpinBox")
        self.gridLayout.addWidget(self.A4_doubleSpinBox, 7, 3, 1, 1)
        self.A5_ang_doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.A5_ang_doubleSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.A5_ang_doubleSpinBox.setMinimum(-180.0)
        self.A5_ang_doubleSpinBox.setMaximum(180.0)
        self.A5_ang_doubleSpinBox.setObjectName("A5_ang_doubleSpinBox")
        self.gridLayout.addWidget(self.A5_ang_doubleSpinBox, 10, 5, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setEnabled(False)
        self.groupBox_4.setGeometry(QtCore.QRect(840, 500, 381, 411))
        self.groupBox_4.setObjectName("groupBox_4")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_4)
        self.groupBox_5.setEnabled(False)
        self.groupBox_5.setGeometry(QtCore.QRect(120, 20, 141, 151))
        self.groupBox_5.setObjectName("groupBox_5")
        self.defocus_dial = QtWidgets.QDial(self.groupBox_5)
        self.defocus_dial.setEnabled(False)
        self.defocus_dial.setGeometry(QtCore.QRect(10, 20, 121, 141))
        self.defocus_dial.setObjectName("defocus_dial")
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_4)
        self.groupBox_6.setEnabled(False)
        self.groupBox_6.setGeometry(QtCore.QRect(20, 180, 341, 221))
        self.groupBox_6.setObjectName("groupBox_6")
        self.defx_dial = QtWidgets.QDial(self.groupBox_6)
        self.defx_dial.setEnabled(False)
        self.defx_dial.setGeometry(QtCore.QRect(20, 40, 121, 141))
        self.defx_dial.setObjectName("defx_dial")
        self.defy_dial = QtWidgets.QDial(self.groupBox_6)
        self.defy_dial.setEnabled(False)
        self.defy_dial.setGeometry(QtCore.QRect(200, 40, 121, 141))
        self.defy_dial.setObjectName("defy_dial")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_6)
        self.comboBox.setEnabled(False)
        self.comboBox.setGeometry(QtCore.QRect(180, 10, 151, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label = QtWidgets.QLabel(self.groupBox_6)
        self.label.setEnabled(False)
        self.label.setGeometry(QtCore.QRect(30, 180, 47, 13))
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.groupBox_6)
        self.label_3.setEnabled(False)
        self.label_3.setGeometry(QtCore.QRect(230, 180, 47, 13))
        self.label_3.setObjectName("label_3")
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(610, 500, 211, 321))
        self.groupBox_7.setMinimumSize(QtCore.QSize(75, 0))
        self.groupBox_7.setObjectName("groupBox_7")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox_7)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 30, 185, 261))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_5.setMinimumSize(QtCore.QSize(75, 0))
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.acc_spinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.acc_spinBox.sizePolicy().hasHeightForWidth())
        self.acc_spinBox.setSizePolicy(sizePolicy)
        self.acc_spinBox.setMinimumSize(QtCore.QSize(75, 0))
        self.acc_spinBox.setMaximum(1000)
        self.acc_spinBox.setSingleStep(20)
        self.acc_spinBox.setObjectName("acc_spinBox")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.acc_spinBox)
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_6.setMinimumSize(QtCore.QSize(75, 0))
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.label_7 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_7.setMinimumSize(QtCore.QSize(75, 0))
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.label_8 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_8.setMinimumSize(QtCore.QSize(75, 0))
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.simdim_spinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.simdim_spinBox.sizePolicy().hasHeightForWidth())
        self.simdim_spinBox.setSizePolicy(sizePolicy)
        self.simdim_spinBox.setMinimumSize(QtCore.QSize(75, 0))
        self.simdim_spinBox.setMaximum(2048)
        self.simdim_spinBox.setObjectName("simdim_spinBox")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.simdim_spinBox)
        self.CLApt_spinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CLApt_spinBox.sizePolicy().hasHeightForWidth())
        self.CLApt_spinBox.setSizePolicy(sizePolicy)
        self.CLApt_spinBox.setMinimumSize(QtCore.QSize(75, 0))
        self.CLApt_spinBox.setMaximum(100)
        self.CLApt_spinBox.setObjectName("CLApt_spinBox")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.CLApt_spinBox)
        self.imdim_comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imdim_comboBox.sizePolicy().hasHeightForWidth())
        self.imdim_comboBox.setSizePolicy(sizePolicy)
        self.imdim_comboBox.setMinimumSize(QtCore.QSize(75, 0))
        self.imdim_comboBox.setObjectName("imdim_comboBox")
        self.imdim_comboBox.addItem("")
        self.imdim_comboBox.addItem("")
        self.imdim_comboBox.addItem("")
        self.imdim_comboBox.addItem("")
        self.imdim_comboBox.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.imdim_comboBox)
        self.label_9 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_9.setMinimumSize(QtCore.QSize(75, 0))
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.simdim_spinBox_2 = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.simdim_spinBox_2.sizePolicy().hasHeightForWidth())
        self.simdim_spinBox_2.setSizePolicy(sizePolicy)
        self.simdim_spinBox_2.setMinimumSize(QtCore.QSize(75, 0))
        self.simdim_spinBox_2.setSuffix("")
        self.simdim_spinBox_2.setMinimum(2)
        self.simdim_spinBox_2.setMaximum(32)
        self.simdim_spinBox_2.setObjectName("simdim_spinBox_2")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.simdim_spinBox_2)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.pushButton_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_2.addWidget(self.pushButton_4)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1021, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "PhasePlate"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Ronchigram"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Aberrations"))
        self.label_2.setText(_translate("MainWindow", "JEOL Notation"))
        self.P5_not_label.setText(_translate("MainWindow", "P5"))
        self.A2_label.setText(_translate("MainWindow", "2-fold Stig"))
        self.Q4_label.setText(_translate("MainWindow", "3rd Order Axial Star"))
        self.A6_not_label.setText(_translate("MainWindow", "A6"))
        self.O2_label.setText(_translate("MainWindow", "Defocus"))
        self.O3_not_label.setText(_translate("MainWindow", "O3"))
        self.R5_label.setText(_translate("MainWindow", "3 Lobe"))
        self.O2_not_label.setText(_translate("MainWindow", "O2"))
        self.A5_not_label.setText(_translate("MainWindow", "A5"))
        self.A6_label.setText(_translate("MainWindow", "6-fold Stig"))
        self.A4_not_label.setText(_translate("MainWindow", "A4"))
        self.P5_label.setText(_translate("MainWindow", "4th Order Axial Coma"))
        self.label_4.setText(_translate("MainWindow", "Magnitude "))
        self.A3_label.setText(_translate("MainWindow", "3-fold Stigmatism"))
        self.P3_label.setText(_translate("MainWindow", "Axial Coma"))
        self.A5_label.setText(_translate("MainWindow", "5-fold Stigmatism"))
        self.Q4_not_label.setText(_translate("MainWindow", "Q4"))
        self.O6_not_label.setText(_translate("MainWindow", "O6"))
        self.ang_label.setText(_translate("MainWindow", "Angle (deg)"))
        self.A3_not_label.setText(_translate("MainWindow", "A3"))
        self.P3_not_label.setText(_translate("MainWindow", "P3"))
        self.O3_label.setText(_translate("MainWindow", "3rd Order Spherical"))
        self.R6_label.setText(_translate("MainWindow", "5th Order Rosette"))
        self.R6_not_label.setText(_translate("MainWindow", "R6"))
        self.O6_label.setText(_translate("MainWindow", "5th Order Spherical"))
        self.R5_not_label.setText(_translate("MainWindow", "R5"))
        self.A4_label.setText(_translate("MainWindow", "4-fold Stigmatism"))
        self.A2_label_2.setText(_translate("MainWindow", "A2"))
        self.O2_doubleSpinBox.setSuffix(_translate("MainWindow", " nm"))
        self.A2_doubleSpinBox.setSuffix(_translate("MainWindow", " nm"))
        self.O3_doubleSpinBox.setSuffix(_translate("MainWindow", " μm"))
        self.O6_doubleSpinBox.setSuffix(_translate("MainWindow", " mm"))
        self.P3_doubleSpinBox.setSuffix(_translate("MainWindow", " nm"))
        self.A6_doubleSpinBox.setSuffix(_translate("MainWindow", " mm"))
        self.A5_doubleSpinBox.setSuffix(_translate("MainWindow", " mm"))
        self.P5_doubleSpinBox.setSuffix(_translate("MainWindow", " μm"))
        self.Q4_doubleSpinBox.setSuffix(_translate("MainWindow", " μm"))
        self.R5_doubleSpinBox.setSuffix(_translate("MainWindow", " μm"))
        self.R6_doubleSpinBox.setSuffix(_translate("MainWindow", " mm"))
        self.A3_doubleSpinBox.setSuffix(_translate("MainWindow", " nm"))
        self.A4_doubleSpinBox.setSuffix(_translate("MainWindow", " μm"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Control"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Focus"))
        self.groupBox_6.setTitle(_translate("MainWindow", "GroupBox"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Defocus"))
        self.comboBox.setItemText(1, _translate("MainWindow", "2-fold Stig"))
        self.label.setText(_translate("MainWindow", "DEF X"))
        self.label_3.setText(_translate("MainWindow", "DEF Y"))
        self.groupBox_7.setTitle(_translate("MainWindow", "GroupBox"))
        self.label_5.setText(_translate("MainWindow", "Accel Volt."))
        self.acc_spinBox.setSuffix(_translate("MainWindow", " kV"))
        self.label_6.setText(_translate("MainWindow", "CL Apt"))
        self.label_7.setText(_translate("MainWindow", "ImDim"))
        self.label_8.setText(_translate("MainWindow", "SimDim"))
        self.simdim_spinBox.setSuffix(_translate("MainWindow", " mrad"))
        self.CLApt_spinBox.setSuffix(_translate("MainWindow", " mrad"))
        self.imdim_comboBox.setItemText(0, _translate("MainWindow", "128"))
        self.imdim_comboBox.setItemText(1, _translate("MainWindow", "256"))
        self.imdim_comboBox.setItemText(2, _translate("MainWindow", "512"))
        self.imdim_comboBox.setItemText(3, _translate("MainWindow", "1024"))
        self.imdim_comboBox.setItemText(4, _translate("MainWindow", "2048"))
        self.label_9.setText(_translate("MainWindow", "Zoom Factor"))
        self.simdim_spinBox_2.setPrefix(_translate("MainWindow", "x "))
        self.pushButton_4.setText(_translate("MainWindow", "Randomise"))
        self.pushButton_2.setText(_translate("MainWindow", "Update"))
