import sys
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout

from ui.MainWindow import Ui_MainWindow

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import cm
import matplotlib.pyplot as plt

import random

from ronchigram import Ronchigram



class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # a figure instance to plot on
        self.Pfigure = plt.figure()
        self.Rfigure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.Ronchcanvas = FigureCanvas(self.Rfigure)
        self.Phasecanvas = FigureCanvas(self.Pfigure)

        # set the layout
        self.Ronch_Layout.addWidget(self.Ronchcanvas)
        self.setLayout(self.Ronch_Layout)
        self.Phase_Layout.addWidget(self.Phasecanvas)
        self.setLayout(self.Phase_Layout)

        self.update_values()

        # set the spinbox  functions
        self.O2_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.A2_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.P3_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.A3_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.O3_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.Q4_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.A4_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.P5_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.R5_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.A5_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.O6_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.R6_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.A6_doubleSpinBox.valueChanged.connect(self.update_plot)

        self.A2_ang_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.P3_ang_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.A3_ang_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.Q4_ang_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.A4_ang_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.P5_ang_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.R5_ang_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.A5_ang_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.R6_ang_doubleSpinBox.valueChanged.connect(self.update_plot)
        self.A6_ang_doubleSpinBox.valueChanged.connect(self.update_plot)

        self.acc_spinBox.setValue(ronch.acc)
        self.CLApt_spinBox.setValue(int(ronch.cl_rad*1e3))
        #self.imdim_
        self.simdim_spinBox.setValue(int(ronch.simdim*1e3))

        self.acc_spinBox.valueChanged.connect(self.update_plot)
        self.CLApt_spinBox.valueChanged.connect(self.update_plot)
        #self.imdim_spinBox.valueChanged.connect(self.update_plot)
        self.simdim_spinBox.valueChanged.connect(self.update_plot)

        self.update_plot()


    def plot_phase(self):
        ''' plot the phase '''

        # instead of ax.hold(False)
        self.Pfigure.clear()

        # create an axis
        ax = self.Pfigure.add_subplot(111)

        # discards the old graph
        # ax.hold(False) # deprecated, see above

        # plot data
        e = ronch.simdim * 1e3
        ronch.calc_chi()
        ax.imshow(np.mod(ronch.chi, 2*np.pi),
                  extent=[-e, e, -e, e])

        # refresh canvas
        self.Phasecanvas.draw()

    def plot_ronchi(self):
        ''' plot ronchigram '''
        # instead of ax.hold(False)
        self.Rfigure.clear()

        # create an axis
        ax = self.Rfigure.add_subplot(111)

        # discards the old graph
        # ax.hold(False) # deprecated, see above

        # plot data
        e = ronch.simdim * 1e3
        ronch.calc_ronchigram()
        ax.imshow(ronch.ronchigram, extent=[-e, e, -e, e], cmap=cm.Greys)

        # refresh canvas
        self.Ronchcanvas.draw()

    def update_values(self):
        ''' update the spinbox values'''
        self.O2_doubleSpinBox.setValue(ronch.aberrations.Cnm[0]*1e9)
        self.A2_doubleSpinBox.setValue(ronch.aberrations.Cnm[1]*1e9)
        self.P3_doubleSpinBox.setValue(ronch.aberrations.Cnm[2]*1e9)
        self.A3_doubleSpinBox.setValue(ronch.aberrations.Cnm[3]*1e9)
        self.O3_doubleSpinBox.setValue(ronch.aberrations.Cnm[4]*1e6)
        self.Q4_doubleSpinBox.setValue(ronch.aberrations.Cnm[5]*1e6)
        self.A4_doubleSpinBox.setValue(ronch.aberrations.Cnm[6]*1e6)
        self.P5_doubleSpinBox.setValue(ronch.aberrations.Cnm[7]*1e6)
        self.R5_doubleSpinBox.setValue(ronch.aberrations.Cnm[8]*1e6)
        self.A5_doubleSpinBox.setValue(ronch.aberrations.Cnm[9]*1e3)
        self.O6_doubleSpinBox.setValue(ronch.aberrations.Cnm[10]*1e3)
        self.R6_doubleSpinBox.setValue(ronch.aberrations.Cnm[11]*1e3)
        self.A6_doubleSpinBox.setValue(ronch.aberrations.Cnm[12]*1e3)

        self.A2_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[1]))
        self.P3_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[2]))
        self.A3_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[3]))
        self.Q4_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[5]))
        self.A4_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[6]))
        self.P5_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[7]))
        self.R5_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[8]))
        self.A5_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[9]))
        self.R6_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[11]))
        self.A6_ang_doubleSpinBox.setValue(np.rad2deg(ronch.aberrations.phinm[12]))

    def get_values(self):
        ronch.aberrations.Cnm[0] = self.O2_doubleSpinBox.value() / 1e9
        ronch.aberrations.Cnm[1] = self.A2_doubleSpinBox.value() / 1e9
        ronch.aberrations.Cnm[2] = self.P3_doubleSpinBox.value() / 1e9
        ronch.aberrations.Cnm[3] = self.A3_doubleSpinBox.value() / 1e9
        ronch.aberrations.Cnm[4] = self.O3_doubleSpinBox.value() / 1e6
        ronch.aberrations.Cnm[5] = self.Q4_doubleSpinBox.value() / 1e6
        ronch.aberrations.Cnm[6] = self.A4_doubleSpinBox.value() / 1e6
        ronch.aberrations.Cnm[7] = self.P5_doubleSpinBox.value() / 1e6
        ronch.aberrations.Cnm[8] = self.R5_doubleSpinBox.value() / 1e6
        ronch.aberrations.Cnm[9] = self.A5_doubleSpinBox.value() / 1e3
        ronch.aberrations.Cnm[10] = self.O6_doubleSpinBox.value() / 1e3
        ronch.aberrations.Cnm[11] = self.R6_doubleSpinBox.value() / 1e3
        ronch.aberrations.Cnm[12] = self.A6_doubleSpinBox.value() / 1e3

        ronch.aberrations.phinm[1] = np.deg2rad(self.A2_ang_doubleSpinBox.value())
        ronch.aberrations.phinm[2] = np.deg2rad(self.P3_ang_doubleSpinBox.value())
        ronch.aberrations.phinm[3] = np.deg2rad(self.A3_ang_doubleSpinBox.value())
        ronch.aberrations.phinm[5] = np.deg2rad(self.Q4_ang_doubleSpinBox.value())
        ronch.aberrations.phinm[6] = np.deg2rad(self.A4_ang_doubleSpinBox.value())
        ronch.aberrations.phinm[7] = np.deg2rad(self.P5_ang_doubleSpinBox.value())
        ronch.aberrations.phinm[8] = np.deg2rad(self.R5_ang_doubleSpinBox.value())
        ronch.aberrations.phinm[9] = np.deg2rad(self.A5_ang_doubleSpinBox.value())
        ronch.aberrations.phinm[11] = np.deg2rad(self.R6_ang_doubleSpinBox.value())
        ronch.aberrations.phinm[12] = np.deg2rad(self.A6_ang_doubleSpinBox.value())

        ronch.acc = self.acc_spinBox.value()
        ronch.cl_rad = self.CLApt_spinBox.value() / 1e3
        # ronch.imdim = int(self.imdim_spinBox.value())
        ronch.simdim = self.simdim_spinBox.value() / 1e3

        #ronch.calc_ronchigram()
        #self.update_plot()

    def update_ronch(self):
        # print(ronch.simdim)
        ronch.setup()
        ronch.calc_wav(ronch.acc)
        ronch.calc_chi()
        ronch.calc_ronchigram()
        ronch.calc_probe()

    def update_plot(self):
        ''' update the display'''
        self.get_values()
        self.update_ronch()
        self.plot_phase()
        self.plot_ronchi()



if __name__ == '__main__':
    ronch = Ronchigram(300)

    app = QApplication(sys.argv)
    app.setStyle("fusion")
    main = Window()
    main.show()

    sys.exit(app.exec_())