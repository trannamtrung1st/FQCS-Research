# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_windowuyIAwA.ui'
##
## Created by: Qt User Interface Compiler version 5.15.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(878, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.containerRight = QWidget(self.centralwidget)
        self.containerRight.setObjectName(u"containerRight")
        self.horizontalLayout = QHBoxLayout(self.containerRight)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.btnAdd = QPushButton(self.containerRight)
        self.btnAdd.setObjectName(u"btnAdd")

        self.horizontalLayout.addWidget(self.btnAdd)

        self.btnQuit = QPushButton(self.containerRight)
        self.btnQuit.setObjectName(u"btnQuit")

        self.horizontalLayout.addWidget(self.btnQuit)


        self.gridLayout.addWidget(self.containerRight, 0, 1, 1, 1, Qt.AlignBottom)

        self.containerLef = QWidget(self.centralwidget)
        self.containerLef.setObjectName(u"containerLef")
        self.verticalLayout = QVBoxLayout(self.containerLef)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.lblCounter = QLabel(self.containerLef)
        self.lblCounter.setObjectName(u"lblCounter")
        font = QFont()
        font.setPointSize(18)
        self.lblCounter.setFont(font)

        self.verticalLayout.addWidget(self.lblCounter, 0, Qt.AlignTop)

        self.lblClicked = QLabel(self.containerLef)
        self.lblClicked.setObjectName(u"lblClicked")
        self.lblClicked.setFont(font)

        self.verticalLayout.addWidget(self.lblClicked, 0, Qt.AlignTop)


        self.gridLayout.addWidget(self.containerLef, 0, 0, 1, 1, Qt.AlignTop)

        self.gridLayout.setColumnStretch(0, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 878, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Main Window", None))
        self.btnAdd.setText(QCoreApplication.translate("MainWindow", u"Add counter", None))
        self.btnQuit.setText(QCoreApplication.translate("MainWindow", u"Quit", None))
        self.lblCounter.setText(QCoreApplication.translate("MainWindow", u"Counter: ", None))
        self.lblClicked.setText(QCoreApplication.translate("MainWindow", u"Clicked:", None))
    # retranslateUi

