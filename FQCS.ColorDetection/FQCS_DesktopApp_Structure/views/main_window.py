# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_windowiEZBno.ui'
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
        MainWindow.resize(800, 596)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lblGroup = QWidget(self.centralwidget)
        self.lblGroup.setObjectName(u"lblGroup")
        self.verticalLayout = QVBoxLayout(self.lblGroup)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.lblData = QLabel(self.lblGroup)
        self.lblData.setObjectName(u"lblData")
        font = QFont()
        font.setFamily(u"Arial")
        font.setPointSize(11)
        font.setKerning(True)
        self.lblData.setFont(font)
        self.lblData.setStyleSheet(u"padding: 5%;\n"
"")

        self.verticalLayout.addWidget(self.lblData, 0, Qt.AlignLeft|Qt.AlignTop)


        self.horizontalLayout.addWidget(self.lblGroup)

        self.btnGroup = QWidget(self.centralwidget)
        self.btnGroup.setObjectName(u"btnGroup")
        self.horizontalLayout_2 = QHBoxLayout(self.btnGroup)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btnAdd = QPushButton(self.btnGroup)
        self.btnAdd.setObjectName(u"btnAdd")

        self.horizontalLayout_2.addWidget(self.btnAdd)

        self.btnQuit = QPushButton(self.btnGroup)
        self.btnQuit.setObjectName(u"btnQuit")

        self.horizontalLayout_2.addWidget(self.btnQuit)


        self.horizontalLayout.addWidget(self.btnGroup, 0, Qt.AlignRight|Qt.AlignBottom)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 26))
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Main window", None))
        self.lblData.setText(QCoreApplication.translate("MainWindow", u"Data", None))
        self.btnAdd.setText(QCoreApplication.translate("MainWindow", u"Add data", None))
        self.btnQuit.setText(QCoreApplication.translate("MainWindow", u"Quit", None))
    # retranslateUi

