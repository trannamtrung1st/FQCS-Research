# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'detection_screenpPNZza.ui'
##
## Created by: Qt User Interface Compiler version 5.15.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_DetectionScreen(object):
    def setupUi(self, DetectionScreen):
        if not DetectionScreen.objectName():
            DetectionScreen.setObjectName(u"DetectionScreen")
        DetectionScreen.resize(1355, 805)
        DetectionScreen.setAutoFillBackground(False)
        DetectionScreen.setStyleSheet(u"background:#E5E5E5")
        self.verticalLayout = QVBoxLayout(DetectionScreen)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.containerScreen = QWidget(DetectionScreen)
        self.containerScreen.setObjectName(u"containerScreen")
        self.containerScreen.setAutoFillBackground(False)
        self.gridLayout = QGridLayout(self.containerScreen)
        self.gridLayout.setObjectName(u"gridLayout")
        self.screen1 = QLabel(self.containerScreen)
        self.screen1.setObjectName(u"screen1")
        self.screen1.setStyleSheet(u"background-color: #AFF;font-weight:bold;")
        self.screen1.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.screen1, 0, 0, 1, 1)

        self.screen2 = QLabel(self.containerScreen)
        self.screen2.setObjectName(u"screen2")
        self.screen2.setStyleSheet(u"background-color: #AFF;font-weight:bold;")
        self.screen2.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.screen2, 0, 1, 1, 1)

        self.screen4 = QLabel(self.containerScreen)
        self.screen4.setObjectName(u"screen4")
        self.screen4.setStyleSheet(u"background-color: #AFF;font-weight:bold;")
        self.screen4.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.screen4, 1, 1, 1, 1)


        self.verticalLayout.addWidget(self.containerScreen)

        self.containerConfig = QWidget(DetectionScreen)
        self.containerConfig.setObjectName(u"containerConfig")
        self.containerConfig.setAutoFillBackground(False)
        self.containerConfig.setStyleSheet(u"background-color: #EEEEEE")
        self.verticalLayout_2 = QVBoxLayout(self.containerConfig)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.lblTitle = QLabel(self.containerConfig)
        self.lblTitle.setObjectName(u"lblTitle")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.lblTitle.setFont(font)
        self.lblTitle.setStyleSheet(u"text-align:center;\n"
"font-weight:bold;")
        self.lblTitle.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.lblTitle)

        self.containerParam = QWidget(self.containerConfig)
        self.containerParam.setObjectName(u"containerParam")
        self.horizontalLayout = QHBoxLayout(self.containerParam)
        self.horizontalLayout.setSpacing(7)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.containerLeft = QWidget(self.containerParam)
        self.containerLeft.setObjectName(u"containerLeft")
        self.containerLeft.setStyleSheet(u"#containerLeft {\n"
"	border: 1px solid #A5A5A5\n"
"}")
        self.verticalLayout_10 = QVBoxLayout(self.containerLeft)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.groupCbbTemplate = QGroupBox(self.containerLeft)
        self.groupCbbTemplate.setObjectName(u"groupCbbTemplate")
        self.verticalLayout_7 = QVBoxLayout(self.groupCbbTemplate)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(-1, 0, -1, 0)
        self.cbbTemplate_5 = QComboBox(self.groupCbbTemplate)
        self.cbbTemplate_5.addItem("")
        self.cbbTemplate_5.addItem("")
        self.cbbTemplate_5.setObjectName(u"cbbTemplate_5")
        self.cbbTemplate_5.setAutoFillBackground(False)
        self.cbbTemplate_5.setStyleSheet(u"height:22px")
        self.cbbTemplate_5.setFrame(True)

        self.verticalLayout_7.addWidget(self.cbbTemplate_5)


        self.verticalLayout_10.addWidget(self.groupCbbTemplate)

        self.groupCbbTemplate_2 = QGroupBox(self.containerLeft)
        self.groupCbbTemplate_2.setObjectName(u"groupCbbTemplate_2")
        self.verticalLayout_8 = QVBoxLayout(self.groupCbbTemplate_2)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(-1, 0, -1, 0)
        self.cbbTemplate_6 = QComboBox(self.groupCbbTemplate_2)
        self.cbbTemplate_6.addItem("")
        self.cbbTemplate_6.addItem("")
        self.cbbTemplate_6.setObjectName(u"cbbTemplate_6")
        self.cbbTemplate_6.setAutoFillBackground(False)
        self.cbbTemplate_6.setStyleSheet(u"height:22px")
        self.cbbTemplate_6.setFrame(True)

        self.verticalLayout_8.addWidget(self.cbbTemplate_6)


        self.verticalLayout_10.addWidget(self.groupCbbTemplate_2)

        self.groupCbbTemplate_3 = QGroupBox(self.containerLeft)
        self.groupCbbTemplate_3.setObjectName(u"groupCbbTemplate_3")
        self.verticalLayout_9 = QVBoxLayout(self.groupCbbTemplate_3)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(-1, 0, -1, 0)
        self.cbbTemplate_7 = QComboBox(self.groupCbbTemplate_3)
        self.cbbTemplate_7.addItem("")
        self.cbbTemplate_7.addItem("")
        self.cbbTemplate_7.setObjectName(u"cbbTemplate_7")
        self.cbbTemplate_7.setAutoFillBackground(False)
        self.cbbTemplate_7.setStyleSheet(u"height:22px")
        self.cbbTemplate_7.setFrame(True)

        self.verticalLayout_9.addWidget(self.cbbTemplate_7)


        self.verticalLayout_10.addWidget(self.groupCbbTemplate_3)


        self.horizontalLayout.addWidget(self.containerLeft)

        self.containerMid = QWidget(self.containerParam)
        self.containerMid.setObjectName(u"containerMid")
        self.containerMid.setStyleSheet(u"#containerMid {\n"
"	border: 1px solid #A5A5A5\n"
"}")
        self.gridLayout_2 = QGridLayout(self.containerMid)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.groupSliderTemplate = QGroupBox(self.containerMid)
        self.groupSliderTemplate.setObjectName(u"groupSliderTemplate")
        self.verticalLayout_11 = QVBoxLayout(self.groupSliderTemplate)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(-1, 0, -1, 0)
        self.sliderTemplate = QSlider(self.groupSliderTemplate)
        self.sliderTemplate.setObjectName(u"sliderTemplate")
        self.sliderTemplate.setOrientation(Qt.Horizontal)

        self.verticalLayout_11.addWidget(self.sliderTemplate)


        self.gridLayout_2.addWidget(self.groupSliderTemplate, 0, 0, 1, 1)

        self.groupSliderTemplate_3 = QGroupBox(self.containerMid)
        self.groupSliderTemplate_3.setObjectName(u"groupSliderTemplate_3")
        self.verticalLayout_13 = QVBoxLayout(self.groupSliderTemplate_3)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(-1, 0, -1, 0)
        self.sliderTemplate_3 = QSlider(self.groupSliderTemplate_3)
        self.sliderTemplate_3.setObjectName(u"sliderTemplate_3")
        self.sliderTemplate_3.setOrientation(Qt.Horizontal)

        self.verticalLayout_13.addWidget(self.sliderTemplate_3)


        self.gridLayout_2.addWidget(self.groupSliderTemplate_3, 0, 2, 1, 1)

        self.groupInputTemplate = QGroupBox(self.containerMid)
        self.groupInputTemplate.setObjectName(u"groupInputTemplate")
        self.verticalLayout_14 = QVBoxLayout(self.groupInputTemplate)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(-1, 0, -1, 0)
        self.inpTemplate = QLineEdit(self.groupInputTemplate)
        self.inpTemplate.setObjectName(u"inpTemplate")

        self.verticalLayout_14.addWidget(self.inpTemplate)


        self.gridLayout_2.addWidget(self.groupInputTemplate, 1, 0, 1, 1)

        self.groupSliderTemplate_2 = QGroupBox(self.containerMid)
        self.groupSliderTemplate_2.setObjectName(u"groupSliderTemplate_2")
        self.verticalLayout_12 = QVBoxLayout(self.groupSliderTemplate_2)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setContentsMargins(-1, 0, -1, 0)
        self.sliderTemplate_2 = QSlider(self.groupSliderTemplate_2)
        self.sliderTemplate_2.setObjectName(u"sliderTemplate_2")
        self.sliderTemplate_2.setOrientation(Qt.Horizontal)

        self.verticalLayout_12.addWidget(self.sliderTemplate_2)


        self.gridLayout_2.addWidget(self.groupSliderTemplate_2, 0, 1, 1, 1)

        self.groupColorPickerTemplate = QGroupBox(self.containerMid)
        self.groupColorPickerTemplate.setObjectName(u"groupColorPickerTemplate")
        self.horizontalLayout_2 = QHBoxLayout(self.groupColorPickerTemplate)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(-1, 0, -1, 0)
        self.btnColorFromTemplate = QPushButton(self.groupColorPickerTemplate)
        self.btnColorFromTemplate.setObjectName(u"btnColorFromTemplate")
        self.btnColorFromTemplate.setStyleSheet(u"background-color:#F33")

        self.horizontalLayout_2.addWidget(self.btnColorFromTemplate)

        self.lblConnectTemplate = QLabel(self.groupColorPickerTemplate)
        self.lblConnectTemplate.setObjectName(u"lblConnectTemplate")
        self.lblConnectTemplate.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.lblConnectTemplate)

        self.btnColorToTemplate = QPushButton(self.groupColorPickerTemplate)
        self.btnColorToTemplate.setObjectName(u"btnColorToTemplate")
        self.btnColorToTemplate.setStyleSheet(u"background-color:#3F3")

        self.horizontalLayout_2.addWidget(self.btnColorToTemplate)

        self.horizontalLayout_2.setStretch(0, 4)
        self.horizontalLayout_2.setStretch(1, 2)
        self.horizontalLayout_2.setStretch(2, 4)

        self.gridLayout_2.addWidget(self.groupColorPickerTemplate, 2, 0, 1, 1)

        self.groupInputTemplate_2 = QGroupBox(self.containerMid)
        self.groupInputTemplate_2.setObjectName(u"groupInputTemplate_2")
        self.verticalLayout_15 = QVBoxLayout(self.groupInputTemplate_2)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setContentsMargins(-1, 0, -1, 0)
        self.inpTemplate_2 = QLineEdit(self.groupInputTemplate_2)
        self.inpTemplate_2.setObjectName(u"inpTemplate_2")

        self.verticalLayout_15.addWidget(self.inpTemplate_2)


        self.gridLayout_2.addWidget(self.groupInputTemplate_2, 1, 2, 1, 1)

        self.groupSpinTemplate = QGroupBox(self.containerMid)
        self.groupSpinTemplate.setObjectName(u"groupSpinTemplate")
        self.verticalLayout_4 = QVBoxLayout(self.groupSpinTemplate)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(-1, 0, -1, 0)
        self.spinTemplate = QSpinBox(self.groupSpinTemplate)
        self.spinTemplate.setObjectName(u"spinTemplate")

        self.verticalLayout_4.addWidget(self.spinTemplate)


        self.gridLayout_2.addWidget(self.groupSpinTemplate, 1, 1, 1, 1)

        self.groupCbbTemplate_4 = QGroupBox(self.containerMid)
        self.groupCbbTemplate_4.setObjectName(u"groupCbbTemplate_4")
        self.verticalLayout_3 = QVBoxLayout(self.groupCbbTemplate_4)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, 0, -1, 0)
        self.cbbTemplate = QComboBox(self.groupCbbTemplate_4)
        self.cbbTemplate.addItem("")
        self.cbbTemplate.addItem("")
        self.cbbTemplate.setObjectName(u"cbbTemplate")
        self.cbbTemplate.setAutoFillBackground(False)
        self.cbbTemplate.setStyleSheet(u"height:22px")

        self.verticalLayout_3.addWidget(self.cbbTemplate)


        self.gridLayout_2.addWidget(self.groupCbbTemplate_4, 2, 1, 1, 1)


        self.horizontalLayout.addWidget(self.containerMid)

        self.containerRight = QWidget(self.containerParam)
        self.containerRight.setObjectName(u"containerRight")
        self.containerRight.setStyleSheet(u"#containerRight {\n"
"	border: 1px solid #A5A5A5\n"
"}")
        self.verticalLayout_16 = QVBoxLayout(self.containerRight)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.containerVerticalBtn = QWidget(self.containerRight)
        self.containerVerticalBtn.setObjectName(u"containerVerticalBtn")
        self.verticalLayout_17 = QVBoxLayout(self.containerVerticalBtn)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.btnCapture = QPushButton(self.containerVerticalBtn)
        self.btnCapture.setObjectName(u"btnCapture")

        self.verticalLayout_17.addWidget(self.btnCapture, 0, Qt.AlignTop)


        self.verticalLayout_16.addWidget(self.containerVerticalBtn, 0, Qt.AlignTop)

        self.containerNavBtn = QWidget(self.containerRight)
        self.containerNavBtn.setObjectName(u"containerNavBtn")
        self.horizontalLayout_3 = QHBoxLayout(self.containerNavBtn)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.btnTemplate_2 = QPushButton(self.containerNavBtn)
        self.btnTemplate_2.setObjectName(u"btnTemplate_2")

        self.horizontalLayout_3.addWidget(self.btnTemplate_2)

        self.btnTemplate_3 = QPushButton(self.containerNavBtn)
        self.btnTemplate_3.setObjectName(u"btnTemplate_3")

        self.horizontalLayout_3.addWidget(self.btnTemplate_3)


        self.verticalLayout_16.addWidget(self.containerNavBtn, 0, Qt.AlignBottom)


        self.horizontalLayout.addWidget(self.containerRight)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 6)
        self.horizontalLayout.setStretch(2, 2)

        self.verticalLayout_2.addWidget(self.containerParam)

        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 9)

        self.verticalLayout.addWidget(self.containerConfig)

        self.verticalLayout.setStretch(0, 8)
        self.verticalLayout.setStretch(1, 2)

        self.retranslateUi(DetectionScreen)

        QMetaObject.connectSlotsByName(DetectionScreen)
    # setupUi

    def retranslateUi(self, DetectionScreen):
        DetectionScreen.setWindowTitle(QCoreApplication.translate("DetectionScreen", u"Form", None))
        self.screen1.setText(QCoreApplication.translate("DetectionScreen", u"SCREEN", None))
        self.screen2.setText(QCoreApplication.translate("DetectionScreen", u"SCREEN", None))
        self.screen4.setText(QCoreApplication.translate("DetectionScreen", u"SCREEN", None))
        self.lblTitle.setText(QCoreApplication.translate("DetectionScreen", u"THIS IS THE CONFIGURATION TITLE", None))
        self.groupCbbTemplate.setTitle(QCoreApplication.translate("DetectionScreen", u"Label combo box", None))
        self.cbbTemplate_5.setItemText(0, QCoreApplication.translate("DetectionScreen", u"Item 1", None))
        self.cbbTemplate_5.setItemText(1, QCoreApplication.translate("DetectionScreen", u"Item 2", None))

        self.groupCbbTemplate_2.setTitle(QCoreApplication.translate("DetectionScreen", u"Label combo box", None))
        self.cbbTemplate_6.setItemText(0, QCoreApplication.translate("DetectionScreen", u"Item 1", None))
        self.cbbTemplate_6.setItemText(1, QCoreApplication.translate("DetectionScreen", u"Item 2", None))

        self.groupCbbTemplate_3.setTitle(QCoreApplication.translate("DetectionScreen", u"Label combo box", None))
        self.cbbTemplate_7.setItemText(0, QCoreApplication.translate("DetectionScreen", u"Item 1", None))
        self.cbbTemplate_7.setItemText(1, QCoreApplication.translate("DetectionScreen", u"Item 2", None))

        self.groupSliderTemplate.setTitle(QCoreApplication.translate("DetectionScreen", u"Label slider", None))
        self.groupSliderTemplate_3.setTitle(QCoreApplication.translate("DetectionScreen", u"Label slider", None))
        self.groupInputTemplate.setTitle(QCoreApplication.translate("DetectionScreen", u"Label input", None))
        self.inpTemplate.setInputMask("")
        self.inpTemplate.setText("")
        self.inpTemplate.setPlaceholderText(QCoreApplication.translate("DetectionScreen", u"Input something", None))
        self.groupSliderTemplate_2.setTitle(QCoreApplication.translate("DetectionScreen", u"Label slider", None))
        self.groupColorPickerTemplate.setTitle(QCoreApplication.translate("DetectionScreen", u"Label color picker", None))
        self.btnColorFromTemplate.setText("")
        self.lblConnectTemplate.setText(QCoreApplication.translate("DetectionScreen", u"---", None))
        self.btnColorToTemplate.setText("")
        self.groupInputTemplate_2.setTitle(QCoreApplication.translate("DetectionScreen", u"Label input", None))
        self.inpTemplate_2.setInputMask("")
        self.inpTemplate_2.setText("")
        self.inpTemplate_2.setPlaceholderText(QCoreApplication.translate("DetectionScreen", u"Input something", None))
        self.groupSpinTemplate.setTitle(QCoreApplication.translate("DetectionScreen", u"Label spin input", None))
        self.groupCbbTemplate_4.setTitle(QCoreApplication.translate("DetectionScreen", u"Label combo box", None))
        self.cbbTemplate.setItemText(0, QCoreApplication.translate("DetectionScreen", u"Item 1", None))
        self.cbbTemplate.setItemText(1, QCoreApplication.translate("DetectionScreen", u"Item 2", None))

        self.btnCapture.setText(QCoreApplication.translate("DetectionScreen", u"CAPTURE", None))
        self.btnTemplate_2.setText(QCoreApplication.translate("DetectionScreen", u"BACK", None))
        self.btnTemplate_3.setText(QCoreApplication.translate("DetectionScreen", u"NEXT", None))
    # retranslateUi

