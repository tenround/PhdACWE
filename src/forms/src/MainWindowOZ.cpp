/*
 * File:   MainWindowOZ.cpp
 * Author: olmozavala
 *
 * Created on September 17, 2012, 8:32 AM
 */

#include "MainWindowOZ.h"
#include <iostream>
#include <QKeyEvent>
#include <QDebug>

using namespace std;
MainWindowOZ:: MainWindowOZ(QMainWindow *parent): QMainWindow(parent){
	
	ui.setupUi(this);
	glWidget = new GLWidget();
	setCentralWidget(glWidget);

}

MainWindowOZ::~MainWindowOZ() {
}

bool MainWindowOZ::eventFilter(QObject *o, QEvent *event){
    if(event->type() == QEvent::KeyPress){
        qWarning() << "qWarning Inside MainWindowOZ.cpp: " << o;
    }
	return false;
}

void MainWindowOZ::keyPressEvent(QKeyEvent* event) {
    
    switch (event->key()) {
        case Qt::Key_Escape:
            close();
            break;
        default:
            event->ignore();
            break;
    }
}

/**
 * This function is called when the user clicks on the 'Quit' option or presses the 'q' key.
 * It is called automatically by the Signal-Slot auto-connect mechanism. 
 * The function needs to follow the signature: on_<object name>_<signal name>(<signal parameters>);
 */
void MainWindowOZ::on_actionQuit_activated(){
	close();
}

/**
 * This function is called when the user presses the 'Open File' button. It 
 * starts the image selection method
 */
void MainWindowOZ::on_actionFile_activated(){
	glWidget->SelectImage();
}