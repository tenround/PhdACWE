/* 
 * File:   MainWindowOZ.h
 * Author: olmozavala
 *
 * Created on September 17, 2012, 8:32 AM
 */

#ifndef _MAINWINDOW_H
#define	_MAINWINDOW_H

#include <QtGui/QMainWindow>
#include "GLWidget.h"
#include "ui_MainWindow.h"

class MainWindowOZ : public QMainWindow {
    Q_OBJECT
public:
    MainWindowOZ(QMainWindow *parent = 0);
    virtual ~MainWindowOZ();
    
protected:
    bool eventFilter(QObject *o, QEvent *e);
    void keyPressEvent(QKeyEvent *event);

private slots:
	void on_actionQuit_activated();
	void on_actionFile_activated();

private:
	Ui::MainWindow ui;
	GLWidget *glWidget;

};

#endif	/* _MAINWINDOW_H */
